"""Crypto news and sentiment.

Aggregates public RSS feeds, scores each headline, and rolls the result up
into a per-coin view the morning briefing can act on.

Three things the previous version got wrong, and why they mattered:

  * The fetch cache was keyed on nothing but time, so asking for the last
    hour of news returned the full 24h cache. The 1h/4h/24h "features" were
    three copies of the same number. Windows are now applied at query time,
    against a single rolling store.
  * Feeds syndicate each other, so one story arriving through four sources
    counted as four-way consensus. Near-duplicate headlines are now merged.
  * Scoring was an unweighted bag of words with no negation handling, so
    "exchange denies hack" scored as bearish as "exchange hacked". Terms are
    weighted, and a negation window flips the sign of what follows.

Sentiment is treated as a *tilt* on a price-driven decision, never as a
standalone entry signal. Headline sentiment decays fast and is widely shown
to be a weak standalone predictor.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("trading_bot")

# Weighted lexicon: magnitude reflects how much the term actually moves a
# crypto tape, not just its dictionary polarity.
BULLISH_TERMS: dict[str, float] = {
    "etf approved": 1.0, "etf approval": 1.0, "spot etf": 0.6,
    "all-time high": 0.8, "record high": 0.8, "breakout": 0.5,
    "institutional inflow": 0.8, "inflows": 0.6, "accumulation": 0.5,
    "partnership": 0.5, "integration": 0.4, "mainnet launch": 0.5,
    "listing": 0.5, "listed": 0.3, "upgrade": 0.4, "halving": 0.6,
    "supply shock": 0.7, "short squeeze": 0.7, "rally": 0.5, "surge": 0.5,
    "soar": 0.5, "jumps": 0.4, "rebound": 0.4, "recovery": 0.4,
    "bullish": 0.6, "adoption": 0.5, "buyback": 0.5, "treasury": 0.3,
    "rate cut": 0.6, "dovish": 0.5, "golden cross": 0.4, "milestone": 0.3,
    "outperform": 0.4, "breakthrough": 0.4, "green light": 0.6,
}

BEARISH_TERMS: dict[str, float] = {
    "hack": -1.0, "hacked": -1.0, "exploit": -0.9, "stolen": -0.8,
    "rug pull": -1.0, "insolvency": -1.0, "bankruptcy": -1.0,
    "delisting": -0.8, "delisted": -0.8, "ban": -0.7, "banned": -0.7,
    "lawsuit": -0.6, "sues": -0.6, "subpoena": -0.6, "investigation": -0.5,
    "crackdown": -0.7, "fraud": -0.8, "ponzi": -0.8, "scam": -0.6,
    "liquidations": -0.7, "liquidated": -0.6, "capitulation": -0.8,
    "selloff": -0.6, "sell-off": -0.6, "crash": -0.8, "plunge": -0.7,
    "plummet": -0.7, "tumble": -0.5, "slump": -0.5, "bearish": -0.6,
    "outflows": -0.6, "death cross": -0.4, "breakdown": -0.5,
    "rate hike": -0.6, "hawkish": -0.5, "recession": -0.5,
    "vulnerability": -0.6, "downgrade": -0.5, "halt withdrawals": -1.0,
    "warns": -0.3, "warning": -0.3, "rejected": -0.6, "denied": -0.4,
}

# Words that invert the polarity of terms appearing shortly after them.
NEGATIONS = {
    "no", "not", "never", "denies", "denied", "deny", "rejects", "refutes",
    "dismisses", "without", "fails", "failed", "unlikely", "halts", "avoids",
}
NEGATION_WINDOW = 4  # tokens
NEGATION_DAMPING = 0.3  # a denial neutralises rather than inverts

COIN_KEYWORDS: dict[str, list[str]] = {
    "BTC": ["bitcoin", "btc", "satoshi"],
    "ETH": ["ethereum", "ether", "eth", "vitalik", "erc-20"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance coin", "bnb"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge"],
    "AVAX": ["avalanche", "avax"],
    "DOT": ["polkadot"],
    "MATIC": ["polygon", "matic"],
    "LINK": ["chainlink"],
    "UNI": ["uniswap"],
    "ATOM": ["cosmos", "atom"],
    "LTC": ["litecoin", "ltc"],
    "ARB": ["arbitrum"],
    "OP": ["optimism"],
    "TON": ["toncoin"],
    "NEAR": ["near protocol"],
    # Added after a screen of OKX's 401 active USDT markets put several
    # names in the top twenty that had no keyword here at all — their
    # sentiment silently read zero, which is indistinguishable from
    # "nothing was written about them".
    "SUI": ["sui network", "mysten"],
    "PEPE": ["pepe coin", "pepecoin"],
    "ZEC": ["zcash"],
    "HYPE": ["hyperliquid"],
    "WLD": ["worldcoin", "world network"],
    "TAO": ["bittensor"],
    "ONDO": ["ondo finance"],
    "BCH": ["bitcoin cash"],
    "FIL": ["filecoin"],
    "AAVE": ["aave"],
    "INJ": ["injective"],
    "TIA": ["celestia"],
    "APT": ["aptos"],
    "SEI": ["sei network"],
    "RNDR": ["render network"],
    "JUP": ["jupiter exchange"],
    "ENA": ["ethena"],
    "STX": ["stacks protocol"],
    "IMX": ["immutable x"],
    "GRT": ["the graph"],
    "SAND": ["the sandbox"],
    "MANA": ["decentraland"],
    "CRV": ["curve finance"],
    "MKR": ["makerdao"],
    "LDO": ["lido finance"],
    "OKB": ["okb"],
}

MARKET_WIDE_TERMS = ("crypto", "cryptocurrency", "digital asset", "blockchain",
                     "altcoin", "stablecoin", "defi")

# Equities are matched by company name as well as ticker, because a
# headline says "Nvidia" far more often than it says "NVDA" — and a bare
# three-letter ticker matches too much prose to be used on its own. Extend
# this from config under news.tickers, or the universe's own symbols will
# still be matched by ticker alone.
TICKER_KEYWORDS: dict[str, list[str]] = {
    "NVDA": ["nvidia"],
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "TSLA": ["tesla"],
    "AMD": ["advanced micro devices", "amd"],
    "COIN": ["coinbase"],
    "MSTR": ["microstrategy", "strategy inc"],
    "GOOGL": ["google", "alphabet"],
    "AMZN": ["amazon"],
    "META": ["meta platforms", "facebook"],
    "SPY": ["s&p 500", "s&p500", "spdr s&p"],
    "QQQ": ["nasdaq 100", "nasdaq-100", "invesco qqq"],
    "GLD": ["gold etf", "spdr gold"],
    "IWM": ["russell 2000"],
    "TLT": ["treasury bond etf", "20+ year treasury"],
}

# Stories that move the whole equity market rather than one name. Mapped
# to SPY, which is the equity market's beta proxy in the way BTC is
# crypto's.
EQUITY_WIDE_TERMS = ("stock market", "wall street", "s&p 500", "nasdaq",
                     "dow jones", "federal reserve", "fed rate", "cpi report",
                     "jobs report", "earnings season")

# Market-wide equity feeds, for tone. Per-ticker news is fetched from the
# templates below, one request per tracked symbol.
EQUITY_FEEDS: list[tuple[str, float]] = [
    ("https://feeds.content.dowjones.io/public/rss/mw_topstories", 0.9),
    ("https://search.cnbc.com/rs/search/combinedcms/view.xml"
     "?partnerId=wrss01&id=100003114", 0.9),
]

# {ticker} is substituted per tracked symbol. Nasdaq's own outbound feed is
# the authoritative one and stays tightly on-topic; Google News is broader
# and noisier, so it is weighted below it.
TICKER_FEED_TEMPLATES: list[tuple[str, float]] = [
    ("https://www.nasdaq.com/feed/rssoutbound?symbol={ticker}", 1.0),
    ("https://news.google.com/rss/search?q={ticker}+stock&hl=en-US"
     "&gl=US&ceid=US:en", 0.6),
]

# Editorial feeds are weighted above social feeds, which are noisier and
# more reflexive.
DEFAULT_FEEDS: list[tuple[str, float]] = [
    ("https://cointelegraph.com/rss", 1.0),
    ("https://www.coindesk.com/arc/outboundfeeds/rss/", 1.0),
    ("https://bitcoinmagazine.com/.rss/full/", 0.8),
    ("https://cryptonews.com/news/feed/", 0.7),
    ("https://decrypt.co/feed", 0.9),
    ("https://www.theblock.co/rss.xml", 1.0),
    ("https://www.reddit.com/r/CryptoCurrency/hot.rss", 0.4),
    ("https://www.reddit.com/r/Bitcoin/hot.rss", 0.4),
]

_TOKEN_RE = re.compile(r"[a-z0-9$%'\-]+")
_HTML_RE = re.compile(r"<[^>]+>")


@dataclass
class NewsItem:
    title: str
    summary: str
    source: str
    url: str
    published: datetime
    source_weight: float = 1.0
    sentiment: float = 0.0
    coins: dict = field(default_factory=dict)

    def age_hours(self, now: datetime) -> float:
        return max(0.0, (now - self.published).total_seconds() / 3600.0)


class NewsSentimentAnalyzer:
    """Rolling store of scored headlines, queryable by coin and window."""

    def __init__(self, config: dict):
        self.config = config
        self.news_config = config.get("news", {})
        self.fetch_interval = int(self.news_config.get("fetch_interval_seconds", 300))
        self.retention_hours = int(self.news_config.get("retention_hours", 48))
        self.half_life_hours = float(self.news_config.get("half_life_hours", 6.0))

        # Equity tickers this run cares about. Nothing is tracked until a
        # universe says so, because a per-ticker feed is one HTTP request
        # per symbol per refresh and fetching news for stocks the bot does
        # not hold is pure cost.
        self.tickers: dict[str, list[str]] = {}
        self.track_tickers(self.news_config.get("tickers") or {})

        self.feeds = self._load_feeds()

        self._items: list[NewsItem] = []
        self._seen: set[str] = set()
        self._last_fetch = 0.0

    def _load_feeds(self) -> list[tuple[str, float]]:
        configured = self.news_config.get("feeds")
        if configured:
            feeds = []
            for entry in configured:
                if isinstance(entry, dict):
                    feeds.append((entry["url"], float(entry.get("weight", 1.0))))
                else:
                    feeds.append((str(entry), 1.0))
            return feeds

        feeds = list(DEFAULT_FEEDS)
        if self.tickers:
            feeds += list(EQUITY_FEEDS)
            for ticker in sorted(self.tickers):
                for template, weight in TICKER_FEED_TEMPLATES:
                    feeds.append((template.format(ticker=ticker), weight))
        return feeds

    def track_tickers(self, tickers) -> None:
        """Register equity tickers to fetch per-symbol news for.

        Accepts a list of tickers or a {ticker: [keywords]} mapping. A
        ticker with no known company name is still matched on the ticker
        itself, which is weaker — a headline says "Nvidia" far more often
        than "NVDA" — so a name in TICKER_KEYWORDS or in config is worth
        having for anything the bot actually trades.
        """
        if isinstance(tickers, dict):
            pairs = {str(k).upper(): list(v) for k, v in tickers.items()}
        else:
            pairs = {str(t).upper(): [] for t in (tickers or [])}

        for ticker, extra in pairs.items():
            keywords = set(TICKER_KEYWORDS.get(ticker, []))
            keywords.update(k.lower() for k in extra)
            keywords.add(ticker.lower())
            self.tickers[ticker] = sorted(keywords)

        if pairs:
            self.feeds = self._load_feeds()

    def track_universe(self, universe) -> None:
        """Track every equity and ETF in a universe."""
        tickers = [i.symbol for i in universe
                   if not i.asset_class.is_crypto and "/" not in i.symbol]
        if tickers:
            self.track_tickers(tickers)

    # ── Ingestion ─────────────────────────────────────────────

    def _session(self):
        """One pooled HTTP session for feed fetches.

        feedparser.parse(url) does its own fetch with no timeout and
        without honouring the environment's proxy settings, so a single
        unreachable feed can block a worker for as long as the OS lets it.
        Fetching the bytes here and handing feedparser a string puts a
        deadline on every request and keeps all traffic on one route.
        """
        import requests

        session = getattr(self, "_http", None)
        if session is not None:
            return session

        session = requests.Session()
        session.trust_env = True
        session.headers.update({
            # Several publishers return 403 to an unadorned client.
            "User-Agent": self.news_config.get(
                "user_agent",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
            ),
            "Accept": "application/rss+xml, application/xml, text/xml, */*",
        })
        bundle = os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
        if os.path.exists(bundle):
            session.verify = bundle
        self._http = session
        return session

    def _fetch_all(self, feedparser) -> list[tuple[str, float, object]]:
        """Every feed, fetched in parallel. Failures come back as None.

        Results are collected as they land rather than in order, so one
        slow publisher costs its own slot and not the whole refresh.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        workers = int(self.news_config.get("fetch_workers", 8))
        timeout = float(self.news_config.get("fetch_timeout_seconds", 15))
        session = self._session()

        def fetch(entry):
            url, weight = entry
            try:
                response = session.get(url, timeout=timeout)
                response.raise_for_status()
                return url, weight, feedparser.parse(response.content)
            except Exception as e:
                logger.debug("Feed failed %s: %s", url, e)
                return url, weight, None

        if workers <= 1 or len(self.feeds) <= 1:
            return [fetch(f) for f in self.feeds]

        results: list[tuple[str, float, object]] = []
        with ThreadPoolExecutor(max_workers=min(workers, len(self.feeds))) as pool:
            futures = [pool.submit(fetch, f) for f in self.feeds]
            try:
                for future in as_completed(futures, timeout=timeout * 2):
                    results.append(future.result())
            except Exception as e:
                # Whatever did land is still usable; a partial tape beats
                # no tape, and the caller counts the misses as failures.
                logger.warning("News fetch incomplete (%s) — %d of %d feeds in",
                               type(e).__name__, len(results), len(self.feeds))
        return results

    def refresh(self, force: bool = False) -> int:
        """Pull new items into the rolling store. Returns items added."""
        now_ts = time.time()
        if not force and self._items and now_ts - self._last_fetch < self.fetch_interval:
            return 0

        try:
            import feedparser
        except ImportError:
            logger.warning("feedparser not installed — news disabled")
            return 0

        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=self.retention_hours)
        added = 0
        failed = 0

        # Fetch concurrently. Tracking a dozen tickers turns one refresh
        # into nearly thirty HTTP requests, and done in series that is
        # minutes of wall clock in the middle of an entry slot — long
        # enough that the prices the plan was built on have moved. The
        # parsing stays on this thread, so nothing below needs a lock.
        for url, weight, feed in self._fetch_all(feedparser):
            try:
                if feed is None or (getattr(feed, "bozo", 0) and not feed.entries):
                    failed += 1
                    continue
                source = (feed.feed.get("title") if hasattr(feed, "feed") else None) or url
                for entry in feed.entries[: int(self.news_config.get("per_feed_limit", 30))]:
                    published = _parse_entry_date(entry) or now
                    if published < cutoff:
                        continue
                    title = (entry.get("title") or "").strip()
                    if not title:
                        continue

                    key = _dedup_key(title)
                    if key in self._seen:
                        continue

                    summary = _HTML_RE.sub(" ", entry.get("summary", entry.get("description", "")))[:600]
                    item = NewsItem(
                        title=title,
                        summary=summary,
                        source=source,
                        url=entry.get("link", ""),
                        published=published,
                        source_weight=weight,
                    )
                    item.sentiment = score_text(f"{title}. {title}. {summary}")
                    item.coins = detect_coins(f"{title} {summary}",
                                             self.tickers)
                    self._items.append(item)
                    self._seen.add(key)
                    added += 1
            except Exception as e:  # feed problems must never stop trading
                failed += 1
                logger.debug("Feed %s failed: %s", url, e)

        self._items = [i for i in self._items if i.published >= cutoff]
        self._seen = {_dedup_key(i.title) for i in self._items}
        self._items.sort(key=lambda i: i.published, reverse=True)
        self._last_fetch = now_ts

        logger.info(
            "News refreshed: +%d new, %d in store, %d/%d feeds failed",
            added, len(self._items), failed, len(self.feeds),
        )
        return added

    def items_in_window(self, hours: float, coin: str | None = None,
                        now: datetime | None = None) -> list[NewsItem]:
        """Items published within the last `hours` — filtered here, not at
        fetch time, so different windows genuinely differ."""
        now = now or datetime.now(timezone.utc)
        cutoff = now - timedelta(hours=hours)
        items = [i for i in self._items if i.published >= cutoff]
        if coin:
            base = coin.split("/")[0].upper()
            items = [i for i in items if base in i.coins]
        return items

    # ── Aggregation ───────────────────────────────────────────

    def sentiment_for(self, symbol: str, hours: float = 24,
                      now: datetime | None = None) -> dict:
        """Recency- and source-weighted sentiment for one coin."""
        now = now or datetime.now(timezone.utc)
        items = self.items_in_window(hours, coin=symbol, now=now)

        if not items:
            return {
                "symbol": symbol, "window_hours": hours, "score": 0.0,
                "articles": 0, "bullish": 0, "bearish": 0, "neutral": 0,
                "consensus": 0.0, "signal_strength": 0.0, "headlines": [],
            }

        weighted_sum = 0.0
        weight_sum = 0.0
        for item in items:
            # Exponential recency decay: a 6h half-life by default, because
            # headline impact on crypto decays within hours.
            decay = 0.5 ** (item.age_hours(now) / max(0.5, self.half_life_hours))
            w = item.source_weight * decay
            weighted_sum += item.sentiment * w
            weight_sum += w

        score = weighted_sum / weight_sum if weight_sum > 0 else 0.0

        bullish = [i for i in items if i.sentiment > 0.2]
        bearish = [i for i in items if i.sentiment < -0.2]
        neutral = [i for i in items if -0.2 <= i.sentiment <= 0.2]

        consensus = abs(len(bullish) - len(bearish)) / len(items)
        # Coverage saturates: 10 independent stories is a signal, 50 is not
        # five times the signal.
        coverage = min(1.0, len(items) / 10.0)
        signal_strength = consensus * coverage * min(1.0, abs(score) * 2)

        headlines = sorted(items, key=lambda i: abs(i.sentiment) * i.source_weight,
                           reverse=True)[:5]

        return {
            "symbol": symbol,
            "window_hours": hours,
            "score": round(score, 4),
            "articles": len(items),
            "bullish": len(bullish),
            "bearish": len(bearish),
            "neutral": len(neutral),
            "consensus": round(consensus, 4),
            "signal_strength": round(signal_strength, 4),
            "headlines": [
                {
                    "title": i.title,
                    "score": round(i.sentiment, 3),
                    "source": i.source,
                    "age_hours": round(i.age_hours(now), 1),
                    "url": i.url,
                }
                for i in headlines
            ],
        }

    def features(self, symbol: str, now: datetime | None = None) -> dict:
        """Sentiment as model features across genuinely distinct windows."""
        now = now or datetime.now(timezone.utc)
        s1 = self.sentiment_for(symbol, 1, now)
        s4 = self.sentiment_for(symbol, 4, now)
        s24 = self.sentiment_for(symbol, 24, now)
        return {
            "news_sentiment_1h": s1["score"],
            "news_sentiment_4h": s4["score"],
            "news_sentiment_24h": s24["score"],
            "news_volume_1h": s1["articles"],
            "news_volume_24h": s24["articles"],
            # Acceleration: a story breaking now against a quiet backdrop.
            "news_momentum": round(s1["score"] - s24["score"], 4),
            "news_signal_strength": s4["signal_strength"],
            "news_consensus": s4["consensus"],
        }

    def market_bias(self, hours: float = 12, now: datetime | None = None) -> dict:
        """Whole-market tone, used to tilt the day's directional budget."""
        now = now or datetime.now(timezone.utc)
        items = self.items_in_window(hours, now=now)
        if not items:
            return {"score": 0.0, "articles": 0, "tone": "neutral"}

        weighted, total_w = 0.0, 0.0
        for item in items:
            decay = 0.5 ** (item.age_hours(now) / max(0.5, self.half_life_hours))
            w = item.source_weight * decay
            weighted += item.sentiment * w
            total_w += w
        score = weighted / total_w if total_w else 0.0

        tone = "bullish" if score > 0.15 else "bearish" if score < -0.15 else "neutral"
        return {"score": round(score, 4), "articles": len(items), "tone": tone}

    def tilt_for(self, symbol: str, now: datetime | None = None) -> float:
        """Directional tilt in [-1, 1] to blend into an edge score.

        Deliberately not a trade trigger on its own: it is scaled by
        consensus so a single loud headline cannot move it far.
        """
        threshold = float(self.news_config.get("signal_threshold", 0.35))
        sent = self.sentiment_for(symbol, float(self.news_config.get("tilt_window_hours", 8)), now)
        if sent["articles"] < int(self.news_config.get("min_articles", 2)):
            return 0.0
        if abs(sent["score"]) < threshold:
            return 0.0
        return round(max(-1.0, min(1.0, sent["score"] * sent["consensus"])), 4)


# ── Scoring helpers ──────────────────────────────────────────

def score_text(text: str) -> float:
    """Score text in [-1, 1] using the weighted lexicon with negation."""
    lowered = text.lower()

    total = 0.0
    hits = 0

    # Multi-word phrases first; they carry the strongest signal and would
    # otherwise be missed by token matching.
    for table in (BULLISH_TERMS, BEARISH_TERMS):
        for term, weight in table.items():
            if " " in term and term in lowered:
                total += weight
                hits += 1

    tokens = _TOKEN_RE.findall(lowered)
    for idx, token in enumerate(tokens):
        weight = BULLISH_TERMS.get(token) or BEARISH_TERMS.get(token)
        if weight is None:
            continue
        window = tokens[max(0, idx - NEGATION_WINDOW):idx]
        if any(w in NEGATIONS for w in window):
            # A denial is closer to neutral than to the opposite claim:
            # "exchange denies hack" is not bullish news, it is a non-event.
            weight = -weight * NEGATION_DAMPING
        total += weight
        hits += 1

    if hits == 0:
        return 0.0
    # Average rather than sum, so a long article is not automatically
    # more extreme than a short one, then squash into range.
    avg = total / hits
    return round(max(-1.0, min(1.0, avg)), 4)


def detect_coins(text: str, tickers: dict[str, list[str]] | None = None
                 ) -> dict[str, int]:
    """Which instruments a story is about.

    Market-wide stories map to the relevant beta proxy — BTC for crypto,
    SPY for equities — because a story about "the stock market" is a story
    about every stock in the book, and dropping it would throw away most
    of the macro tape.
    """
    lowered = text.lower()
    found: dict[str, int] = {}
    for coin, keywords in COIN_KEYWORDS.items():
        for kw in keywords:
            # Word-boundary match so "sol" does not fire on "solution".
            if re.search(rf"\b{re.escape(kw)}\b", lowered):
                found[coin] = found.get(coin, 0) + 1

    for ticker, keywords in (tickers or {}).items():
        for kw in keywords:
            if re.search(rf"\b{re.escape(kw)}\b", lowered):
                found[ticker] = found.get(ticker, 0) + 1

    if not found:
        if any(term in lowered for term in EQUITY_WIDE_TERMS):
            found["SPY"] = 1
        elif any(term in lowered for term in MARKET_WIDE_TERMS):
            found["BTC"] = 1
    return found


def _dedup_key(title: str) -> str:
    """Normalised key that collapses syndicated copies of one story."""
    tokens = _TOKEN_RE.findall(title.lower())
    meaningful = [t for t in tokens if len(t) > 3][:8]
    return " ".join(sorted(meaningful))


def _parse_entry_date(entry) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        parsed = entry.get(key)
        if parsed:
            try:
                return datetime(*parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                continue
    return None
