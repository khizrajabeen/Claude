"""News and sentiment analysis for crypto trading.

Aggregates news from multiple free sources:
  - CoinTelegraph RSS
  - CoinDesk RSS
  - Bitcoin Magazine RSS
  - CryptoPanic API (free tier)
  - Reddit crypto subreddits (via RSS)

Scores sentiment using keyword/phrase analysis and feeds it
into the ML model as features and direct trade signals.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta

import feedparser
import requests

logger = logging.getLogger("trading_bot")

# ── Sentiment word lists (from financial NLP research) ────────

BULLISH_WORDS = {
    "surge", "soar", "rally", "breakout", "bullish", "pump", "moon",
    "all-time high", "ath", "adoption", "partnership", "approval",
    "etf approved", "institutional", "accumulation", "upgrade",
    "buy signal", "golden cross", "recovery", "rebound", "breakthrough",
    "milestone", "record", "growth", "profit", "gain", "support",
    "halving", "supply shock", "shortage", "demand", "inflow",
    "mainstream", "integration", "launch", "listing", "upgrade",
}

BEARISH_WORDS = {
    "crash", "plunge", "dump", "bearish", "selloff", "sell-off",
    "collapse", "liquidation", "hack", "exploit", "vulnerability",
    "ban", "regulation", "crackdown", "lawsuit", "sec", "fraud",
    "ponzi", "scam", "rug pull", "death cross", "breakdown",
    "outflow", "capitulation", "panic", "fear", "risk",
    "recession", "inflation", "rate hike", "delisting", "warning",
    "investigation", "subpoena", "bankruptcy", "insolvency",
}

# Coin-specific keywords for multi-coin support
COIN_KEYWORDS = {
    "BTC": ["bitcoin", "btc", "satoshi"],
    "ETH": ["ethereum", "eth", "vitalik", "erc-20", "layer 2"],
    "SOL": ["solana", "sol"],
    "BNB": ["binance coin", "bnb", "binance"],
    "XRP": ["ripple", "xrp"],
    "ADA": ["cardano", "ada"],
    "DOGE": ["dogecoin", "doge", "elon"],
    "AVAX": ["avalanche", "avax"],
    "DOT": ["polkadot", "dot"],
    "MATIC": ["polygon", "matic"],
    "LINK": ["chainlink", "link"],
    "UNI": ["uniswap", "uni"],
    "ATOM": ["cosmos", "atom"],
    "LTC": ["litecoin", "ltc"],
    "ARB": ["arbitrum", "arb"],
}

# RSS feed sources
RSS_FEEDS = [
    "https://cointelegraph.com/rss",
    "https://www.coindesk.com/arc/outboundfeeds/rss/",
    "https://bitcoinmagazine.com/.rss/full/",
    "https://cryptonews.com/news/feed/",
    "https://www.reddit.com/r/CryptoCurrency/hot.rss",
    "https://www.reddit.com/r/Bitcoin/hot.rss",
]


@dataclass
class NewsItem:
    """A single news article/post."""
    title: str
    summary: str
    source: str
    url: str
    published: datetime
    sentiment_score: float = 0.0  # -1 (bearish) to +1 (bullish)
    relevance: dict = None  # Which coins this relates to

    def __post_init__(self):
        if self.relevance is None:
            self.relevance = {}


class NewsSentimentAnalyzer:
    """Aggregate and analyze crypto news sentiment."""

    def __init__(self, config: dict):
        self.config = config
        self.news_config = config.get("news", {})
        self.cache: list[NewsItem] = []
        self.last_fetch_time = 0.0
        self.fetch_interval = self.news_config.get("fetch_interval_seconds", 300)

    def fetch_news(self, max_age_hours: int = 24) -> list[NewsItem]:
        """Fetch news from all RSS sources."""
        now = time.time()
        if now - self.last_fetch_time < self.fetch_interval and self.cache:
            return self.cache

        cutoff = datetime.utcnow() - timedelta(hours=max_age_hours)
        items = []

        for feed_url in RSS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:20]:  # Latest 20 per source
                    pub_date = self._parse_date(entry)
                    if pub_date and pub_date < cutoff:
                        continue

                    title = entry.get("title", "")
                    summary = entry.get("summary", entry.get("description", ""))
                    # Strip HTML tags from summary
                    summary = re.sub(r"<[^>]+>", "", summary)[:500]

                    item = NewsItem(
                        title=title,
                        summary=summary,
                        source=feed.feed.get("title", feed_url),
                        url=entry.get("link", ""),
                        published=pub_date or datetime.utcnow(),
                    )

                    # Score sentiment
                    item.sentiment_score = self._score_sentiment(title, summary)
                    item.relevance = self._detect_coins(title, summary)

                    items.append(item)

            except Exception as e:
                logger.debug(f"Failed to fetch {feed_url}: {e}")

        # Sort by recency
        items.sort(key=lambda x: x.published, reverse=True)
        self.cache = items
        self.last_fetch_time = now

        logger.info(f"Fetched {len(items)} news items from {len(RSS_FEEDS)} sources")
        return items

    def get_sentiment_for_coin(self, coin: str, hours: int = 24) -> dict:
        """Get aggregated sentiment for a specific coin.

        Returns:
            dict with:
                score: weighted sentiment (-1 to +1)
                article_count: number of relevant articles
                bullish_count: number of bullish articles
                bearish_count: number of bearish articles
                neutral_count: number of neutral articles
                top_headlines: list of most relevant headlines
                signal_strength: confidence in the sentiment signal (0 to 1)
        """
        items = self.fetch_news(max_age_hours=hours)

        # Filter to coin-relevant articles
        coin_base = coin.split("/")[0] if "/" in coin else coin
        relevant = [item for item in items if coin_base in item.relevance]

        if not relevant:
            return {
                "score": 0.0,
                "article_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "neutral_count": 0,
                "top_headlines": [],
                "signal_strength": 0.0,
            }

        # Time-weighted sentiment (recent news matters more)
        now = datetime.utcnow()
        weighted_scores = []
        for item in relevant:
            age_hours = max(0.1, (now - item.published).total_seconds() / 3600)
            # Exponential decay: recent news weighted much higher
            weight = 1.0 / (1 + age_hours * 0.2)
            weighted_scores.append(item.sentiment_score * weight)

        avg_score = sum(weighted_scores) / sum(
            1.0 / (1 + max(0.1, (now - item.published).total_seconds() / 3600) * 0.2)
            for item in relevant
        )

        bullish = [i for i in relevant if i.sentiment_score > 0.2]
        bearish = [i for i in relevant if i.sentiment_score < -0.2]
        neutral = [i for i in relevant if -0.2 <= i.sentiment_score <= 0.2]

        # Signal strength: more articles + stronger consensus = stronger signal
        consensus = abs(len(bullish) - len(bearish)) / max(1, len(relevant))
        volume_factor = min(1.0, len(relevant) / 10)  # Saturates at 10 articles
        signal_strength = consensus * volume_factor

        return {
            "score": round(avg_score, 3),
            "article_count": len(relevant),
            "bullish_count": len(bullish),
            "bearish_count": len(bearish),
            "neutral_count": len(neutral),
            "top_headlines": [
                {"title": i.title, "score": round(i.sentiment_score, 2), "source": i.source}
                for i in sorted(relevant, key=lambda x: abs(x.sentiment_score), reverse=True)[:5]
            ],
            "signal_strength": round(signal_strength, 3),
        }

    def get_sentiment_features(self, coin: str) -> dict:
        """Get sentiment as numerical features for the ML model."""
        sent = self.get_sentiment_for_coin(coin, hours=24)
        sent_4h = self.get_sentiment_for_coin(coin, hours=4)
        sent_1h = self.get_sentiment_for_coin(coin, hours=1)

        return {
            "news_sentiment_24h": sent["score"],
            "news_sentiment_4h": sent_4h["score"],
            "news_sentiment_1h": sent_1h["score"],
            "news_volume_24h": sent["article_count"],
            "news_signal_strength": sent["signal_strength"],
            "news_bullish_ratio": (
                sent["bullish_count"] / max(1, sent["article_count"])
            ),
            "news_bearish_ratio": (
                sent["bearish_count"] / max(1, sent["article_count"])
            ),
        }

    def should_trade_on_news(self, coin: str) -> dict | None:
        """Check if there's a strong enough news signal to trigger a trade.

        Returns None if no signal, or a dict with direction and confidence.
        """
        threshold = self.news_config.get("signal_threshold", 0.6)
        sent = self.get_sentiment_for_coin(coin, hours=4)

        if sent["signal_strength"] < 0.3:
            return None  # Not enough consensus

        if sent["score"] > threshold:
            return {
                "direction": "long",
                "confidence": min(1.0, sent["signal_strength"] + abs(sent["score"])),
                "reason": f"Strong bullish news ({sent['bullish_count']} articles)",
                "headlines": sent["top_headlines"][:3],
            }
        elif sent["score"] < -threshold:
            return {
                "direction": "short",
                "confidence": min(1.0, sent["signal_strength"] + abs(sent["score"])),
                "reason": f"Strong bearish news ({sent['bearish_count']} articles)",
                "headlines": sent["top_headlines"][:3],
            }

        return None

    # ── Scoring ───────────────────────────────────────────────

    def _score_sentiment(self, title: str, summary: str) -> float:
        """Score text sentiment from -1 (bearish) to +1 (bullish)."""
        text = f"{title} {summary}".lower()

        bull_hits = sum(1 for word in BULLISH_WORDS if word in text)
        bear_hits = sum(1 for word in BEARISH_WORDS if word in text)

        total = bull_hits + bear_hits
        if total == 0:
            return 0.0

        # Title words count double
        title_lower = title.lower()
        for word in BULLISH_WORDS:
            if word in title_lower:
                bull_hits += 1
        for word in BEARISH_WORDS:
            if word in title_lower:
                bear_hits += 1

        total = bull_hits + bear_hits
        score = (bull_hits - bear_hits) / total
        return max(-1.0, min(1.0, score))

    def _detect_coins(self, title: str, summary: str) -> dict:
        """Detect which coins a news article is about."""
        text = f"{title} {summary}".lower()
        detected = {}

        for coin, keywords in COIN_KEYWORDS.items():
            for kw in keywords:
                if kw in text:
                    detected[coin] = detected.get(coin, 0) + 1
                    break

        # If no specific coin detected, assume it's about BTC (market-wide news)
        if not detected and any(w in text for w in ["crypto", "market", "blockchain"]):
            detected["BTC"] = 1

        return detected

    def _parse_date(self, entry) -> datetime | None:
        """Parse publication date from feed entry."""
        for field in ["published_parsed", "updated_parsed"]:
            parsed = entry.get(field)
            if parsed:
                try:
                    return datetime(*parsed[:6])
                except Exception:
                    pass
        return None
