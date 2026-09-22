"""What to trade, decided each day rather than written in a config file.

A static universe goes stale in two directions. It holds names whose
liquidity has drained away, and it misses the ones that now dominate the
tape: a scan of OKX's 406 active USDT spot markets put ZEC and HYPE in the
top twelve by turnover, and neither was in the configured list.

The screen ranks the whole quote-currency universe on what can be measured
before a trade is taken, and hands the day a shortlist:

  **Turnover.** Rank by 24-hour quote volume. This is the closest thing to
  "top ten" that an exchange can answer honestly — market capitalisation
  is a supply figure and says nothing about whether a position can be got
  out of, and it is liquidity that decides whether an edge survives the
  spread.

  **Age.** A market listed last week has no history to measure a trend,
  a beta or a volatility on, and its first weeks are dominated by listing
  flow rather than by anything a strategy models. New listings are
  reported, and by default excluded, because the honest thing is to say
  "this exists and I cannot trade it yet" rather than to trade it blind.

  **News.** Per-coin sentiment over the recent window, so a name in the
  headlines is visible before the day's candidates are chosen rather than
  after.

  **Correlation to the majors.** Altcoins are levered Bitcoin — measured
  betas of 0.85 to 1.44 with an average pairwise correlation of 0.62 — so
  the screen reports how much of a candidate is really just BTC. Ten names
  that all track Bitcoin is one position held ten times.

Nothing here decides to trade. It decides what is worth looking at, which
is a different and smaller claim.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

logger = logging.getLogger("trading_bot")

# Quote currencies whose "pairs" are not really trades — a stablecoin
# against a stablecoin is a funding operation, not a position.
STABLECOINS = {"USDT", "USDC", "DAI", "TUSD", "FDUSD", "USDE", "PYUSD",
               "EURT", "EURS", "USDP", "BUSD"}


@dataclass
class CoinCandidate:
    """One market, as the screen sees it before any trade is considered."""

    symbol: str
    base: str
    quote_volume: float = 0.0
    rank: int = 0
    age_days: float | None = None
    is_new: bool = False
    news_score: float = 0.0
    news_articles: int = 0
    beta: float | None = None
    beta_r2: float | None = None
    price: float = 0.0
    change_24h_pct: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def tradable(self) -> bool:
        return not self.notes

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol, "base": self.base, "rank": self.rank,
            "quote_volume": round(self.quote_volume, 2),
            "age_days": round(self.age_days, 1) if self.age_days is not None else None,
            "is_new": self.is_new,
            "news_score": round(self.news_score, 3),
            "news_articles": self.news_articles,
            "beta": round(self.beta, 3) if self.beta is not None else None,
            "beta_r2": round(self.beta_r2, 3) if self.beta_r2 is not None else None,
            "change_24h_pct": round(self.change_24h_pct, 2),
            "notes": list(self.notes),
        }


class CoinScreener:
    """Ranks an exchange's markets into a shortlist worth looking at."""

    def __init__(self, config: dict, exchange=None, news=None):
        screen = config.get("screen", {})
        self.config = config
        self.exchange = exchange
        self.news = news
        self.quote = str(screen.get("quote", "USDT")).upper()
        self.top_n = int(screen.get("top_n", 10))
        self.min_quote_volume = float(screen.get("min_quote_volume_24h", 10_000_000))
        self.min_age_days = float(screen.get("min_age_days", 90))
        self.new_listing_days = float(screen.get("new_listing_days", 30))
        self.include_new = bool(screen.get("include_new_listings", False))
        self.exclude = {s.upper() for s in screen.get("exclude", []) or []}
        self.news_window_hours = float(screen.get("news_window_hours", 24))

    # ── The scan ──────────────────────────────────────────────

    def scan(self, now: datetime | None = None,
             beta_book=None) -> list[CoinCandidate]:
        """Every market that could be traded, ranked, with its reasons."""
        now = now or datetime.now(timezone.utc)
        if self.exchange is None:
            logger.warning("Screener has no exchange — nothing to scan")
            return []

        try:
            markets = self.exchange.load_markets()
        except Exception as e:
            logger.warning("Screen failed to load markets: %s", e)
            return []

        symbols = [s for s, m in markets.items() if self._is_candidate(s, m)]
        if not symbols:
            return []

        tickers = self._tickers(symbols)
        candidates: list[CoinCandidate] = []
        for symbol in symbols:
            ticker = tickers.get(symbol) or {}
            volume = float(ticker.get("quoteVolume") or 0.0)
            if volume <= 0:
                continue
            market = markets[symbol]
            candidate = CoinCandidate(
                symbol=symbol,
                base=str(market.get("base") or symbol.split("/")[0]).upper(),
                quote_volume=volume,
                price=float(ticker.get("last") or 0.0),
                change_24h_pct=float(ticker.get("percentage") or 0.0),
            )
            self._age(candidate, market, now)
            self._flag(candidate)
            candidates.append(candidate)

        candidates.sort(key=lambda c: -c.quote_volume)
        for position, candidate in enumerate(candidates, 1):
            candidate.rank = position

        self._attach_news(candidates, now)
        self._attach_beta(candidates, beta_book)
        return candidates

    def shortlist(self, now: datetime | None = None,
                  beta_book=None) -> list[CoinCandidate]:
        """The top `top_n` markets that carry no disqualifying note."""
        return [c for c in self.scan(now, beta_book) if c.tradable][: self.top_n]

    # ── Pieces ────────────────────────────────────────────────

    def _is_candidate(self, symbol: str, market: dict) -> bool:
        if not market.get("spot") or not market.get("active"):
            return False
        if str(market.get("quote") or "").upper() != self.quote:
            return False
        base = str(market.get("base") or "").upper()
        # A stablecoin against a stablecoin is a funding operation, not a
        # position — it will rank high on turnover and never move.
        return base not in STABLECOINS

    def _tickers(self, symbols: list[str]) -> dict:
        try:
            return self.exchange.fetch_tickers(symbols) or {}
        except Exception as e:
            logger.warning("Screen could not fetch tickers: %s", e)
            return {}

    def _age(self, candidate: CoinCandidate, market: dict,
             now: datetime) -> None:
        """How long this market has existed, where the venue says so."""
        created = market.get("created")
        if created in (None, ""):
            info = market.get("info") or {}
            created = info.get("listTime") or info.get("onlineTime")
        try:
            listed = datetime.fromtimestamp(float(created) / 1000.0, tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            return
        candidate.age_days = max(0.0, (now - listed).total_seconds() / 86_400.0)
        candidate.is_new = candidate.age_days < self.new_listing_days

    def _flag(self, candidate: CoinCandidate) -> None:
        """Record why a market is not worth looking at, rather than dropping it."""
        if candidate.base in self.exclude or candidate.symbol.upper() in self.exclude:
            candidate.notes.append("excluded by config")
        if candidate.quote_volume < self.min_quote_volume:
            candidate.notes.append(
                f"${candidate.quote_volume/1e6:.1f}M turnover below "
                f"${self.min_quote_volume/1e6:.0f}M")
        if candidate.age_days is not None and candidate.age_days < self.min_age_days:
            # A market listed last week has no history to measure a trend,
            # a beta or a volatility on.
            if not (self.include_new and candidate.is_new):
                candidate.notes.append(
                    f"listed {candidate.age_days:.0f}d ago, needs "
                    f"{self.min_age_days:.0f}d of history")

    def _attach_news(self, candidates: list[CoinCandidate],
                     now: datetime) -> None:
        if self.news is None:
            return
        # Only for the names that could actually be traded; sentiment for
        # four hundred markets is four hundred lookups for nothing.
        for candidate in [c for c in candidates if c.tradable][: self.top_n * 3]:
            try:
                sentiment = self.news.sentiment_for(
                    candidate.symbol, self.news_window_hours, now)
            except Exception:
                continue
            candidate.news_score = float(sentiment.get("score") or 0.0)
            candidate.news_articles = int(sentiment.get("articles") or 0)

    def _attach_beta(self, candidates: list[CoinCandidate], beta_book) -> None:
        if beta_book is None:
            return
        for candidate in candidates:
            if beta_book.known(candidate.symbol):
                candidate.beta = beta_book.beta(candidate.symbol)
                candidate.beta_r2 = beta_book.r2(candidate.symbol)


def render(candidates: list[CoinCandidate], logger_, limit: int = 20) -> None:
    """Print the screen the way it should be read."""
    logger_.info("═" * 78)
    logger_.info("  COIN SCREEN — %d markets ranked by 24h turnover", len(candidates))
    logger_.info("═" * 78)
    logger_.info("  %-4s %-14s %>12s %8s %7s %6s %6s  %s".replace(">", ""),
                 "#", "market", "24h vol $M", "chg%", "age d", "news", "beta",
                 "note")
    for candidate in candidates[:limit]:
        note = "; ".join(candidate.notes) or ("NEW" if candidate.is_new else "")
        logger_.info(
            "  %-4d %-14s %12.1f %+8.2f %7s %+6.2f %6s  %s",
            candidate.rank, candidate.symbol, candidate.quote_volume / 1e6,
            candidate.change_24h_pct,
            f"{candidate.age_days:.0f}" if candidate.age_days is not None else "?",
            candidate.news_score,
            f"{candidate.beta:.2f}" if candidate.beta is not None else "-",
            note,
        )
    logger_.info("═" * 78)
