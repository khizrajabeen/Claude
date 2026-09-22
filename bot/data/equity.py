"""US equity and ETF bars from Nasdaq's public quote API.

Chosen because it needs no key and serves arbitrary symbols. Two
consequences worth stating plainly rather than discovering later:

  * **Daily bars only.** That is not a limitation the bot has to work
    around — it is the right granularity here. A US equity trades 6.5
    hours, so an hourly bar is one of seven in the whole session, and
    anything faster is mostly microstructure this bot has no edge in. The
    strategies are trend and pullback systems whose published holding
    periods are days to weeks.
  * **No live intraday quote.** The most recent daily close stands in for
    the current price, which means an equity position is marked and managed
    on closes. Stops are therefore evaluated against daily highs and lows,
    which is honest for a daily system and would not be for an intraday
    one.

Responses are cached on disk for the session, because the endpoint is
rate-limited and a bench re-requests the same history for every variant.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import requests

from bot.data.base import ProviderError
from bot.markets.instrument import AssetClass, Instrument

logger = logging.getLogger("trading_bot")

BASE_URL = "https://api.nasdaq.com/api/quote/{symbol}/historical"
BROWSER_HEADERS = {
    # The endpoint rejects requests without a browser-shaped User-Agent.
    "User-Agent": ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
}


class EquityProvider:
    """Daily OHLCV for US stocks and ETFs."""

    name = "equity"

    def __init__(self, config: dict):
        self.config = config
        equity = config.get("equity", {})
        self.max_retries = int(equity.get("max_retries", 3))
        self.retry_delay = float(equity.get("retry_delay_seconds", 2.0))
        self.request_gap = float(equity.get("request_gap_seconds", 0.6))
        self.cache_dir = Path(equity.get("cache_dir", "data/equity"))
        self.cache_hours = float(equity.get("cache_hours", 6.0))

        self._session = requests.Session()
        self._session.trust_env = True     # the sandbox routes through a proxy
        self._session.headers.update(BROWSER_HEADERS)
        self._memory: dict[str, pd.DataFrame] = {}
        self._last_request = 0.0

    # ── DataProvider ──────────────────────────────────────────

    def handles(self, instrument: Instrument) -> bool:
        return instrument.asset_class in (AssetClass.EQUITY, AssetClass.ETF)

    def bars(self, instrument: Instrument, timeframe: str, limit: int) -> pd.DataFrame:
        if timeframe not in ("1d", "1w"):
            # Better to say so than to silently serve daily bars to a
            # strategy that believes it is looking at hours.
            raise ProviderError(
                f"{self.name} serves daily bars only; {instrument.symbol} "
                f"asked for {timeframe}"
            )

        daily = self._daily(instrument, limit if timeframe == "1d" else limit * 7)
        if timeframe == "1w":
            return _to_weekly(daily).tail(limit)
        return daily.tail(limit)

    def price(self, instrument: Instrument) -> float | None:
        try:
            bars = self._daily(instrument, 5)
        except ProviderError:
            return None
        return float(bars["close"].iloc[-1]) if not bars.empty else None

    def quote_volume(self, instrument: Instrument) -> float | None:
        """Dollar volume, averaged over the last fortnight of sessions."""
        try:
            bars = self._daily(instrument, 20)
        except ProviderError:
            return None
        if bars.empty:
            return None
        recent = bars.tail(10)
        return float((recent["close"] * recent["volume"]).mean())

    def spread_bps(self, instrument: Instrument) -> float | None:
        return None  # no quote data on this endpoint

    def order_book(self, instrument: Instrument, depth: int = 20) -> dict | None:
        return None  # no depth-of-book on this endpoint

    def funding_rate(self, instrument: Instrument) -> float | None:
        return None  # equities have no funding leg

    def market_limits(self, instrument: Instrument) -> dict:
        # Whole shares. Fractional trading exists but is broker-specific,
        # and rounding down is the safe assumption.
        return {"min_qty": 1.0, "qty_step": 1.0, "min_notional": 0.0}

    # ── Fetching ──────────────────────────────────────────────

    def _daily(self, instrument: Instrument, limit: int) -> pd.DataFrame:
        """Daily bars, from the shallowest source that is deep enough.

        Both caches have to be checked against `limit`, not merely for
        existence. A briefing asks for 120 bars and a replay's deepest
        strategy asks for 780; serving the replay the briefing's cache
        hands it two thirds of a year less history than it asked for, and
        it has no way to tell — the frame it gets back looks perfectly
        valid, just short, so every deep strategy silently declines and
        the replay reports "no opportunity".
        """
        cached = self._memory.get(instrument.feed_symbol)
        if cached is not None and len(cached) >= limit:
            return cached.tail(limit)

        disk = self._read_cache(instrument)
        if disk is not None and len(disk) >= limit:
            self._memory[instrument.feed_symbol] = disk
            return disk.tail(limit)

        frame = self._download(instrument, limit)
        # Keep whichever is deeper: a later shallow request must not
        # evict the history a deep one already paid for.
        if cached is not None and len(cached) > len(frame):
            frame = cached
        if disk is not None and len(disk) > len(frame):
            frame = disk
        self._memory[instrument.feed_symbol] = frame
        self._write_cache(instrument, frame)
        return frame.tail(limit)

    def _download(self, instrument: Instrument, limit: int) -> pd.DataFrame:
        # Ask for calendar days generously: roughly 252 sessions a year.
        span_days = int(max(limit, 30) * 1.6) + 10
        end = date.today()
        start = end - timedelta(days=span_days)
        asset_class = "etf" if instrument.asset_class is AssetClass.ETF else "stocks"

        params = {
            "assetclass": asset_class,
            "fromdate": start.isoformat(),
            "todate": end.isoformat(),
            "limit": max(limit + 10, 50),
        }
        url = BASE_URL.format(symbol=instrument.feed_symbol.upper())

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            try:
                response = self._session.get(url, params=params, timeout=30)
                response.raise_for_status()
                payload = response.json()
            except Exception as e:
                last_error = e
                logger.debug("Equity fetch failed for %s (attempt %d): %s",
                             instrument.symbol, attempt, e)
                time.sleep(self.retry_delay * attempt)
                continue

            rows = (((payload or {}).get("data") or {}).get("tradesTable") or {}).get("rows")
            if not rows:
                # A stock queried as an ETF (or the reverse) returns an
                # empty body rather than an error, so retry the other class
                # once before giving up.
                if asset_class == "stocks":
                    params["assetclass"] = "etf"
                    asset_class = "etf"
                    continue
                raise ProviderError(
                    f"no history for {instrument.symbol} — check the symbol "
                    f"and whether it is a stock or an ETF"
                )
            return _rows_to_frame(rows)

        raise ProviderError(
            f"{instrument.symbol}: {self.max_retries} attempts failed ({last_error})"
        )

    def _throttle(self) -> None:
        """Space requests out; the endpoint rate-limits bursts."""
        elapsed = time.monotonic() - self._last_request
        if elapsed < self.request_gap:
            time.sleep(self.request_gap - elapsed)
        self._last_request = time.monotonic()

    # ── Disk cache ────────────────────────────────────────────

    def _cache_path(self, instrument: Instrument) -> Path:
        safe = instrument.feed_symbol.upper().replace("/", "_")
        return self.cache_dir / f"{safe}.json"

    def _read_cache(self, instrument: Instrument) -> pd.DataFrame | None:
        path = self._cache_path(instrument)
        if not path.exists():
            return None
        age_hours = (time.time() - path.stat().st_mtime) / 3600
        if age_hours > self.cache_hours:
            return None
        try:
            with open(path) as f:
                rows = json.load(f)
            return _rows_to_frame(rows)
        except (OSError, ValueError, KeyError):
            return None

    def _write_cache(self, instrument: Instrument, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            rows = [
                {"date": stamp.strftime("%m/%d/%Y"), "open": row["open"],
                 "high": row["high"], "low": row["low"], "close": row["close"],
                 "volume": row["volume"]}
                for stamp, row in frame.iterrows()
            ]
            with open(self._cache_path(instrument), "w") as f:
                json.dump(rows, f)
        except OSError as e:
            logger.debug("Could not cache %s: %s", instrument.symbol, e)


# ── parsing ──────────────────────────────────────────────────

def _clean(value) -> float:
    """Nasdaq returns '$338.98' and '50,484,690'."""
    if value in (None, "", "N/A"):
        return float("nan")
    return float(str(value).replace("$", "").replace(",", "").strip())


def _rows_to_frame(rows: list[dict]) -> pd.DataFrame:
    records = []
    for row in rows:
        try:
            stamp = pd.Timestamp(datetime.strptime(row["date"], "%m/%d/%Y"), tz="UTC")
        except (KeyError, ValueError):
            continue
        records.append({
            "timestamp": stamp,
            "open": _clean(row.get("open")),
            "high": _clean(row.get("high")),
            "low": _clean(row.get("low")),
            "close": _clean(row.get("close")),
            "volume": _clean(row.get("volume")),
        })

    if not records:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = pd.DataFrame(records).set_index("timestamp").sort_index()
    # Newest-first is how the endpoint returns them; sorting above fixes
    # that, and dropping unpriced rows keeps indicators finite.
    return frame[~frame.index.duplicated(keep="last")].dropna(subset=["close"])


def _to_weekly(daily: pd.DataFrame) -> pd.DataFrame:
    if daily.empty:
        return daily
    return daily.resample("W-FRI").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["close"])
