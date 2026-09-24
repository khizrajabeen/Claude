"""Alpaca: US stocks and ETFs, with a broker behind the data.

The Nasdaq endpoint this bot has used for equities serves daily bars and
nothing else — no intraday, no quotes, and about three years of history
against a strategy roster whose deepest member wants 724 bars. Alpaca
serves minute bars, quotes and a paper trading account behind the same
credentials, which is what makes a stock leg worth running at all rather
than simulating on one price a day.

Credentials come from the environment, never from a config file that
could be committed:

    ALPACA_API_KEY_ID, ALPACA_API_SECRET_KEY, ALPACA_PAPER

`ALPACA_PAPER` defaults to true. Live trading is opt-in by an explicit
setting, because the difference between the two endpoints is one string
and the consequence of getting it wrong is real money.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.data.base import ProviderError
from bot.markets.instrument import AssetClass, Instrument

logger = logging.getLogger("trading_bot")

CRYPTO_URL = "https://data.alpaca.markets/v1beta3/crypto/us"
DATA_URL = "https://data.alpaca.markets/v2"
PAPER_URL = "https://paper-api.alpaca.markets/v2"
LIVE_URL = "https://api.alpaca.markets/v2"

# Alpaca's bar sizes, keyed by this bot's timeframe names.
TIMEFRAMES = {
    "1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min",
    "1h": "1Hour", "4h": "4Hour", "1d": "1Day", "1w": "1Week",
}

# Roughly how many calendar days a bar count spans, for the request
# window. Generous on purpose: asking for too wide a range costs nothing,
# asking for too narrow a one silently returns short history.
_SPAN = {"1Min": 1 / 390, "5Min": 5 / 390, "15Min": 15 / 390,
         "30Min": 30 / 390, "1Hour": 1 / 6.5, "4Hour": 4 / 6.5,
         "1Day": 365 / 252, "1Week": 7}


class AlpacaProvider:
    """Bars, quotes and account state for US equities."""

    def __init__(self, config: dict, session=None):
        alpaca = (config.get("alpaca") or {})
        self.key = os.environ.get("ALPACA_API_KEY_ID") or alpaca.get("key_id", "")
        self.secret = (os.environ.get("ALPACA_API_SECRET_KEY")
                       or alpaca.get("secret_key", ""))
        paper_env = os.environ.get("ALPACA_PAPER")
        self.paper = (paper_env.lower() != "false" if paper_env is not None
                      else bool(alpaca.get("paper", True)))
        self.feed = alpaca.get("feed", "iex")   # "sip" needs a paid plan
        self.timeout = float(alpaca.get("timeout_seconds", 20))
        self._session = session
        self._warned = False
        self._crypto_cache: dict[tuple, pd.DataFrame] = {}

    # ── Plumbing ──────────────────────────────────────────────

    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)

    @property
    def trading_url(self) -> str:
        return PAPER_URL if self.paper else LIVE_URL

    def session(self):
        if self._session is not None:
            return self._session
        import requests

        session = requests.Session()
        session.trust_env = True
        session.headers.update({
            "APCA-API-KEY-ID": self.key,
            "APCA-API-SECRET-KEY": self.secret,
            "accept": "application/json",
        })
        bundle = os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
        if os.path.exists(bundle):
            session.verify = bundle
        self._session = session
        return session

    def _get(self, url: str, params: dict | None = None) -> dict:
        if not self.configured:
            raise ProviderError(
                "Alpaca keys are not set — export ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY, or use the settings page's "
                "Export .env button"
            )
        response = None
        for attempt in range(5):
            response = self.session().get(url, params=params or {},
                                          timeout=self.timeout)
            if response.status_code != 429:
                break
            # Alpaca says when to come back; believe it rather than
            # guessing, and fall back to exponential backoff if absent.
            wait = float(response.headers.get("Retry-After") or 0) or min(2 ** attempt, 20)
            logger.info("Alpaca rate limit — waiting %.0fs (attempt %d/5)",
                        wait, attempt + 1)
            time.sleep(wait)
        if response.status_code == 403:
            raise ProviderError(
                "Alpaca refused the request (403) — check the keys match the "
                f"{'paper' if self.paper else 'live'} endpoint being used"
            )
        response.raise_for_status()
        return response.json() or {}

    # ── DataProvider interface ────────────────────────────────

    @staticmethod
    def crypto_symbol(symbol: str) -> str:
        """The bot's spelling as Alpaca's crypto feed spells it."""
        base = symbol.split(":")[0]
        if "/" not in base:
            return base
        left, right = base.split("/", 1)
        return f"{left}/{'USD' if right in ('USDT', 'USDC') else right}"

    def handles(self, instrument: Instrument) -> bool:
        # Alpaca serves both legs of this account: US equities and ETFs,
        # and 33 USD crypto pairs. Backtesting crypto on OKX prices and
        # then trading it on Alpaca was comparing two different books —
        # the spread that decides whether a trade pays for itself is a
        # property of the venue, not of the coin.
        if instrument.asset_class == AssetClass.CRYPTO_SPOT:
            return self.configured and instrument.venue == "alpaca"
        if instrument.asset_class not in (AssetClass.EQUITY, AssetClass.ETF):
            return False
        if not self.configured:
            if not self._warned:
                logger.info("Alpaca not configured — equities fall back to "
                            "the daily feed")
                self._warned = True
            return False
        return True

    def bars(self, instrument: Instrument, timeframe: str,
             limit: int) -> pd.DataFrame:
        if instrument.asset_class == AssetClass.CRYPTO_SPOT:
            return self._crypto_bars(instrument, timeframe, limit)
        size = TIMEFRAMES.get(timeframe, "1Day")
        span_days = max(2.0, limit * _SPAN.get(size, 1.5) * 1.4) + 5
        start = datetime.now(timezone.utc) - timedelta(days=span_days)

        rows: list[dict] = []
        page = None
        # Alpaca caps a page at 10,000 bars; a deep request needs paging,
        # and stopping at the first page would quietly return a fraction
        # of the history asked for.
        while len(rows) < limit:
            payload = self._get(
                f"{DATA_URL}/stocks/{instrument.feed_symbol}/bars",
                {"timeframe": size, "start": start.isoformat(),
                 "limit": min(10_000, limit * 2), "adjustment": "split",
                 "feed": self.feed, **({"page_token": page} if page else {})},
            )
            batch = payload.get("bars") or []
            rows.extend(batch)
            page = payload.get("next_page_token")
            if not page or not batch:
                break

        if not rows:
            return pd.DataFrame()
        frame = pd.DataFrame(rows)
        frame["t"] = pd.to_datetime(frame["t"], utc=True)
        frame = (frame.rename(columns={"o": "open", "h": "high", "l": "low",
                                       "c": "close", "v": "volume"})
                 .set_index("t")
                 .sort_index())
        keep = ["open", "high", "low", "close", "volume"]
        return frame[keep].tail(limit)

    def prefetch_crypto(self, instruments, timeframe: str, limit: int) -> int:
        """Fill the cache for many coins in one request.

        The crypto endpoint accepts `symbols` plural. Thirty coins asked
        for one at a time is thirty requests against a 200-per-minute
        budget, repeated per timeframe and per replay day; the same data
        in one request is one. Returns how many symbols came back.
        """
        symbols = [self.crypto_symbol(i.feed_symbol) for i in instruments]
        if not symbols:
            return 0
        size = TIMEFRAMES.get(timeframe, "1Day")
        span = self._crypto_span(size, limit)
        start = datetime.now(timezone.utc) - timedelta(days=span)

        collected: dict[str, list] = {s: [] for s in symbols}
        page = None
        while True:
            payload = self._get(f"{CRYPTO_URL}/bars", {
                "symbols": ",".join(symbols), "timeframe": size,
                "start": start.isoformat(), "limit": 10_000,
                **({"page_token": page} if page else {}),
            })
            for sym, rows in (payload.get("bars") or {}).items():
                collected.setdefault(sym, []).extend(rows)
            page = payload.get("next_page_token")
            if not page or all(len(v) >= limit for v in collected.values()):
                break

        filled = 0
        for sym, rows in collected.items():
            if not rows:
                continue
            self._crypto_cache[(sym, size)] = self._frame(rows).tail(limit)
            filled += 1
        return filled

    @staticmethod
    def _crypto_span(size: str, limit: int) -> float:
        per_day = {"1Min": 1440, "5Min": 288, "15Min": 96, "30Min": 48,
                   "1Hour": 24, "4Hour": 6, "1Day": 1, "1Week": 1 / 7}
        return max(2.0, limit / per_day.get(size, 1)) * 1.3 + 2

    @staticmethod
    def _frame(rows: list) -> pd.DataFrame:
        frame = pd.DataFrame(rows)
        frame["t"] = pd.to_datetime(frame["t"], utc=True)
        frame = (frame.rename(columns={"o": "open", "h": "high", "l": "low",
                                       "c": "close", "v": "volume"})
                 .set_index("t").sort_index())
        return frame[["open", "high", "low", "close", "volume"]]

    def _crypto_bars(self, instrument: Instrument, timeframe: str,
                     limit: int) -> pd.DataFrame:
        """Crypto bars. Same shape out, different endpoint in.

        Crypto trades every hour of every day, so a bar count spans far
        fewer calendar days than the same count of equity bars — the
        equity spans in `_SPAN` would ask for years of history and get
        throttled for it.
        """
        size = TIMEFRAMES.get(timeframe, "1Day")
        symbol = self.crypto_symbol(instrument.feed_symbol)
        cached = self._crypto_cache.get((symbol, size))
        if cached is not None and len(cached) >= limit:
            return cached.tail(limit)

        start = datetime.now(timezone.utc) - timedelta(
            days=self._crypto_span(size, limit))
        rows: list[dict] = []
        page = None
        while len(rows) < limit:
            payload = self._get(f"{CRYPTO_URL}/bars", {
                "symbols": symbol, "timeframe": size,
                "start": start.isoformat(), "limit": min(10_000, limit * 2),
                **({"page_token": page} if page else {}),
            })
            batch = (payload.get("bars") or {}).get(symbol) or []
            rows.extend(batch)
            page = payload.get("next_page_token")
            if not page or not batch:
                break

        if not rows:
            return pd.DataFrame()
        frame = self._frame(rows)
        self._crypto_cache[(symbol, size)] = frame
        return frame.tail(limit)

    def price(self, instrument: Instrument) -> float | None:
        if instrument.asset_class == AssetClass.CRYPTO_SPOT:
            symbol = self.crypto_symbol(instrument.feed_symbol)
            try:
                q = self._get(f"{CRYPTO_URL}/latest/quotes", {"symbols": symbol})
                x = (q.get("quotes") or {}).get(symbol) or {}
                ask, bid = float(x.get("ap") or 0), float(x.get("bp") or 0)
                return (ask + bid) / 2 if ask and bid else None
            except Exception as e:
                logger.debug("Alpaca crypto price failed for %s: %s",
                             instrument.symbol, e)
                return None
        return self._equity_price(instrument)

    def _equity_price(self, instrument: Instrument) -> float | None:
        try:
            payload = self._get(
                f"{DATA_URL}/stocks/{instrument.feed_symbol}/trades/latest",
                {"feed": self.feed})
        except Exception as e:
            logger.debug("Alpaca price failed for %s: %s", instrument.symbol, e)
            return None
        price = ((payload.get("trade") or {}).get("p"))
        return float(price) if price else None

    def quote_volume(self, instrument: Instrument) -> float | None:
        frame = self.bars(instrument, "1d", 2)
        if frame.empty:
            return None
        row = frame.iloc[-1]
        return float(row["close"]) * float(row["volume"])

    def spread_bps(self, instrument: Instrument) -> float | None:
        try:
            payload = self._get(
                f"{DATA_URL}/stocks/{instrument.feed_symbol}/quotes/latest",
                {"feed": self.feed})
        except Exception:
            return None
        quote = payload.get("quote") or {}
        bid, ask = quote.get("bp"), quote.get("ap")
        if not bid or not ask:
            return None
        mid = (float(bid) + float(ask)) / 2
        return (float(ask) - float(bid)) / mid * 10_000 if mid else None

    def funding_rate(self, instrument: Instrument) -> float | None:
        return None      # equities have no funding leg

    def market_limits(self, instrument: Instrument) -> dict:
        # Whole shares. Alpaca supports fractional, but a fractional
        # position cannot be shorted and the sizing does not model that.
        return {"min_qty": 1.0, "qty_step": 1.0, "min_notional": 1.0}

    # ── Account ───────────────────────────────────────────────

    def account(self) -> dict:
        """Equity, cash and buying power, for reconciling against paper."""
        payload = self._get(f"{self.trading_url}/account")
        return {
            "equity": float(payload.get("equity") or 0.0),
            "cash": float(payload.get("cash") or 0.0),
            "buying_power": float(payload.get("buying_power") or 0.0),
            "currency": payload.get("currency", "USD"),
            "paper": self.paper,
            "blocked": bool(payload.get("trading_blocked")),
        }

    def positions(self) -> list[dict]:
        payload = self._get(f"{self.trading_url}/positions")
        rows = payload if isinstance(payload, list) else []
        return [{
            "symbol": r.get("symbol"),
            "quantity": float(r.get("qty") or 0.0),
            "side": "long" if float(r.get("qty") or 0) >= 0 else "short",
            "entry_price": float(r.get("avg_entry_price") or 0.0),
            "market_value": float(r.get("market_value") or 0.0),
            "unrealized_pnl": float(r.get("unrealized_pl") or 0.0),
        } for r in rows]

    def clock(self) -> dict:
        """Whether the market is open, from the venue rather than a table."""
        payload = self._get(f"{self.trading_url}/clock")
        return {
            "is_open": bool(payload.get("is_open")),
            "next_open": payload.get("next_open"),
            "next_close": payload.get("next_close"),
        }
