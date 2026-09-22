"""Exchange client wrapper around ccxt.

Three things this layer is responsible for beyond passing calls through:

  * Working behind a proxy. ccxt sets ``session.trust_env = False``, so a
    process whose only route out is ``HTTPS_PROXY`` fails every request with
    a bare NetworkError. We opt back in and honour a custom CA bundle.
  * Retrying transient failures with backoff instead of dying mid-session.
  * Serving market metadata (tick size, lot step, min notional) so the risk
    layer sizes orders the exchange will actually accept.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import ccxt
import pandas as pd

logger = logging.getLogger("trading_bot")

RETRYABLE = (
    ccxt.NetworkError,
    ccxt.RequestTimeout,
    ccxt.ExchangeNotAvailable,
    ccxt.DDoSProtection,
    ccxt.RateLimitExceeded,
)


class ExchangeClient:
    """Wrapper around a ccxt exchange with retries and market metadata."""

    def __init__(self, config: dict):
        self.config = config.get("exchange", {})
        self.max_retries = int(self.config.get("max_retries", 4))
        self.name = self.config["name"].lower()
        self.exchange = self._init_exchange()
        self._markets: dict = {}

    # ── Setup ─────────────────────────────────────────────────

    def _init_exchange(self) -> ccxt.Exchange:
        if self.name not in ccxt.exchanges:
            raise ValueError(
                f"Exchange '{self.name}' not supported by ccxt. "
                f"Try one of: binance, kraken, coinbase, okx, kucoin, bybit"
            )

        options = {}
        market_type = self.config.get("market_type", "spot")
        if market_type in ("swap", "future"):
            options["defaultType"] = market_type

        exchange = getattr(ccxt, self.name)({
            "apiKey": self.config.get("api_key", ""),
            "secret": self.config.get("api_secret", ""),
            "password": self.config.get("api_password", "") or None,
            "enableRateLimit": True,
            "timeout": int(self.config.get("timeout_ms", 30_000)),
            "options": options,
        })

        # ccxt disables environment proxy discovery by default; without this
        # every request fails in a sandbox whose only egress is a proxy.
        if self.config.get("trust_env", True):
            exchange.session.trust_env = True

        ca_bundle = self.config.get("verify_ca") or os.environ.get("REQUESTS_CA_BUNDLE")
        if not ca_bundle and Path("/root/.ccr/ca-bundle.crt").exists():
            ca_bundle = "/root/.ccr/ca-bundle.crt"
        if ca_bundle and Path(ca_bundle).exists():
            exchange.verify = ca_bundle
            exchange.session.verify = ca_bundle
            logger.debug("Using CA bundle %s", ca_bundle)

        if self.config.get("sandbox", False):
            try:
                exchange.set_sandbox_mode(True)
                logger.info("Exchange %s running in SANDBOX mode", self.name)
            except ccxt.NotSupported:
                logger.warning("%s has no sandbox — falling back to read-only public data",
                               self.name)

        return exchange

    def _retry(self, fn, *args, **kwargs):
        """Call a ccxt method, retrying transient errors with backoff."""
        delay = 2.0
        last: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except RETRYABLE as e:
                last = e
                if attempt == self.max_retries:
                    break
                logger.warning(
                    "%s failed (attempt %d/%d): %s — retrying in %.0fs",
                    getattr(fn, "__name__", "request"), attempt, self.max_retries,
                    str(e)[:160], delay,
                )
                time.sleep(delay)
                delay *= 2
        raise last  # type: ignore[misc]

    # ── Market metadata ───────────────────────────────────────

    def load_markets(self, reload: bool = False) -> dict:
        if not self._markets or reload:
            self._markets = self._retry(self.exchange.load_markets, reload)
        return self._markets

    def market_limits(self, symbol: str) -> dict:
        """Lot step, min quantity and min notional for a symbol.

        Returned as zeros when the exchange does not publish them, which the
        sizing layer treats as "no constraint" rather than guessing.
        """
        try:
            markets = self.load_markets()
        except Exception as e:
            logger.debug("Could not load markets: %s", e)
            return {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0}

        market = markets.get(symbol)
        if not market:
            return {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0}

        limits = market.get("limits") or {}
        amount = limits.get("amount") or {}
        cost = limits.get("cost") or {}
        precision = market.get("precision") or {}

        step = precision.get("amount")
        if isinstance(step, int):
            step = 10 ** (-step)
        return {
            "min_qty": float(amount.get("min") or 0.0),
            "qty_step": float(step or 0.0),
            "min_notional": float(cost.get("min") or 0.0),
        }

    def symbol_supported(self, symbol: str) -> bool:
        try:
            return symbol in self.load_markets()
        except Exception:
            return False

    def resolve_symbol(self, symbol: str) -> str | None:
        """Map a requested pair onto one this exchange actually lists.

        Venues disagree about quote currencies (USDT vs USD vs USDC), so a
        config written for one exchange should not silently trade nothing on
        another.
        """
        try:
            markets = self.load_markets()
        except Exception:
            return symbol
        if symbol in markets:
            return symbol
        base, _, quote = symbol.partition("/")
        for alt_quote in (quote, "USDT", "USD", "USDC", "EUR"):
            candidate = f"{base}/{alt_quote}"
            if candidate in markets:
                if candidate != symbol:
                    logger.info("Mapped %s to %s on %s", symbol, candidate, self.name)
                return candidate
        return None

    # ── Market data ───────────────────────────────────────────

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500,
                    since: int | None = None) -> pd.DataFrame:
        raw = self._retry(self.exchange.fetch_ohlcv, symbol, timeframe,
                          since=since, limit=limit)
        return ohlcv_to_frame(raw)

    def fetch_ohlcv_paged(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """Collect `bars` candles ending now, paging as far back as needed.

        Venues cap a single call differently — OKX and Coinbase at 300,
        Kraken at 721, KuCoin at 1000 — and silently return the cap rather
        than erroring. A strategy asking for 600 bars would quietly receive
        300 and produce no signal at all, so anything that needs real depth
        must come through here.

        Pages backwards from the newest bar. Anchoring on the oldest page
        instead returns a window that ends weeks ago.
        """
        from bot.analysis.indicators import TIMEFRAME_SECONDS

        bar_ms = TIMEFRAME_SECONDS.get(timeframe, 3600) * 1000
        newest = self.fetch_ohlcv(symbol, timeframe, limit=min(bars, 1000))
        if newest.empty or len(newest) >= bars:
            return newest.tail(bars)

        chunks = [newest]
        collected = len(newest)
        oldest_ms = int(newest.index[0].timestamp() * 1000)
        page = max(len(newest), 100)

        while collected < bars:
            since = oldest_ms - page * bar_ms
            df = self.fetch_ohlcv(symbol, timeframe, limit=page, since=since)
            if df.empty:
                break
            new_oldest = int(df.index[0].timestamp() * 1000)
            if new_oldest >= oldest_ms:
                break  # the venue ignored `since`; another call would loop
            chunks.append(df)
            collected += len(df)
            oldest_ms = new_oldest

        combined = pd.concat(chunks)
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        return combined.tail(bars)

    def fetch_ticker(self, symbol: str) -> dict:
        return self._retry(self.exchange.fetch_ticker, symbol)

    def fetch_tickers(self, symbols: list[str] | None = None) -> dict:
        """Every ticker in one call.

        The screen ranks four hundred markets by turnover; asking for them
        one at a time is four hundred round trips and a rate limit. Venues
        that cannot serve the batch fall back to individual calls rather
        than returning nothing.
        """
        try:
            return self._retry(self.exchange.fetch_tickers, symbols) or {}
        except Exception as e:
            logger.debug("Batch tickers unavailable (%s) — falling back", e)

        out = {}
        for symbol in (symbols or [])[:200]:
            try:
                out[symbol] = self.fetch_ticker(symbol)
            except Exception:
                continue
        return out

    def fetch_order_book(self, symbol: str, limit: int = 20) -> dict:
        return self._retry(self.exchange.fetch_order_book, symbol, limit)

    def fetch_trades(self, symbol: str, limit: int = 500) -> list:
        return self._retry(self.exchange.fetch_trades, symbol, limit=limit)

    def fetch_balance(self) -> dict:
        return self._retry(self.exchange.fetch_balance)

    def get_current_price(self, symbol: str) -> float:
        ticker = self.fetch_ticker(symbol)
        price = ticker.get("last") or ticker.get("close")
        if price is None:
            bid, ask = ticker.get("bid"), ticker.get("ask")
            if bid and ask:
                price = (bid + ask) / 2
        if price is None:
            raise ccxt.ExchangeError(f"No price available for {symbol}")
        return float(price)

    def spread_bps(self, symbol: str) -> float | None:
        """Current bid/ask spread in basis points, or None if unavailable."""
        try:
            ticker = self.fetch_ticker(symbol)
            bid, ask = ticker.get("bid"), ticker.get("ask")
            if bid and ask and bid > 0:
                return float((ask - bid) / ((ask + bid) / 2) * 10_000)
        except Exception as e:
            logger.debug("Spread unavailable for %s: %s", symbol, e)
        return None

    def fetch_funding_rate(self, symbol: str) -> float | None:
        """Current perpetual funding rate as a per-settlement decimal."""
        if not self.exchange.has.get("fetchFundingRate"):
            return None
        try:
            data = self._retry(self.exchange.fetch_funding_rate, symbol)
            rate = data.get("fundingRate")
            return float(rate) if rate is not None else None
        except Exception as e:
            logger.debug("Funding rate unavailable for %s: %s", symbol, e)
            return None

    def quote_volume_24h(self, symbol: str) -> float | None:
        """24h traded value in quote currency.

        Derivatives venues report `baseVolume` in *contracts*, not in the
        base asset, and leave `quoteVolume` empty. OKX's BTC perp contract
        is 0.01 BTC, so taking baseVolume at face value overstates turnover
        a hundredfold — which does not merely look wrong, it waves a
        genuinely thin market straight through the liquidity filter. The
        contract size is applied wherever the venue publishes one.
        """
        try:
            ticker = self.fetch_ticker(symbol)
        except Exception as e:
            logger.debug("Volume unavailable for %s: %s", symbol, e)
            return None

        reported = ticker.get("quoteVolume")
        if reported:
            return float(reported)

        base_volume = ticker.get("baseVolume")
        last = ticker.get("last") or ticker.get("close")
        if not base_volume or not last:
            return None

        contract_size = 1.0
        try:
            market = self.load_markets().get(symbol) or {}
            if market.get("contract"):
                contract_size = float(market.get("contractSize") or 1.0)
        except Exception:
            pass

        return float(base_volume) * contract_size * float(last)

    # ── Orders ────────────────────────────────────────────────

    def create_market_order(self, symbol: str, side: str, amount: float) -> dict:
        logger.info("LIVE %s %s %s @ market", side.upper(), amount, symbol)
        return self._retry(self.exchange.create_order, symbol, "market", side, amount)

    def create_limit_order(self, symbol: str, side: str, amount: float,
                           price: float, post_only: bool = False) -> dict:
        params = {"postOnly": True} if post_only else {}
        logger.info("LIVE %s %s %s @ %s", side.upper(), amount, symbol, price)
        return self._retry(self.exchange.create_order, symbol, "limit", side,
                           amount, price, params)

    def create_stop_order(self, symbol: str, side: str, amount: float,
                          stop_price: float) -> dict:
        """Resting stop order, so the stop survives the bot going offline."""
        params = {"stopPrice": stop_price, "reduceOnly": True}
        return self._retry(self.exchange.create_order, symbol, "market", side,
                           amount, None, params)

    def fetch_open_orders(self, symbol: str | None = None) -> list:
        return self._retry(self.exchange.fetch_open_orders, symbol)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        return self._retry(self.exchange.cancel_order, order_id, symbol)


def ohlcv_to_frame(raw: list) -> pd.DataFrame:
    """Convert ccxt OHLCV rows into a UTC-indexed DataFrame."""
    df = pd.DataFrame(raw, columns=["timestamp", "open", "high", "low", "close", "volume"])
    if df.empty:
        return df.set_index("timestamp")
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return (
        df.drop_duplicates(subset="timestamp")
          .set_index("timestamp")
          .sort_index()
          .astype(float)
    )
