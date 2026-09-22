"""Crypto bars from ccxt — spot and perpetual futures.

Spot and perps are different instruments even when they track the same
coin: the perp has a funding leg, can be shorted, and often has a deeper
book. They are therefore separate providers over one exchange client, and
the perp provider is the only one that reports funding.
"""

from __future__ import annotations

import logging

import pandas as pd

from bot.data.base import ProviderError
from bot.exchange import ExchangeClient
from bot.markets.instrument import AssetClass, Instrument

logger = logging.getLogger("trading_bot")


class CryptoProvider:
    """Bars and quotes for crypto spot and perps via ccxt."""

    name = "crypto"

    def __init__(self, config: dict):
        self.config = config
        self._clients: dict[tuple[str, str], ExchangeClient] = {}
        self._resolved: dict[str, str | None] = {}

    # ── Client management ─────────────────────────────────────

    def _client(self, instrument: Instrument) -> ExchangeClient:
        """One ccxt client per (venue, market type).

        Spot and swap need different `defaultType`, so a single client
        cannot serve both — asking a spot client for BTC/USDT:USDT returns
        nothing and looks like a delisted symbol.
        """
        market_type = "swap" if instrument.asset_class is AssetClass.CRYPTO_PERP else "spot"
        key = (instrument.venue, market_type)
        if key not in self._clients:
            merged = dict(self.config)
            merged["exchange"] = {
                **self.config.get("exchange", {}),
                "name": instrument.venue,
                "market_type": market_type,
            }
            self._clients[key] = ExchangeClient(merged)
        return self._clients[key]

    def _symbol(self, instrument: Instrument) -> str:
        cache_key = f"{instrument.venue}:{instrument.feed_symbol}"
        if cache_key not in self._resolved:
            client = self._client(instrument)
            self._resolved[cache_key] = client.resolve_symbol(instrument.feed_symbol)
        resolved = self._resolved[cache_key]
        if resolved is None:
            raise ProviderError(f"{instrument.symbol} not listed on {instrument.venue}")
        return resolved

    # ── DataProvider ──────────────────────────────────────────

    def handles(self, instrument: Instrument) -> bool:
        return instrument.asset_class.is_crypto

    def bars(self, instrument: Instrument, timeframe: str, limit: int) -> pd.DataFrame:
        client = self._client(instrument)
        return client.fetch_ohlcv_paged(self._symbol(instrument), timeframe, limit)

    def price(self, instrument: Instrument) -> float | None:
        try:
            return self._client(instrument).get_current_price(self._symbol(instrument))
        except Exception as e:
            logger.debug("Price unavailable for %s: %s", instrument.symbol, e)
            return None

    def quote_volume(self, instrument: Instrument) -> float | None:
        try:
            return self._client(instrument).quote_volume_24h(self._symbol(instrument))
        except Exception:
            return None

    def spread_bps(self, instrument: Instrument) -> float | None:
        try:
            return self._client(instrument).spread_bps(self._symbol(instrument))
        except Exception:
            return None

    def order_book(self, instrument: Instrument, depth: int = 20) -> dict | None:
        try:
            return self._client(instrument).fetch_order_book(self._symbol(instrument), depth)
        except Exception:
            return None

    def funding_rate(self, instrument: Instrument) -> float | None:
        if not instrument.asset_class.pays_funding:
            return None
        try:
            return self._client(instrument).fetch_funding_rate(self._symbol(instrument))
        except Exception:
            return None

    def market_limits(self, instrument: Instrument) -> dict:
        try:
            return self._client(instrument).market_limits(self._symbol(instrument))
        except Exception:
            return {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0}
