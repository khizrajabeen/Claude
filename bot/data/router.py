"""One door to every market.

The briefing asks the router for bars and never learns whether the answer
came from a crypto exchange or an equity endpoint. That is what lets a
single daily session hold a Bitcoin perp and a Nasdaq stock at the same
time without either being a special case.

The router also owns the per-instrument bar cache. A bench replays the same
history for every variant, and an equity endpoint that rate-limits bursts
will not survive that without one.
"""

from __future__ import annotations

import logging

import pandas as pd

from bot.data.base import ProviderError
from bot.data.alpaca import AlpacaProvider
from bot.data.crypto import CryptoProvider
from bot.data.equity import EquityProvider
from bot.markets.instrument import Instrument

logger = logging.getLogger("trading_bot")


class DataRouter:
    """Routes each instrument to the provider that handles its asset class."""

    def __init__(self, config: dict, providers: list | None = None):
        self.config = config
        # Order matters: Alpaca is asked first because it serves intraday
        # bars and quotes, and it declines when no keys are set — so the
        # daily feed behind it is the fallback rather than the default.
        self.providers = providers if providers is not None else [
            CryptoProvider(config),
            AlpacaProvider(config),
            EquityProvider(config),
        ]
        self._unhandled: set[str] = set()

    def provider_for(self, instrument: Instrument):
        for provider in self.providers:
            if provider.handles(instrument):
                return provider
        if instrument.key not in self._unhandled:
            logger.warning("No provider handles %s (%s)",
                           instrument.symbol, instrument.asset_class.value)
            self._unhandled.add(instrument.key)
        return None

    # ── Market data ───────────────────────────────────────────

    def bars(self, instrument: Instrument, timeframe: str | None = None,
             limit: int = 500) -> pd.DataFrame:
        """Bars for an instrument, empty when unavailable.

        Returning empty rather than raising is deliberate: one dead symbol
        must not take down the day. The briefing marks it untradable and
        moves on.
        """
        provider = self.provider_for(instrument)
        if provider is None:
            return pd.DataFrame()
        try:
            return provider.bars(instrument, timeframe or instrument.timeframe, limit)
        except (ProviderError, Exception) as e:
            logger.debug("Bars unavailable for %s: %s", instrument.symbol, e)
            return pd.DataFrame()

    def price(self, instrument: Instrument) -> float | None:
        provider = self.provider_for(instrument)
        return provider.price(instrument) if provider else None

    def prices(self, instruments: list[Instrument]) -> dict[str, float]:
        """Current price per instrument key, skipping any that fail."""
        out: dict[str, float] = {}
        for instrument in instruments:
            price = self.price(instrument)
            if price is not None:
                out[instrument.symbol] = price
        return out

    def quote_volume(self, instrument: Instrument) -> float | None:
        provider = self.provider_for(instrument)
        return provider.quote_volume(instrument) if provider else None

    def spread_bps(self, instrument: Instrument) -> float | None:
        provider = self.provider_for(instrument)
        getter = getattr(provider, "spread_bps", None) if provider else None
        return getter(instrument) if getter else None

    def order_book(self, instrument: Instrument, depth: int = 20) -> dict | None:
        provider = self.provider_for(instrument)
        getter = getattr(provider, "order_book", None) if provider else None
        return getter(instrument, depth) if getter else None

    def funding_rate(self, instrument: Instrument) -> float | None:
        provider = self.provider_for(instrument)
        return provider.funding_rate(instrument) if provider else None

    def market_limits(self, instrument: Instrument) -> dict:
        provider = self.provider_for(instrument)
        getter = getattr(provider, "market_limits", None) if provider else None
        if getter:
            return getter(instrument)
        return {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0}
