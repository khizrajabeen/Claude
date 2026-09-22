"""What every data source must provide.

The router below hands an instrument to whichever provider claims its
asset class, so the briefing does not need to know whether it is looking at
a crypto perp or a US equity. That separation is what makes the daily
session asset-class-agnostic.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from bot.markets.instrument import AssetClass, Instrument


@runtime_checkable
class DataProvider(Protocol):
    """A source of bars and quotes for one or more asset classes."""

    name: str

    def handles(self, instrument: Instrument) -> bool:
        ...

    def bars(self, instrument: Instrument, timeframe: str, limit: int) -> pd.DataFrame:
        """UTC-indexed OHLCV, oldest first, most recent bar last."""
        ...

    def price(self, instrument: Instrument) -> float | None:
        ...

    def quote_volume(self, instrument: Instrument) -> float | None:
        """Recent traded value in quote currency, for the liquidity filter."""
        ...

    def order_book(self, instrument: Instrument, depth: int = 20) -> dict | None:
        """Depth of book, or None where the venue does not publish it."""
        ...

    def funding_rate(self, instrument: Instrument) -> float | None:
        """Per-settlement funding as a decimal, or None where it does not apply."""
        ...


class ProviderError(Exception):
    """A provider could not serve a request. Callers skip the instrument
    rather than failing the day."""
