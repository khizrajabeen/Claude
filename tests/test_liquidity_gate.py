"""The liquidity gate, and why volume is not universally liquidity.

A $5M daily-volume floor is sensible on a venue that matches orders in
its own book. Applied to Alpaca it rejected all thirty coins including
BTC — because Alpaca routes crypto to external market makers and
reports only what crossed its own tape, roughly $330k a day on BTC
against hundreds of millions globally, while quoting a 2 bps spread.

The backtest returned no trades at all and said nothing about why, which
is the failure this guards.
"""

import pytest

from bot.daily.briefing import BriefingBuilder, SymbolRead


class _NoRouter:
    """The filters need no market data; they read a SymbolRead."""

    def bars(self, *a, **kw):
        return None

    def price(self, *a, **kw):
        return None


class _NoNews:
    def refresh(self, *a, **kw):
        return None

    def track_universe(self, *a, **kw):
        return None


def gate(config=None):
    return BriefingBuilder(config or {}, _NoRouter(), news=_NoNews(), universe=[])


def read(**kw):
    defaults = dict(symbol="BTC/USD", price=80_000.0, atr_pct=1.5,
                    asset_class="crypto_spot", venue="alpaca")
    return SymbolRead(**{**defaults, **kw})


def test_thin_reported_volume_does_not_reject_an_alpaca_pair():
    b = gate()
    r = read(quote_volume_24h=330_000.0, spread_bps=2.0)
    b._apply_filters(r)
    assert r.tradable, r.skip_reason


def test_the_same_volume_does_reject_an_exchange_pair():
    """Where the venue matches in its own book, volume means what it says."""
    b = gate()
    r = read(venue="okx", quote_volume_24h=330_000.0, spread_bps=2.0)
    b._apply_filters(r)
    assert not r.tradable
    assert "thin" in r.skip_reason


def test_spread_still_gates_an_alpaca_pair():
    """Dropping the volume check does not drop the cost check."""
    b = gate({"filters": {"max_spread_bps": 60}})
    r = read(quote_volume_24h=330_000.0, spread_bps=140.0)
    b._apply_filters(r)
    assert not r.tradable
    assert "spread" in r.skip_reason


def test_which_venues_count_is_configurable():
    b = gate({"filters": {"volume_is_liquidity": ["alpaca"],
                          "min_quote_volume_24h": 5_000_000}})
    r = read(quote_volume_24h=330_000.0, spread_bps=2.0)
    b._apply_filters(r)
    assert not r.tradable, "an explicit config must still be honoured"


def test_an_unknown_venue_is_not_gated_on_volume():
    """Better to let the spread decide than to reject on a number whose
    meaning is unknown."""
    b = gate()
    r = read(venue="", quote_volume_24h=1.0, spread_bps=3.0)
    b._apply_filters(r)
    assert r.tradable
