"""Building the universe from what the venue charges.

The old universe was fifteen pairs copied from an exchange's volume
ranking, all assumed to cost the same 5.5 bps. Two separate mistakes,
and the tests here guard the corrections: cost is per instrument and
measured, and the thing that measures liquidity is the spread, not the
size resting at the touch.
"""

import pytest

from bot.markets.alpaca_universe import (STABLECOINS, TAKER_BPS, measure,
                                         render, to_config_block)


class FakeSession:
    def __init__(self, quotes):
        self.quotes = quotes
        self.asked = None

    def get(self, url, params=None, timeout=None):
        self.asked = params
        return self

    def json(self):
        return {"quotes": self.quotes}


class FakeBroker:
    def __init__(self, quotes, tradable=None):
        self._session = FakeSession(quotes)
        self._tradable = tradable

    def session(self):
        return self._session

    def tradable(self):
        if self._tradable is not None:
            return self._tradable
        return set(self._session.quotes) | {"AAPL"}


def q(ask, bid, asize=1.0, bsize=1.0):
    return {"ap": ask, "bp": bid, "as": asize, "bs": bsize}


def test_cost_is_the_spread_plus_the_fee_on_both_sides():
    # A 10 bps spread on a $100 mid: ask 100.05, bid 99.95.
    rows = measure(FakeBroker({"BTC/USD": q(100.05, 99.95)}))
    assert len(rows) == 1
    row = rows[0]
    assert row["spread_bps"] == pytest.approx(10.0, abs=0.01)
    assert row["round_trip_bps"] == pytest.approx(10.0 + TAKER_BPS * 2, abs=0.01)
    assert row["slippage_bps"] == pytest.approx(5.0, abs=0.01), \
        "crossing costs half the spread per side"
    assert row["fee_bps"] == TAKER_BPS


def test_a_high_priced_coin_is_not_excluded_for_a_small_top_of_book():
    """Depth at the touch is quoted in coin units.

    BTC shows about $83 resting at a 1.6 bps spread while a dead
    altcoin shows $1 at 59. A dollar depth floor calibrated for the
    second excluded the first — and BTC, ETH and SOL with it, which was
    every liquid asset on the venue.
    """
    rows = measure(FakeBroker({
        "BTC/USD": q(83_246, 83_232, 0.001, 0.001),     # ~$83 at the touch
    }))
    assert [r["symbol"] for r in rows] == ["BTC/USD"]


def test_a_plainly_broken_quote_is_dropped():
    rows = measure(FakeBroker({
        "GOOD/USD": q(100.05, 99.95),
        "BROKEN/USD": q(100.0, 0.0),        # no bid at all
        "CROSSED/USD": q(99.0, 101.0),      # ask below bid
    }))
    assert [r["symbol"] for r in rows] == ["GOOD/USD"]


def test_hopeless_pairs_are_kept_out_of_the_ranking():
    rows = measure(FakeBroker({
        "TIGHT/USD": q(100.01, 99.99),
        "AWFUL/USD": q(110.0, 90.0),        # 2000 bps wide
    }), max_round_trip_bps=160.0)
    assert [r["symbol"] for r in rows] == ["TIGHT/USD"]


def test_stablecoins_are_not_coins():
    """Holding a dollar against a dollar has no trend to trade."""
    quotes = {f"{s}/USD": q(1.0005, 0.9995) for s in STABLECOINS}
    quotes["BTC/USD"] = q(100.05, 99.95)
    rows = measure(FakeBroker(quotes))
    assert [r["symbol"] for r in rows] == ["BTC/USD"]


def test_only_usd_quotes_are_considered():
    """BTC/USDT and ETH/BTC are the same exposure in different wrappers."""
    b = FakeBroker({"BTC/USD": q(100.05, 99.95)},
                   tradable={"BTC/USD", "BTC/USDT", "ETH/BTC", "AAPL"})
    measure(b)
    asked = b.session().asked["symbols"].split(",")
    assert asked == ["BTC/USD"]


def test_cheapest_first_because_that_is_the_ranking_that_matters():
    rows = measure(FakeBroker({
        "WIDE/USD": q(100.5, 99.5),
        "TIGHT/USD": q(100.01, 99.99),
        "MID/USD": q(100.1, 99.9),
    }))
    assert [r["symbol"] for r in rows] == ["TIGHT/USD", "MID/USD", "WIDE/USD"]


def test_the_config_block_carries_each_instrument_its_own_cost():
    rows = measure(FakeBroker({"BTC/USD": q(100.05, 99.95)}))
    block = to_config_block(rows)
    assert 'symbol: "BTC/USD"' in block
    assert 'venue: "alpaca"' in block
    assert "fee_bps: 25.0" in block
    assert "slippage_bps: 5.0" in block


def test_an_empty_venue_renders_a_sentence_not_a_crash():
    assert "No tradable pairs" in render([])
