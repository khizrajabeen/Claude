"""Pyramiding must add size, not risk.

This is the bug that made the strategy lose money, and it hid behind a
comment claiming the opposite. `add_to` left the stop where it was, on
the stated grounds that moving it would "quietly increase the risk it
was sized for". Leaving it is what increases the risk: the added unit
carries its own full stop distance, so a half-size add on a 1R position
takes the real risk to 2R.

Across 87 replayed trades it showed up as an average loss of 1.43R
against an average win of 0.77R, with 92% of losses exceeding the 1R
the position was supposedly sized to.
"""

import pytest

from bot.risk.budget import SizedOrder
from bot.trading.broker import PaperBroker

CFG = {"paper": {"initial_balance": 100_000, "fee_taker_bps": 25,
                 "slippage_bps": 0, "impact_coefficient": 0,
                 "funding_enabled": False}}


def order(qty, entry, stop, side="long", risk=None):
    r = abs(entry - stop)
    return SizedOrder(symbol="X/USD", side=side, quantity=qty, entry_price=entry,
                      stop_price=stop, take_profit=entry + (1 if side == "long" else -1) * 2 * r,
                      risk_usd=risk if risk is not None else r * qty,
                      notional=qty * entry, leverage=1.0, atr=r / 2,
                      r_distance=r, asset_class="crypto_spot")


def risk_now(p):
    return abs(p.entry_price - p.stop_price) * p.quantity


def test_a_pyramid_holds_the_original_risk_budget():
    b = PaperBroker(CFG)
    p = b.open(order(10.0, 100.0, 96.0), day="d", asset_class="crypto_spot")
    assert risk_now(p) == pytest.approx(40.0)

    b.add_to(p, order(5.0, 104.0, 100.0))
    assert p.quantity == 15.0, "the size did grow"
    assert risk_now(p) == pytest.approx(40.0, abs=0.01), \
        "but the risk did not — that was the whole bug"


def test_two_pyramids_still_hold_it():
    b = PaperBroker(CFG)
    p = b.open(order(10.0, 100.0, 96.0), day="d", asset_class="crypto_spot")
    b.add_to(p, order(5.0, 104.0, 100.0))
    b.add_to(p, order(5.0, 108.0, 104.0))
    assert p.quantity == 20.0
    assert risk_now(p) == pytest.approx(40.0, abs=0.01)


def test_a_stopped_pyramid_loses_about_one_r_not_two():
    b = PaperBroker(CFG)
    p = b.open(order(10.0, 100.0, 96.0), day="d", asset_class="crypto_spot")
    b.add_to(p, order(5.0, 104.0, 100.0))
    trade = b.close(p, p.stop_price, reason="stop_loss", day="d")
    # Fees put it slightly past 1R; the old behaviour was -1.567R.
    assert -1.05 < trade.r_multiple < -0.5, trade.r_multiple


def test_shorts_tighten_the_other_way():
    b = PaperBroker(CFG)
    p = b.open(order(10.0, 100.0, 104.0, side="short"),
               day="d", asset_class="crypto_spot")
    assert risk_now(p) == pytest.approx(40.0)
    b.add_to(p, order(5.0, 96.0, 100.0, side="short"))
    assert p.stop_price < 104.0, "a short's stop tightens downward"
    assert risk_now(p) == pytest.approx(40.0, abs=0.01)


def test_a_trailed_stop_is_not_loosened_by_an_add():
    """If the trail has already moved past the risk-budget point, it is
    the tighter of the two and must stay."""
    b = PaperBroker(CFG)
    p = b.open(order(10.0, 100.0, 96.0), day="d", asset_class="crypto_spot")
    p.stop_price = 101.0                 # trailed into profit
    b.add_to(p, order(5.0, 104.0, 100.0))
    assert p.stop_price == 101.0, "an add must never loosen a trailed stop"


def test_breakeven_after_a_pyramid_is_not_a_loss():
    """The contradiction the old code produced: a stop parked at the
    ORIGINAL entry while the averaged entry sat above it, so 'breakeven'
    lost 1.25R by construction."""
    b = PaperBroker(CFG)
    p = b.open(order(10.0, 100.0, 96.0), day="d", asset_class="crypto_spot")
    b.add_to(p, order(5.0, 104.0, 100.0))
    assert p.stop_price < p.entry_price, "still a stop, not a target"
    # Breakeven means the AVERAGED entry, which is where a flat exit sits.
    trade = b.close(p, p.entry_price, reason="breakeven_stop", day="d")
    assert trade.r_multiple > -0.25, \
        f"a flat exit should be about 0R, got {trade.r_multiple}"
