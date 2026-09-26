"""Every reported R must be arithmetic, not assertion.

A reviewer worked one ETH trade by hand and found 0.143R the record did
not account for. Recomputing all 43 trades of that replay showed the
same gap on every one of them, and it was a real bug: `close()` charged
the exit fee to the trade's P&L but not the entry fee, while reporting
both under `fees`. The equity curve was right because the entry fee had
already left cash at `open()`; only the per-trade numbers were wrong,
and every conclusion drawn from them inherited it.

These tests fail if a trade stops being reconstructible from its own
columns.
"""

import pytest

from bot.risk.budget import SizedOrder
from bot.trading.broker import PaperBroker
from bot.utils.reconcile import Row

CFG = {"paper": {"initial_balance": 100_000, "fee_taker_bps": 25,
                 "slippage_bps": 0, "impact_coefficient": 0,
                 "funding_enabled": False}}


def order(qty=10.0, entry=100.0, stop=96.0, side="long"):
    r = abs(entry - stop)
    return SizedOrder(symbol="X/USD", side=side, quantity=qty, entry_price=entry,
                      stop_price=stop, take_profit=entry + 2 * r, risk_usd=r * qty,
                      notional=qty * entry, leverage=1.0, atr=r / 2,
                      r_distance=r, asset_class="crypto_spot")


def row_from(trade) -> Row:
    d = trade.to_dict()
    return Row(symbol=d["symbol"], side=d["side"], entry=d["entry_price"],
               exit=d["exit_price"], quantity=d["quantity"],
               original_quantity=d["original_quantity"],
               initial_stop=d["initial_stop"], final_stop=d["final_stop"],
               units=d["units"], realized_before_exit=d["realized_before_exit"],
               exit_quantity=d["exit_quantity"], pnl=d["pnl"], fees=d["fees"],
               funding=d["funding"], r_reported=d["r_multiple"],
               exit_reason=d["exit_reason"], initial_risk_usd=d["risk_usd"],
               scale_out_fees=d["scale_out_fees"])


def assert_reconciles(trade):
    r = row_from(trade)
    assert r.risk_usd > 0, "no denominator to measure R against"
    assert r.net == pytest.approx(r.pnl, abs=0.01), \
        f"cash: recomputed {r.net:.4f} vs reported {r.pnl:.4f}"
    assert r.r_computed == pytest.approx(r.r_reported, abs=0.005), \
        f"R: recomputed {r.r_computed:.4f} vs reported {r.r_reported:.4f}"
    return r


def test_a_plain_stop_out_reconciles():
    b = PaperBroker(CFG)
    p = b.open(order(), day="d", asset_class="crypto_spot")
    assert_reconciles(b.close(p, p.stop_price, reason="stop_loss", day="d"))


def test_the_entry_fee_is_charged_to_the_trade_exactly_once():
    """It left cash at open, so it must not leave again at close — but it
    must still appear in the trade's own P&L."""
    b = PaperBroker(CFG)
    p = b.open(order(), day="d", asset_class="crypto_spot")
    entry_fee = p.entry_fee
    cash_before = b.cash
    t = b.close(p, p.entry_price, reason="signal", day="d")

    r = assert_reconciles(t)
    assert t.fees == pytest.approx(entry_fee + (t.exit_price * t.quantity * 0.0025),
                                   rel=0.02)
    # Flat exit: the trade lost exactly its two fees.
    assert t.pnl == pytest.approx(-t.fees, abs=0.01)
    # Cash only moved by the exit leg, because the entry fee already went.
    assert b.cash - cash_before == pytest.approx(
        p.margin - (t.exit_price * t.quantity * 0.0025), abs=0.01)


def test_a_scaled_out_trade_reconciles():
    """`pnl` folds in profit banked before the exit; the exit price alone
    cannot show it, which is what made every trade fail by up to $48."""
    b = PaperBroker(CFG)
    p = b.open(order(), day="d", asset_class="crypto_spot")
    b.scale_out(p, 108.0, 0.33, reason="take_partial")
    t = b.close(p, 96.0, reason="stop_loss", day="d")
    r = assert_reconciles(t)
    assert r.realized_before_exit > 0, "the banked leg must be recorded"
    assert r.exit_quantity < r.original_quantity


def test_a_pyramided_trade_reconciles():
    b = PaperBroker(CFG)
    p = b.open(order(), day="d", asset_class="crypto_spot")
    b.add_to(p, order(qty=5.0, entry=104.0, stop=100.0))
    t = b.close(p, p.stop_price, reason="stop_loss", day="d")
    r = assert_reconciles(t)
    assert r.units == 2


def test_a_short_reconciles():
    b = PaperBroker(CFG)
    p = b.open(order(side="short", entry=100.0, stop=104.0),
               day="d", asset_class="crypto_spot")
    assert_reconciles(b.close(p, 96.0, reason="take_profit", day="d"))


def test_slippage_past_the_stop_is_measurable():
    """The decomposition the reviewer asked for: how much of a loss is
    the price path, and how much is filling worse than the stop."""
    b = PaperBroker(CFG)
    p = b.open(order(), day="d", asset_class="crypto_spot")
    t = b.close(p, 94.0, reason="stop_loss", day="d")   # 2 past the stop
    r = row_from(t)
    assert r.slipped_past_stop > 0.4, r.slipped_past_stop
    assert r.r_from_fees < 0
