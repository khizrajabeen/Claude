"""Independent review regressions: failed on a9467a4; must now pass."""
from datetime import datetime, timezone

import pandas as pd
import pytest

from bot.daily.session import SimulatedClock
from bot.risk.budget import SizedOrder
from bot.trading.broker import PaperBroker
from bot.utils.backtester import ReplayExchange
from test_reconcile import row_from


def broker(fees=0):
    return PaperBroker({"paper": {"initial_balance": 10000,
        "fee_taker_bps": fees, "slippage_bps": 0,
        "impact_coefficient": 0, "funding_enabled": False}})


def order(quantity=10, entry=100):
    return SizedOrder(symbol="X/USD", side="long", quantity=quantity,
        entry_price=entry, stop_price=entry-4, take_profit=entry+8,
        risk_usd=quantity*4, notional=quantity*entry, leverage=1,
        atr=2, r_distance=4, asset_class="crypto_spot")


def test_pyramid_r_uses_the_original_dollar_budget():
    b = broker()
    p = b.open(order())
    original_risk = p.risk_usd
    b.add_to(p, order(5, 104))
    t = b.close(p, p.stop_price, "stop_loss")
    assert t.pnl == pytest.approx(-original_risk)
    assert t.r_multiple == pytest.approx(t.pnl / original_risk)


def test_gross_and_fees_include_every_scale_out_leg():
    b = broker(fees=25)
    p = b.open(order())
    b.scale_out(p, 108, 0.5)
    t = b.close(p, 108, "take_profit")
    assert t.pnl == pytest.approx(b.cash - 10000)
    assert t.fees == pytest.approx(b.fees_paid)
    assert row_from(t).gross == pytest.approx(80)


def test_stop_shortfall_uses_quantity_executed_at_stop():
    b = broker()
    p = b.open(order())
    b.scale_out(p, 104, 0.5)
    t = b.close(p, 95, "stop_loss")
    # Five remaining units each fill $1 below the $96 stop; initial risk $40.
    assert row_from(t).slipped_past_stop == pytest.approx(5 / 40)


def test_open_timestamp_bar_is_not_complete_at_its_open():
    at = datetime(2026, 1, 1, tzinfo=timezone.utc)
    frame = pd.DataFrame({"open": [100.], "high": [200.], "low": [90.],
        "close": [150.], "volume": [10.]}, index=pd.DatetimeIndex([at]))
    replay = ReplayExchange({"X/USD": {"1h": frame}}, SimulatedClock(at))
    # This is the opening-timestamp convention preserved by ohlcv_to_frame.
    assert replay.fetch_ohlcv("X/USD", "1h").empty
