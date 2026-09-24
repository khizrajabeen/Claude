"""Scaling out of winners and pyramiding into them.

The trailing stop protects a winner. These are the same idea at the other
two ends: scaling out converts part of an open gain to cash, pyramiding
presses a position that is working.

The failure this mostly guards against is a bookkeeping one. A position
that sells half at 2R and then stops at breakeven made money, and a record
that shows it as flat would make every scale-out look like a wasted trade.
"""

from datetime import datetime, timedelta, timezone

import pytest

from bot.risk.budget import SizedOrder
from bot.trading.broker import PaperBroker

T0 = datetime(2026, 5, 4, 6, 0, tzinfo=timezone.utc)


def make_order(config, side="long", price=100.0, atr=1.0, qty=10.0):
    stop = price - (1 if side == "long" else -1) * 2 * atr
    return SizedOrder(
        symbol="BTC/USDT", side=side, quantity=qty, entry_price=price,
        stop_price=stop, take_profit=price + (1 if side == "long" else -1) * 4 * atr,
        risk_usd=qty * 2 * atr, notional=qty * price, leverage=1.0,
        atr=atr, r_distance=2 * atr,
    )


# ── Scaling out ──────────────────────────────────────────────

def test_selling_a_slice_reduces_the_position_and_banks_cash(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    before_qty, before_cash = position.quantity, broker.cash

    net = broker.scale_out(position, 104.0, 0.5, now=T0)
    assert position.quantity == pytest.approx(before_qty * 0.5)
    assert net > 0
    assert broker.cash > before_cash


def test_the_remaining_position_keeps_its_original_entry_and_stop(config):
    """Re-basing the cost would make the R multiple measure against
    something other than what was actually risked."""
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    entry, stop = position.entry_price, position.initial_stop

    broker.scale_out(position, 104.0, 0.5, now=T0)
    assert position.entry_price == pytest.approx(entry)
    assert position.initial_stop == pytest.approx(stop)


def test_banked_profit_survives_into_the_closed_trade(config):
    """Sold half at a profit, then stopped at breakeven: that trade made
    money, and reporting only the final leg would show it as flat."""
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)

    broker.scale_out(position, 108.0, 0.5, now=T0)
    trade = broker.close(position, position.entry_price, "stop_loss", now=T0)

    assert trade.pnl > 0, "the banked half is part of this trade"
    assert trade.r_multiple > 0


def test_r_is_measured_against_the_size_that_was_risked(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    sized = position.quantity

    broker.scale_out(position, 104.0, 0.5, now=T0)
    trade = broker.close(position, 104.0, "take_profit", now=T0)
    assert trade.quantity == pytest.approx(sized)


def test_scaling_out_records_where_it_happened(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    broker.scale_out(position, 104.0, 0.5, now=T0)
    assert position.scaled_out_at
    assert position.scaled_out_at[0] == pytest.approx(2.0, abs=0.3)


def test_a_zero_fraction_does_nothing(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    qty = position.quantity
    assert broker.scale_out(position, 104.0, 0.0, now=T0) == 0.0
    assert position.quantity == qty


def test_scaling_out_releases_margin_proportionally(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    margin = position.margin
    broker.scale_out(position, 104.0, 0.5, now=T0)
    assert position.margin == pytest.approx(margin * 0.5)


def test_a_loser_can_be_scaled_out_of_too(config):
    """Nothing here assumes a profit; the caller decides when."""
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    net = broker.scale_out(position, 99.0, 0.5, now=T0)
    assert net < 0
    assert position.realized_pnl < 0


# ── Pyramiding ───────────────────────────────────────────────

def test_adding_a_unit_grows_the_position(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    before = position.quantity

    assert broker.add_to(position, make_order(config, price=104.0, qty=5.0), now=T0)
    assert position.quantity == pytest.approx(before + 5.0)
    assert position.units == 2


def test_the_combined_entry_is_a_weighted_average(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config, price=100.0, qty=10.0), now=T0)
    first = position.entry_price

    broker.add_to(position, make_order(config, price=110.0, qty=10.0), now=T0)
    assert first < position.entry_price < 110.0


def test_a_pyramid_holds_the_risk_it_was_sized_for(config):
    """A pyramid adds size, not risk.

    This test used to assert the opposite and state the right reason
    for it: that a pyramid "has quietly increased the risk it was sized
    for". It then checked that the stop had NOT moved — which is the
    thing that increases the risk. The added unit carries its own full
    stop distance, so leaving the stop alone lets exposure grow with
    size while the stop stays put.

    Measured on a 10-unit position risking $40, one half-size add took
    the real risk to $80 and the stop-out lost 1.57R. Holding the stop
    still was the bug; the test was guarding it.
    """
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    before = abs(position.entry_price - position.stop_price) * position.quantity
    stop = position.stop_price

    broker.add_to(position, make_order(config, price=106.0, qty=5.0), now=T0)
    after = abs(position.entry_price - position.stop_price) * position.quantity

    assert position.quantity > 10.0, "the size grew"
    assert after == pytest.approx(before, rel=0.02), "the risk did not"
    assert position.stop_price > stop, "and the stop tightened to hold it"


def test_adding_without_cash_is_refused_not_crashed(config):
    config["paper"]["initial_balance"] = 1200.0
    broker = PaperBroker(config)
    position = broker.open(make_order(config, qty=10.0), now=T0)
    assert not broker.add_to(position, make_order(config, qty=1000.0), now=T0)
    assert position.units == 1


def test_adding_nothing_is_refused(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    assert not broker.add_to(position, make_order(config, qty=0.0), now=T0)


def test_a_pyramided_position_charges_its_fees(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)
    fees = position.entry_fee
    broker.add_to(position, make_order(config, price=104.0, qty=5.0), now=T0)
    assert position.entry_fee > fees


# ── The session's rules ──────────────────────────────────────

def _session(config, **scaling):
    from tests.test_session import build_session
    config.setdefault("scaling", {}).update(scaling)
    session, _, broker, clock = build_session(config)
    return session, broker, clock


def test_each_profit_rung_fires_only_once(config):
    session, broker, clock = _session(
        config, take_profit_at=[{"at_r": 1.0, "fraction": 0.3}],
        scale_out_classes=[], max_units=1)
    position = broker.open(make_order(config), now=clock.now())

    session._book_profit(position, 104.0, clock.now())
    after_first = position.quantity
    session._book_profit(position, 104.0, clock.now())
    assert position.quantity == pytest.approx(after_first)
    assert position.scaled_out_at_levels == [1.0]


def test_nothing_is_booked_below_the_first_rung(config):
    session, broker, clock = _session(
        config, take_profit_at=[{"at_r": 2.0, "fraction": 0.5}],
        scale_out_classes=[], max_units=1)
    position = broker.open(make_order(config), now=clock.now())
    qty = position.quantity
    session._book_profit(position, 100.5, clock.now())
    assert position.quantity == pytest.approx(qty)


def test_a_class_outside_the_list_is_left_alone(config):
    session, broker, clock = _session(
        config, take_profit_at=[{"at_r": 1.0, "fraction": 0.5}],
        scale_out_classes=["equity"], max_units=1)
    position = broker.open(make_order(config), now=clock.now(),
                           asset_class="crypto_perp")
    qty = position.quantity
    session._book_profit(position, 108.0, clock.now())
    assert position.quantity == pytest.approx(qty)


def test_scaling_out_never_closes_the_last_of_a_position(config):
    """The stop and target own the exit; a position closed by a scale-out
    would bypass the trade record entirely."""
    session, broker, clock = _session(
        config, take_profit_at=[{"at_r": 1.0, "fraction": 1.0}],
        scale_out_classes=[], max_units=1)
    position = broker.open(make_order(config), now=clock.now())
    session._book_profit(position, 110.0, clock.now())
    assert position.quantity > 0
