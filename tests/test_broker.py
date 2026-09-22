"""Paper broker accounting.

These pin down the things the previous engine got wrong: margin that was
never reserved, equity that ignored open PnL, and slippage charged on base
units instead of notional.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.risk.budget import RiskBudget
from bot.trading.broker import InsufficientFunds, PaperBroker

T0 = datetime(2026, 3, 2, 6, 0, tzinfo=timezone.utc)


def make_order(config, side="long", price=100.0, atr=1.0, equity=10_000.0, risk_pct=1.0):
    return RiskBudget(config).size_order(
        symbol="BTC/USDT", side=side, entry_price=price, atr=atr,
        equity=equity, risk_pct=risk_pct,
    )


def test_opening_reserves_margin_and_charges_fees(config):
    broker = PaperBroker(config)
    order = make_order(config)
    cash_before = broker.cash

    position = broker.open(order, now=T0)

    assert broker.cash < cash_before, "margin must actually leave the cash balance"
    assert position.margin == pytest.approx(position.quantity * position.entry_price)
    assert position.entry_fee > 0
    # Nothing is created or destroyed: cash + margin + fee == starting cash.
    assert broker.cash + position.margin + position.entry_fee == pytest.approx(cash_before)


def test_margin_cannot_be_double_spent(config):
    """The old engine let every position claim the whole balance."""
    # Two positions at 60% of equity each cannot both be funded.
    config["sizing"]["max_position_pct"] = 60.0
    broker = PaperBroker(config)

    first = make_order(config, price=100.0, atr=0.2, risk_pct=1.0)
    assert first.notional == pytest.approx(6_000.0)
    broker.open(first, now=T0)

    second = make_order(config, price=100.0, atr=0.2, risk_pct=1.0)
    second.symbol = "ETH/USDT"
    with pytest.raises(InsufficientFunds):
        broker.open(second, now=T0)


def test_equity_includes_unrealized_loss(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0)

    marked_down = position.entry_price * 0.9
    equity = broker.equity({"BTC/USDT": marked_down})

    assert equity < config["paper"]["initial_balance"]
    assert broker.unrealized_pnl({"BTC/USDT": marked_down}) < 0


def test_a_stop_out_costs_about_one_r(config):
    broker = PaperBroker(config)
    order = make_order(config, price=100.0, atr=1.0, risk_pct=1.0)
    position = broker.open(order, now=T0)

    trade = broker.close(position, position.stop_price, "stop_loss",
                         now=T0 + timedelta(hours=4))

    # Exactly -1R before costs; fees and exit slippage make it slightly worse.
    assert -1.15 < trade.r_multiple < -1.0
    assert trade.pnl < 0


def test_hitting_target_pays_about_two_r(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config, price=100.0, atr=1.0), now=T0)

    trade = broker.close(position, position.take_profit, "take_profit",
                         now=T0 + timedelta(hours=4))

    assert 1.8 < trade.r_multiple < 2.0


def test_slippage_scales_with_notional_not_quantity(config):
    """0.1 BTC and 0.1 SOL are wildly different orders."""
    broker = PaperBroker(config)
    adv = 10_000_000.0

    cheap, _ = broker._fill_price(100.0, "long", 1_000, True, adv, None, 300.0)
    dear, _ = broker._fill_price(100.0, "long", 1_000_000, True, adv, None, 300.0)

    assert dear > cheap, "a larger notional must pay more impact"
    # And impact stays sane for a small order in a deep book.
    assert (cheap / 100.0 - 1) * 10_000 < 10


def test_slippage_always_works_against_the_trader(config):
    broker = PaperBroker(config)
    assert broker._fill_price(100.0, "long", 1000, True, None, None)[0] > 100.0
    assert broker._fill_price(100.0, "long", 1000, False, None, None)[0] < 100.0
    assert broker._fill_price(100.0, "short", 1000, True, None, None)[0] < 100.0
    assert broker._fill_price(100.0, "short", 1000, False, None, None)[0] > 100.0


def test_funding_charges_longs_and_pays_shorts(config):
    long_broker = PaperBroker(config)
    long_position = long_broker.open(make_order(config, side="long"), now=T0,
                                     asset_class="crypto_perp")
    long_broker.accrue_funding(T0, T0 + timedelta(hours=10),
                               {"BTC/USDT": 0.0001}, {"BTC/USDT": 100.0})

    short_broker = PaperBroker(config)
    short_position = short_broker.open(make_order(config, side="short"), now=T0,
                                       asset_class="crypto_perp")
    short_broker.accrue_funding(T0, T0 + timedelta(hours=10),
                                {"BTC/USDT": 0.0001}, {"BTC/USDT": 100.0})

    assert long_position.funding_paid > 0, "a long pays positive funding"
    assert short_position.funding_paid < 0, "a short receives it"


def test_funding_only_accrues_across_settlements(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0, asset_class="crypto_perp")

    # 06:00 to 07:00 crosses no settlement (00/08/16 UTC).
    broker.accrue_funding(T0, T0 + timedelta(hours=1), {"BTC/USDT": 0.01}, {})
    assert position.funding_paid == 0.0

    broker.accrue_funding(T0, T0 + timedelta(hours=3), {"BTC/USDT": 0.01}, {})
    assert position.funding_paid > 0


def test_only_perpetuals_pay_funding(config):
    """A stock has no funding leg, and charging it one every eight hours
    quietly taxes the equity half of the book for a cost it never incurs.
    """
    for asset_class in ("crypto_spot", "equity", "etf"):
        broker = PaperBroker(config)
        position = broker.open(make_order(config), now=T0,
                               asset_class=asset_class)
        broker.accrue_funding(T0, T0 + timedelta(hours=10),
                              {"BTC/USDT": 0.01}, {"BTC/USDT": 100.0})
        assert position.funding_paid == 0.0, asset_class
        assert broker.funding_paid == 0.0, asset_class


def test_a_record_with_no_asset_class_still_pays_funding(config):
    """Every position written before asset classes existed was a perp."""
    broker = PaperBroker(config)
    position = broker.open(make_order(config), now=T0, asset_class="")
    broker.accrue_funding(T0, T0 + timedelta(hours=10),
                          {"BTC/USDT": 0.01}, {"BTC/USDT": 100.0})
    assert position.funding_paid > 0


def test_stop_wins_when_a_bar_spans_both_barriers(config):
    """Without tick data, assuming the target filled first is how a
    backtest lies to you."""
    broker = PaperBroker(config)
    position = broker.open(make_order(config, price=100.0, atr=1.0), now=T0)

    reason = broker.stop_or_target_hit(
        position, high=position.take_profit + 1, low=position.stop_price - 1
    )
    assert reason == "stop_loss"


def test_mae_and_mfe_are_recorded_in_r(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config, price=100.0, atr=1.0), now=T0)

    broker.mark("BTC/USDT", position.entry_price + 3.0)
    broker.mark("BTC/USDT", position.entry_price - 1.0)
    trade = broker.close(position, position.entry_price, "signal", now=T0)

    assert trade.mfe_r > 0.5
    assert trade.mae_r < 0


def test_closing_returns_margin_to_cash(config):
    broker = PaperBroker(config)
    start = broker.cash
    position = broker.open(make_order(config), now=T0)
    trade = broker.close(position, position.entry_price, "signal", now=T0)

    # Back to start, less the round-trip costs.
    assert broker.cash < start
    assert broker.cash == pytest.approx(start - position.entry_fee + trade.pnl)
    assert broker.reserved_margin() == 0


def test_a_trailing_stop_in_profit_is_not_reported_as_a_loss(config):
    """A stop trailed past the entry banks a profit; calling that
    "stop_loss" made the exit table unreadable — a replay showed
    stop_loss exits at +0.54R sitting alongside real losses."""
    broker = PaperBroker(config)
    position = broker.open(make_order(config, price=100.0, atr=1.0), now=T0)

    position.stop_price = position.entry_price + 2.0      # trailed into profit
    reason = broker.stop_or_target_hit(
        position, high=position.entry_price + 2.5,
        low=position.entry_price + 1.0,
    )
    assert reason == "trailing_stop"


def test_a_stop_at_breakeven_is_named_as_such(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config, price=100.0, atr=1.0), now=T0)

    position.stop_price = position.entry_price
    position.moved_to_breakeven = True
    reason = broker.stop_or_target_hit(
        position, high=position.entry_price + 1.0, low=position.entry_price - 0.5,
    )
    assert reason == "breakeven_stop"


def test_the_original_stop_is_still_a_loss(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config, price=100.0, atr=1.0), now=T0)
    reason = broker.stop_or_target_hit(
        position, high=position.entry_price, low=position.stop_price - 0.5,
    )
    assert reason == "stop_loss"


def test_a_short_trailed_into_profit_is_also_a_trailing_stop(config):
    broker = PaperBroker(config)
    position = broker.open(make_order(config, side="short", price=100.0, atr=1.0),
                           now=T0)

    position.stop_price = position.entry_price - 2.0      # below entry = profit
    reason = broker.stop_or_target_hit(
        position, high=position.entry_price - 1.0,
        low=position.entry_price - 2.5,
    )
    assert reason == "trailing_stop"
