"""Risk budgeting: volatility-targeted sizing and the portfolio gates."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.risk.budget import RiskBudget, symbol_beta
from bot.trading.models import Position

NOW = datetime(2026, 3, 2, 12, 0, tzinfo=timezone.utc)


def position(symbol="BTC/USDT", side="long", entry=100.0, qty=1.0, stop=98.0):
    return Position(
        id=symbol, symbol=symbol, side=side, entry_price=entry, quantity=qty,
        leverage=1.0, opened_at=NOW, stop_price=stop, initial_stop=stop,
    )


def test_dollar_risk_is_constant_across_volatility(config):
    """The point of ATR sizing: a calm asset and a wild one risk the same."""
    config["sizing"]["max_position_pct"] = 100.0
    budget = RiskBudget(config)

    calm = budget.size_order("A/USDT", "long", 100.0, atr=0.5, equity=10_000, risk_pct=1.0)
    wild = budget.size_order("B/USDT", "long", 100.0, atr=5.0, equity=10_000, risk_pct=1.0)

    assert calm.risk_usd == pytest.approx(100.0)
    assert wild.risk_usd == pytest.approx(100.0)
    # Same risk, very different size.
    assert calm.quantity == pytest.approx(wild.quantity * 10)


def test_stop_sits_an_atr_multiple_away(config):
    budget = RiskBudget(config)
    long_order = budget.size_order("A/USDT", "long", 100.0, atr=2.0, equity=10_000, risk_pct=1.0)
    short_order = budget.size_order("A/USDT", "short", 100.0, atr=2.0, equity=10_000, risk_pct=1.0)

    assert long_order.stop_price == pytest.approx(96.0)   # 100 - 2 x 2
    assert long_order.take_profit == pytest.approx(108.0)  # 2R
    assert short_order.stop_price == pytest.approx(104.0)
    assert short_order.take_profit == pytest.approx(92.0)


def test_degenerate_atr_cannot_produce_an_infinite_position(config):
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 100.0, atr=0.0, equity=10_000, risk_pct=1.0)

    assert "min_stop_floor" in order.caps_applied
    assert order.r_distance > 0
    assert order.notional <= 10_000 * config["sizing"]["max_position_pct"] / 100 + 1


def test_reported_risk_reflects_caps_not_the_budget(config):
    """A capped order risks less than the budget; the heat check must see
    the real number, not the intended one."""
    config["sizing"]["max_position_pct"] = 5.0
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 100.0, atr=0.1, equity=10_000, risk_pct=1.0)

    assert "max_position_pct" in order.caps_applied
    assert order.risk_usd < 100.0
    assert order.risk_usd == pytest.approx(order.quantity * order.r_distance)


def test_portfolio_heat_is_the_sum_of_open_risk(config):
    budget = RiskBudget(config)
    book = [position("A/USDT", qty=10, entry=100, stop=98),   # $20
            position("B/USDT", qty=5, entry=100, stop=94)]    # $30
    assert budget.portfolio_heat(book, 10_000) == pytest.approx(0.5)


def test_heat_limit_blocks_the_next_trade(config):
    config["risk"]["max_portfolio_heat_pct"] = 1.5
    budget = RiskBudget(config)
    book = [position(f"S{i}/USDT", qty=10, entry=100, stop=93) for i in range(2)]  # 1.4%
    assert budget.portfolio_heat(book, 10_000) == pytest.approx(1.4)

    order = budget.size_order("NEW/USDT", "long", 100.0, atr=1.0, equity=10_000, risk_pct=1.0)
    decision = budget.check_new_trade(order, book, 10_000, 10_000, 10_000, now=NOW)

    assert not decision
    assert decision.reason == "max_portfolio_heat"
    assert decision.details["heat_after_pct"] > 1.5


def test_correlated_position_cap(config):
    config["risk"]["max_correlated_positions"] = 2
    budget = RiskBudget(config)
    book = [position("A/USDT", qty=0.1), position("B/USDT", qty=0.1)]

    order = budget.size_order("C/USDT", "long", 100.0, atr=1.0, equity=10_000, risk_pct=0.2)
    assert budget.check_new_trade(order, book, 10_000, 10_000, 10_000, now=NOW).reason \
        == "max_correlated_positions"


def test_daily_loss_breaker_uses_equity_not_realized_pnl(config):
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 100.0, atr=1.0, equity=9_700, risk_pct=0.2)

    # Equity down 3% on the day against a 2% limit.
    decision = budget.check_new_trade(order, [], equity=9_700,
                                      day_start_equity=10_000, peak_equity=10_000, now=NOW)
    assert decision.reason == "daily_loss_limit"


def test_drawdown_halt(config):
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 100.0, atr=1.0, equity=8_000, risk_pct=0.2)
    decision = budget.check_new_trade(order, [], equity=8_000,
                                      day_start_equity=8_000, peak_equity=10_000, now=NOW)
    assert decision.reason == "max_drawdown_halt"


def test_cooldown_blocks_entries(config):
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 100.0, atr=1.0, equity=10_000, risk_pct=0.2)
    decision = budget.check_new_trade(
        order, [], 10_000, 10_000, 10_000,
        cooldown_until=NOW + timedelta(hours=1), now=NOW,
    )
    assert decision.reason == "cooldown"


def test_one_position_per_symbol(config):
    budget = RiskBudget(config)
    book = [position("BTC/USDT", qty=0.1)]
    order = budget.size_order("BTC/USDT", "long", 100.0, atr=1.0, equity=10_000, risk_pct=0.2)
    assert budget.check_new_trade(order, book, 10_000, 10_000, 10_000, now=NOW).reason \
        == "symbol_already_open"


def test_risk_shrinks_in_drawdown_and_on_losing_streaks(config):
    budget = RiskBudget(config)
    base = budget.risk_per_trade_pct()

    assert budget.risk_per_trade_pct(drawdown_pct=7.5) < base
    assert budget.risk_per_trade_pct(consecutive_losses=5) < base
    # And never below the configured floor.
    assert budget.risk_per_trade_pct(drawdown_pct=14, consecutive_losses=9) \
        >= config["risk"]["min_risk_per_trade_pct"]


def test_risk_never_exceeds_the_configured_base(config):
    """Kelly is used as a cap, never as licence to size up."""
    budget = RiskBudget(config)
    generous = {"sample": 200, "kelly": 0.9, "win_rate": 0.8,
                "avg_win_r": 3.0, "avg_loss_r": 0.5}
    assert budget.risk_per_trade_pct(rolling=generous) <= config["risk"]["risk_per_trade_pct"]


def test_stop_only_ratchets_forward(config):
    budget = RiskBudget(config)
    pos = position(entry=100.0, stop=96.0, qty=1.0)
    pos.initial_stop = 96.0
    pos.best_price = 110.0

    new_stop, reason = budget.update_stop(pos, price=110.0, atr=2.0)
    assert new_stop > pos.stop_price
    assert reason in ("breakeven", "atr_trail")

    # A pullback must not drag the stop back down.
    pos.stop_price = new_stop
    pulled_back, _ = budget.update_stop(pos, price=101.0, atr=2.0)
    assert pulled_back >= new_stop


def test_breakeven_stop_clears_the_entry(config):
    budget = RiskBudget(config)
    pos = position(entry=100.0, stop=96.0, qty=1.0)
    pos.initial_stop = 96.0
    pos.best_price = 104.5  # just past 1R

    new_stop, reason = budget.update_stop(pos, price=104.5, atr=0.0)
    assert reason == "breakeven"
    assert new_stop > 100.0, "must clear entry so fees don't turn it into a loss"


def test_stablecoins_carry_no_market_beta():
    assert symbol_beta("USDT/USD") == 0.0
    assert symbol_beta("BTC/USDT") == 1.0
    assert symbol_beta("WEIRDCOIN/USDT") > 0, "unknown coins are still crypto"


# ── Cost gate ────────────────────────────────────────────────

def test_a_trade_must_clear_its_own_costs(config):
    """The difference between trading 20 times and 350 times is paying the
    round trip seventeen times more often for the same gross edge."""
    budget = RiskBudget(config)
    price = 100.0

    # A wide-ATR market: the expected move dwarfs the toll.
    wide = budget.size_order("A/USDT", "long", price, atr=1.0, equity=10_000, risk_pct=0.75)
    clears, detail = budget.clears_costs(wide, conviction=0.5, round_trip_bps=17.0)
    assert clears and detail["ratio"] > 3.0

    # A tight-ATR market: the same trade is negative-expectancy on costs.
    tight = budget.size_order("A/USDT", "long", price, atr=0.1, equity=10_000, risk_pct=0.75)
    blocked, detail = budget.clears_costs(tight, conviction=0.5, round_trip_bps=17.0)
    assert not blocked and detail["ratio"] < 3.0


def test_weak_conviction_is_not_credited_with_the_full_target(config):
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 100.0, atr=0.3, equity=10_000, risk_pct=0.75)

    _, strong = budget.clears_costs(order, conviction=0.9, round_trip_bps=17.0)
    _, weak = budget.clears_costs(order, conviction=0.2, round_trip_bps=17.0)
    assert strong["expected_bps"] > weak["expected_bps"]


def test_higher_costs_block_more_trades(config):
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 100.0, atr=0.4, equity=10_000, risk_pct=0.75)

    cheap, _ = budget.clears_costs(order, 0.5, round_trip_bps=10.0)
    dear, _ = budget.clears_costs(order, 0.5, round_trip_bps=120.0)
    assert cheap and not dear


def test_a_degenerate_order_never_clears(config):
    budget = RiskBudget(config)
    order = budget.size_order("A/USDT", "long", 0.0, atr=1.0, equity=10_000, risk_pct=0.75)
    clears, _ = budget.clears_costs(order, 1.0, round_trip_bps=17.0)
    assert not clears
