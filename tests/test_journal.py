"""Records must survive the day boundary — that is the whole point."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.daily.journal import BotState, DaySummary, Journal
from bot.trading.models import Position, Trade

T0 = datetime(2026, 4, 1, 9, 0, tzinfo=timezone.utc)


def make_trade(idx=0, pnl=50.0, r=2.0, symbol="BTC/USDT", day="2026-04-01"):
    return Trade(
        id=f"t{idx}", symbol=symbol, side="long", entry_price=100.0,
        exit_price=100 + pnl / 10, quantity=10.0, leverage=1.0,
        opened_at=T0, closed_at=T0 + timedelta(hours=2), pnl=pnl,
        pnl_pct=pnl / 10, r_multiple=r, fees=1.0, funding=0.05,
        slippage_cost=0.2, exit_reason="take_profit" if pnl > 0 else "stop_loss",
        risk_usd=25.0, opened_on_day=day, closed_on_day=day,
    )


def test_fresh_state_seeds_from_config(config):
    state = Journal(config).load_state()
    assert state.cash == config["paper"]["initial_balance"]
    assert state.peak_equity == state.cash
    assert state.positions == []


def test_state_round_trips_including_open_positions(config):
    journal = Journal(config)
    state = journal.load_state()
    state.cash = 8_432.10
    state.consecutive_losses = 2
    state.last_completed_day = "2026-04-01"
    state.positions.append(Position(
        id="p1", symbol="ETH/USDT", side="short", entry_price=2_500.0,
        quantity=1.5, leverage=1.0, opened_at=T0, stop_price=2_560.0,
        risk_usd=90.0, margin=3_750.0, opened_on_day="2026-04-01",
    ))
    journal.save_state(state)

    # A brand new process reading the same directory.
    resumed = Journal(config).load_state()
    assert resumed.cash == pytest.approx(8_432.10)
    assert resumed.consecutive_losses == 2
    assert resumed.last_completed_day == "2026-04-01"
    assert len(resumed.positions) == 1

    carried = resumed.positions[0]
    assert carried.symbol == "ETH/USDT"
    assert carried.side == "short"
    assert carried.stop_price == pytest.approx(2_560.0)
    assert carried.opened_at == T0


def test_corrupt_state_falls_back_rather_than_crashing(config):
    journal = Journal(config)
    journal.state_path.write_text("{ not json")

    state = journal.load_state()
    assert state.cash == config["paper"]["initial_balance"]


def test_trades_append_and_reload(config):
    journal = Journal(config)
    for i in range(5):
        journal.append_trade(make_trade(i, pnl=10.0 * i))

    reloaded = Journal(config).load_trades()
    assert len(reloaded) == 5
    assert reloaded[0].symbol == "BTC/USDT"
    assert reloaded[-1].pnl == pytest.approx(40.0)
    assert isinstance(reloaded[0].opened_at, datetime)


def test_rolling_stats_feed_tomorrows_sizing(config):
    journal = Journal(config)
    for i in range(6):
        journal.append_trade(make_trade(i, pnl=100.0, r=2.0))
    for i in range(6, 10):
        journal.append_trade(make_trade(i, pnl=-50.0, r=-1.0))

    stats = journal.rolling_stats(window=50)
    assert stats["sample"] == 10
    assert stats["win_rate"] == pytest.approx(0.6)
    assert stats["avg_win_r"] == pytest.approx(2.0)
    assert stats["avg_loss_r"] == pytest.approx(1.0)
    assert stats["expectancy_r"] == pytest.approx(0.6 * 2.0 - 0.4 * 1.0)
    assert stats["profit_factor"] == pytest.approx(3.0)
    assert 0 < stats["kelly"] <= 1


def test_rolling_stats_window_is_respected(config):
    journal = Journal(config)
    for i in range(30):
        journal.append_trade(make_trade(i, pnl=-10.0, r=-1.0))
    for i in range(30, 40):
        journal.append_trade(make_trade(i, pnl=10.0, r=1.0))

    assert journal.rolling_stats(window=10)["win_rate"] == pytest.approx(1.0)
    assert journal.rolling_stats(window=40)["win_rate"] == pytest.approx(0.25)


def test_day_summaries_accumulate(config):
    journal = Journal(config)
    journal.append_day(DaySummary(day="2026-04-01", starting_equity=10_000,
                                  ending_equity=10_120, realized_pnl=120,
                                  return_pct=1.2, wins=2, losses=1, trades_closed=3))
    journal.append_day(DaySummary(day="2026-04-02", starting_equity=10_120,
                                  ending_equity=10_050, realized_pnl=-70,
                                  return_pct=-0.69, wins=1, losses=2, trades_closed=3))

    days = Journal(config).load_days()
    assert [d["day"] for d in days] == ["2026-04-01", "2026-04-02"]
    assert days[-1]["ending_equity"] == pytest.approx(10_050)
    assert Journal(config).last_day()["day"] == "2026-04-02"


def test_per_symbol_stats_identify_losers(config):
    journal = Journal(config)
    for i in range(4):
        journal.append_trade(make_trade(i, pnl=80.0, r=2.0, symbol="BTC/USDT"))
    for i in range(4, 8):
        journal.append_trade(make_trade(i, pnl=-40.0, r=-1.0, symbol="DOGE/USDT"))

    stats = journal.symbol_stats()
    assert stats["BTC/USDT"]["avg_r"] == pytest.approx(2.0)
    assert stats["DOGE/USDT"]["avg_r"] == pytest.approx(-1.0)


def test_briefing_archive_round_trips(config):
    journal = Journal(config)
    payload = {"day": "2026-04-01", "symbols": {"BTC/USDT": {"price": 50_000}}}
    journal.save_briefing("2026-04-01", payload)

    assert Journal(config).load_briefing("2026-04-01") == payload
    assert Journal(config).load_briefing("2026-04-02") is None


def test_state_write_is_atomic(config):
    """A partial write must never leave an unreadable state file."""
    journal = Journal(config)
    state = journal.load_state()
    journal.save_state(state)

    leftovers = list(journal.dir.glob("*.tmp"))
    assert not leftovers, f"temp files left behind: {leftovers}"
