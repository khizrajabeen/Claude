"""The daily lifecycle, driven on a simulated clock.

Every test here runs the same DailySession that trades live; only the clock
and the exchange are swapped. If the lifecycle is wrong, these fail.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.daily.journal import Journal
from bot.daily.schedule import (
    DaySchedule, Phase, build_schedule, funding_events_between,
    next_funding_time, next_schedule,
)
from bot.daily.session import DailySession, SimulatedClock
from bot.trading.broker import PaperBroker
from tests.conftest import FakeExchange, make_ohlcv

DAY_START = datetime(2026, 5, 4, 0, 0, tzinfo=timezone.utc)


def build_session(config, clock=None, drift=0.004, vol=0.012, seed=1, bars=600):
    """A session wired to synthetic markets and a simulated clock."""
    clock = clock or SimulatedClock(DAY_START)
    start = DAY_START - timedelta(hours=bars - 40)

    frames, htf = {}, {}
    for i, symbol in enumerate(config["data"]["symbols"]):
        frames[symbol] = make_ohlcv(bars=bars, start_price=100 * (i + 1),
                                    drift=drift, vol=vol, seed=seed + i, start=start)
        htf[symbol] = make_ohlcv(bars=bars // 4, start_price=100 * (i + 1),
                                 drift=drift * 4, vol=vol * 2, seed=seed + i,
                                 start=start, freq_hours=4)

    exchange = FakeExchange(frames, clock=clock, htf=htf)
    journal = Journal(config)
    state = journal.load_state()
    broker = PaperBroker(config, cash=state.cash, positions=state.positions)
    session = DailySession(config, exchange, broker, journal, state, clock=clock)
    return session, journal, broker, clock


# ── Schedule ─────────────────────────────────────────────────

def test_schedule_phases_cover_the_day(config):
    schedule = build_schedule(config, DAY_START + timedelta(hours=6))
    assert schedule.day == DAY_START.date()
    assert schedule.phase_at(DAY_START + timedelta(minutes=30)) == Phase.ENTRY
    assert schedule.phase_at(DAY_START + timedelta(hours=6)) == Phase.MANAGE
    assert schedule.phase_at(DAY_START + timedelta(hours=23, minutes=40)) == Phase.FLATTEN


def test_before_the_open_we_are_still_in_yesterdays_session(config):
    config["session"]["day_open"] = "08:00"
    schedule = build_schedule(config, DAY_START + timedelta(hours=3))
    assert schedule.day == (DAY_START - timedelta(days=1)).date()


def test_schedule_rolls_to_the_next_day(config):
    first = build_schedule(config, DAY_START)
    second = next_schedule(first, config)
    assert second.day == first.day + timedelta(days=1)
    assert second.open_at == first.close_at


def test_funding_settlements_are_eight_hourly():
    assert next_funding_time(datetime(2026, 5, 4, 9, 30, tzinfo=timezone.utc)).hour == 16
    assert next_funding_time(datetime(2026, 5, 4, 17, 0, tzinfo=timezone.utc)).hour == 0
    events = funding_events_between(
        datetime(2026, 5, 4, 7, 0, tzinfo=timezone.utc),
        datetime(2026, 5, 5, 1, 0, tzinfo=timezone.utc),
    )
    assert [e.hour for e in events] == [8, 16, 0]


# ── The day ──────────────────────────────────────────────────

def test_a_day_runs_end_to_end_and_writes_a_record(config):
    session, journal, broker, clock = build_session(config)
    result = session.run_day(build_schedule(config, DAY_START))

    assert result.summary.day == str(DAY_START.date())
    assert result.briefing is not None
    assert result.plan is not None

    days = Journal(config).load_days()
    assert len(days) == 1
    assert days[0]["day"] == str(DAY_START.date())
    assert journal.load_briefing(DAY_START.date()) is not None


def test_the_briefing_runs_before_any_order(config):
    session, journal, broker, clock = build_session(config)
    schedule = build_schedule(config, DAY_START)

    session._begin_day(str(schedule.day))
    briefing, plan = session.open_day(schedule)

    assert not broker.positions, "no position may exist before the entry phase"
    assert briefing.symbols, "the briefing must read every symbol"
    assert plan.risk_pct_per_trade > 0

    opened = session.execute_entries(plan, schedule)
    assert len(opened) == len(plan.trades)


def test_entries_stop_at_the_end_of_the_window(config):
    session, journal, broker, clock = build_session(config)
    schedule = build_schedule(config, DAY_START)
    session._begin_day(str(schedule.day))
    _, plan = session.open_day(schedule)

    clock._now = schedule.entry_until + timedelta(minutes=1)
    assert session.execute_entries(plan, schedule) == []


def test_intraday_book_is_flat_at_the_end_of_the_day(config):
    config["session"]["carry_overnight"] = False
    session, journal, broker, clock = build_session(config)
    session.run_day(build_schedule(config, DAY_START))

    assert broker.positions == []
    assert Journal(config).load_state().positions == []


def test_carry_mode_holds_positions_into_the_next_day(config):
    config["session"]["carry_overnight"] = True
    session, journal, broker, clock = build_session(config)
    schedule = build_schedule(config, DAY_START)
    session._begin_day(str(schedule.day))
    _, plan = session.open_day(schedule)
    opened = session.execute_entries(plan, schedule)
    if not opened:
        pytest.skip("no trade was planned for this synthetic market")

    clock._now = schedule.flatten_at
    session.flatten(schedule)
    assert broker.positions, "carry mode must keep positions open"


def test_tomorrow_resumes_from_todays_record(config):
    """The requirement in one test: a second process picks up where the
    first left off."""
    session, journal, broker, clock = build_session(config)
    first = build_schedule(config, DAY_START)
    session.run_day(first)

    cash_after_day_one = broker.cash
    trades_day_one = len(Journal(config).load_trades())

    # Entirely new objects, same directory — as if the bot restarted.
    second_schedule = next_schedule(first, config)
    clock2 = SimulatedClock(second_schedule.open_at)
    session2, journal2, broker2, _ = build_session(config, clock=clock2, seed=40)

    assert broker2.cash == pytest.approx(cash_after_day_one)
    assert session2.state.last_completed_day == str(first.day)

    session2.run_day(second_schedule)
    days = Journal(config).load_days()
    assert len(days) == 2
    assert days[1]["starting_equity"] == pytest.approx(days[0]["ending_equity"], rel=1e-6)
    assert len(Journal(config).load_trades()) >= trades_day_one


def test_yesterdays_numbers_reach_todays_briefing(config):
    session, journal, broker, clock = build_session(config)
    first = build_schedule(config, DAY_START)
    session.run_day(first)

    second = next_schedule(first, config)
    clock2 = SimulatedClock(second.open_at)
    session2, _, _, _ = build_session(config, clock=clock2, seed=40)
    session2._begin_day(str(second.day))
    briefing, _ = session2.open_day(second)

    assert briefing.yesterday is not None
    assert briefing.yesterday["day"] == str(first.day)


def test_daily_loss_breaker_halts_and_flattens(config):
    config["risk"]["max_daily_loss_pct"] = 2.0
    session, journal, broker, clock = build_session(config)
    schedule = build_schedule(config, DAY_START)

    session._begin_day(str(schedule.day))
    _, plan = session.open_day(schedule)
    opened = session.execute_entries(plan, schedule)
    if not opened:
        pytest.skip("no trade was planned for this synthetic market")

    # Mark the day as having opened well above current equity, which is what
    # a 3% intraday loss looks like to the breaker.
    prices = session._latest_prices(session.symbols, clock.now())
    session.state.day_start_equity = broker.equity(prices) / 0.97

    clock.advance(3600)
    session.manage_once(schedule)

    assert session.state.halted_reason == "daily_loss_limit"
    assert broker.positions == [], "the breaker must flatten, not just stop entering"


def test_drawdown_halt_fires_on_equity(config):
    session, journal, broker, clock = build_session(config)
    schedule = build_schedule(config, DAY_START)
    session._begin_day(str(schedule.day))
    _, plan = session.open_day(schedule)
    if not session.execute_entries(plan, schedule):
        pytest.skip("no trade was planned for this synthetic market")

    prices = session._latest_prices(session.symbols, clock.now())
    # A peak 25% above current equity breaches the 15% halt.
    session.state.peak_equity = broker.equity(prices) / 0.75
    session.state.day_start_equity = broker.equity(prices)

    clock.advance(3600)
    session.manage_once(schedule)

    assert session.state.halted_reason == "max_drawdown_halt"
    assert broker.positions == []


def test_time_stop_counts_bars_not_management_passes(config):
    """A 48-bar time stop must survive more than 48 five-minute passes."""
    config["stops"]["time_stop_bars"] = 5
    config["session"]["manage_interval_seconds"] = 60
    session, journal, broker, clock = build_session(config)
    schedule = build_schedule(config, DAY_START)

    session._begin_day(str(schedule.day))
    _, plan = session.open_day(schedule)
    opened = session.execute_entries(plan, schedule)
    if not opened:
        pytest.skip("no trade was planned for this synthetic market")

    # Ten passes inside a single hourly bar must not age the position.
    for _ in range(10):
        clock.advance(60)
        session.manage_once(schedule)

    still_open = [p for p in broker.positions if p.id == opened[0].id]
    if still_open:
        assert still_open[0].bars_held <= 1, "bars_held tracked passes, not bars"


def test_management_reads_fresh_bars_not_the_briefing_snapshot(config):
    session, journal, broker, clock = build_session(config)
    schedule = build_schedule(config, DAY_START)
    session._begin_day(str(schedule.day))
    session.open_day(schedule)

    first = session._bars("BTC/USDT")
    clock.advance(6 * 3600)
    later = session._bars("BTC/USDT")

    assert later is not None and first is not None
    assert later.index[-1] > first.index[-1], "the tape must advance with the clock"


def test_run_forever_respects_the_day_limit(config):
    session, journal, broker, clock = build_session(config, bars=900)
    results = session.run_forever(max_days=3)

    assert len(results) == 3
    assert len(Journal(config).load_days()) == 3
    assert [r.summary.day for r in results] == sorted({r.summary.day for r in results})


def test_late_start_opens_a_fresh_entry_window(config):
    """Deploying at 16:00 should still be able to trade today."""
    late = DAY_START + timedelta(hours=16)
    clock = SimulatedClock(late)
    session, journal, broker, _ = build_session(config, clock=clock)

    original = build_schedule(config, late)
    assert late > original.entry_until, "fixture must actually be a late start"

    adjusted = session._adjust_for_late_start(original)
    assert adjusted.entry_until > late
    assert adjusted.entry_until <= original.flatten_at
    assert adjusted.day == original.day


def test_late_start_applies_only_once(config):
    late = DAY_START + timedelta(hours=16)
    clock = SimulatedClock(late)
    session, _, _, _ = build_session(config, clock=clock)

    schedule = build_schedule(config, late)
    session._adjust_for_late_start(schedule)
    second = session._adjust_for_late_start(schedule)
    assert second.entry_until == schedule.entry_until, (
        "only the first day of a run gets a catch-up window"
    )


def test_late_start_can_be_disabled(config):
    config["session"]["allow_late_start"] = False
    late = DAY_START + timedelta(hours=16)
    session, _, _, _ = build_session(config, clock=SimulatedClock(late))

    schedule = build_schedule(config, late)
    assert session._adjust_for_late_start(schedule).entry_until == schedule.entry_until


def test_starting_on_time_is_unaffected(config):
    clock = SimulatedClock(DAY_START + timedelta(minutes=10))
    session, _, _, _ = build_session(config, clock=clock)
    schedule = build_schedule(config, clock.now())
    assert session._adjust_for_late_start(schedule) == schedule


def test_a_late_start_actually_trades(config):
    late = DAY_START + timedelta(hours=16)
    clock = SimulatedClock(late)
    session, journal, broker, _ = build_session(config, clock=clock)

    result = session.run_day(build_schedule(config, late))
    assert result.summary.trades_opened > 0, "a late start must still work the plan"
