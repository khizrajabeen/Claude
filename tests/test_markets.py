"""Calendars, instruments, timeframe choice and entry slots.

These are the pieces that let one book hold a Bitcoin perp and a Nasdaq
stock at once. Most of what can go wrong here is a silent wrong answer —
a market reported open on Thanksgiving, a stock admitted to a midnight
entry slot — so the tests are about the boundaries, not the happy path.
"""

from datetime import date, datetime, timedelta, timezone

import pytest

from bot.daily.schedule import (
    CRYPTO_CLASSES,
    EQUITY_CLASSES,
    EntrySlot,
    build_schedule,
)
from bot.markets.calendar import CRYPTO, US_EQUITY, calendar_for
from bot.markets.instrument import (
    DEFAULT_COSTS,
    AssetClass,
    build_instrument,
    build_universe,
)
from bot.markets.timeframes import (
    CRYPTO_LADDER,
    choose_timeframe,
    fits_in_session,
    ladder_for,
)


def utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


# ── Calendars ────────────────────────────────────────────────

def test_crypto_never_closes():
    assert CRYPTO.is_open(utc(2026, 1, 1, 3, 17))          # New Year, 3am
    assert CRYPTO.is_trading_day(date(2026, 12, 25))       # Christmas
    assert CRYPTO.minutes_until_close(utc(2026, 1, 1)) > 0


def test_us_equity_session_bounds():
    monday = date(2026, 9, 21)
    assert US_EQUITY.is_trading_day(monday)
    open_at, close_at = US_EQUITY.session(monday)
    assert (open_at.hour, open_at.minute) == (13, 30)
    assert (close_at.hour, close_at.minute) == (20, 0)

    assert not US_EQUITY.is_open(utc(2026, 9, 21, 13, 29))
    assert US_EQUITY.is_open(utc(2026, 9, 21, 13, 30))
    assert US_EQUITY.is_open(utc(2026, 9, 21, 19, 59))
    assert not US_EQUITY.is_open(utc(2026, 9, 21, 20, 0))


def test_us_equity_is_shut_at_weekends():
    saturday = date(2026, 9, 26)
    assert saturday.weekday() == 5
    assert not US_EQUITY.is_trading_day(saturday)
    assert US_EQUITY.session(saturday) is None
    assert not US_EQUITY.is_open(utc(2026, 9, 26, 15, 0))


def test_a_holiday_is_not_a_trading_day():
    # Independence Day 2026 falls on a Saturday; the observed holiday is
    # Friday the 3rd. Either way the market is shut on both.
    assert not US_EQUITY.is_trading_day(date(2026, 7, 3))
    assert not US_EQUITY.is_trading_day(date(2026, 12, 25))
    assert not US_EQUITY.is_open(utc(2026, 12, 25, 15, 0))


def test_next_open_skips_the_weekend():
    friday_evening = utc(2026, 9, 25, 21, 0)
    nxt = US_EQUITY.next_open(friday_evening)
    assert nxt.date() == date(2026, 9, 28)      # Monday
    assert (nxt.hour, nxt.minute) == (13, 30)


def test_minutes_until_close_is_none_when_shut():
    assert US_EQUITY.minutes_until_close(utc(2026, 9, 26, 15, 0)) is None
    assert US_EQUITY.minutes_until_close(utc(2026, 9, 21, 19, 0)) == pytest.approx(60)


def test_calendar_for_falls_back_rather_than_raising():
    assert calendar_for("crypto") is CRYPTO
    assert calendar_for("us_equity") is US_EQUITY
    assert calendar_for("nonsense") is not None


# ── Instruments ──────────────────────────────────────────────

def test_asset_class_capabilities():
    assert AssetClass.CRYPTO_PERP.pays_funding
    assert not AssetClass.CRYPTO_SPOT.pays_funding
    assert AssetClass.CRYPTO_PERP.can_short
    assert AssetClass.CRYPTO_SPOT.is_crypto
    assert not AssetClass.EQUITY.is_crypto
    assert not AssetClass.EQUITY.pays_funding


def test_costs_differ_by_an_order_of_magnitude_across_classes():
    """A stock's round trip is far cheaper than a crypto taker's.

    Charging one figure to both is what made the cost gate veto equity
    trades that comfortably clear their real costs.
    """
    perp = build_instrument({"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"})
    stock = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
    etf = build_instrument({"symbol": "SPY", "asset_class": "etf"})

    assert perp.round_trip_bps > stock.round_trip_bps > etf.round_trip_bps
    assert perp.round_trip_bps > 2 * stock.round_trip_bps


def test_a_bare_symbol_is_read_as_crypto_spot():
    assert build_instrument("ETH/USDT").asset_class is AssetClass.CRYPTO_SPOT


def test_instrument_is_open_follows_its_own_calendar():
    perp = build_instrument({"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"})
    stock = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
    midnight = utc(2026, 9, 21, 0, 30)

    assert perp.is_open(midnight)
    assert not stock.is_open(midnight)
    assert stock.is_open(utc(2026, 9, 21, 15, 0))


def test_universe_spans_the_configured_classes(config):
    config["data"]["instruments"] = [
        {"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"},
        {"symbol": "NVDA", "asset_class": "equity"},
        {"symbol": "SPY", "asset_class": "etf"},
    ]
    universe = build_universe(config)
    classes = {i.asset_class for i in universe}
    assert AssetClass.CRYPTO_PERP in classes
    assert AssetClass.EQUITY in classes
    assert AssetClass.ETF in classes


def test_universe_deduplicates(config):
    config["data"]["instruments"] = [
        {"symbol": "BTC/USDT", "asset_class": "crypto_spot"},
        {"symbol": "BTC/USDT", "asset_class": "crypto_spot"},
    ]
    config["data"]["symbols"] = ["BTC/USDT"]
    assert len(build_universe(config)) == 1


def test_spot_and_perp_of_one_coin_are_distinct_instruments(config):
    config["data"]["instruments"] = [
        {"symbol": "BTC/USDT", "asset_class": "crypto_spot"},
        {"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"},
    ]
    config["data"]["symbols"] = []
    universe = build_universe(config)
    assert len(universe) == 2
    assert len({i.key for i in universe}) == 2


# ── Timeframe choice ─────────────────────────────────────────

def test_a_quiet_crypto_market_steps_to_a_slower_bar():
    perp = build_instrument({"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp",
                             "timeframe": "1h"})
    tf, higher, reason = choose_timeframe(perp, atr_pct=0.10)
    assert CRYPTO_LADDER.index(tf) > CRYPTO_LADDER.index("1h")
    assert "quiet" in reason


def test_a_wild_crypto_market_steps_to_a_faster_bar():
    perp = build_instrument({"symbol": "SOL/USDT:USDT", "asset_class": "crypto_perp",
                             "timeframe": "1h"})
    tf, higher, reason = choose_timeframe(perp, atr_pct=4.0)
    assert CRYPTO_LADDER.index(tf) < CRYPTO_LADDER.index("1h")
    assert "volatile" in reason


def test_an_ordinary_crypto_market_stays_put():
    perp = build_instrument({"symbol": "ETH/USDT:USDT", "asset_class": "crypto_perp",
                             "timeframe": "1h"})
    tf, _, _ = choose_timeframe(perp, atr_pct=0.9)
    assert tf == "1h"


def test_equities_stay_daily_however_volatile():
    stock = build_instrument({"symbol": "TSLA", "asset_class": "equity"})
    for atr in (0.05, 1.0, 9.0):
        tf, higher, _ = choose_timeframe(stock, atr_pct=atr)
        assert tf == "1d"


def test_the_ladder_never_walks_off_its_own_ends():
    fast = build_instrument({"symbol": "X/USDT", "asset_class": "crypto_spot",
                             "timeframe": CRYPTO_LADDER[0]})
    slow = build_instrument({"symbol": "Y/USDT", "asset_class": "crypto_spot",
                             "timeframe": CRYPTO_LADDER[-1]})
    assert choose_timeframe(fast, atr_pct=99.0)[0] == CRYPTO_LADDER[0]
    assert choose_timeframe(slow, atr_pct=0.001)[0] == CRYPTO_LADDER[-1]


def test_a_shut_market_fits_nothing():
    stock = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
    assert not fits_in_session(stock, "1d", utc(2026, 9, 26, 15, 0))   # Saturday
    assert fits_in_session(stock, "1d", utc(2026, 9, 21, 15, 0))


def test_crypto_always_fits():
    perp = build_instrument({"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"})
    assert fits_in_session(perp, "4h", utc(2026, 9, 26, 3, 0))


def test_the_replay_ladder_covers_every_rung_the_chooser_can_reach():
    """Downloading less than the chooser can ask for looks, in a report,
    exactly like having had no opportunity."""
    perp = build_instrument({"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"})
    ladder = ladder_for(perp)
    for rung in CRYPTO_LADDER:
        assert rung in ladder
        higher = choose_timeframe(perp, atr_pct=None)[1]
        assert higher in ladder or higher == rung


# ── Entry slots ──────────────────────────────────────────────

def test_crypto_gets_a_slot_at_every_four_hour_boundary(config):
    schedule = build_schedule(config, utc(2026, 9, 21, 1, 0))
    crypto = [s for s in schedule.slots if s.asset_classes == CRYPTO_CLASSES]
    assert len(crypto) >= 6
    assert {s.start.hour for s in crypto} == {0, 4, 8, 12, 16, 20}


def test_the_equity_slot_sits_inside_the_us_session(config):
    schedule = build_schedule(config, utc(2026, 9, 21, 1, 0))
    equity = [s for s in schedule.slots if s.asset_classes == EQUITY_CLASSES]
    assert len(equity) == 1
    slot = equity[0]
    open_at, close_at = US_EQUITY.session(date(2026, 9, 21))
    assert slot.start >= open_at, "must not trade the opening auction"
    assert slot.end <= close_at, "must not open a position at the bell"
    assert slot.end < close_at, "must leave room to manage the position"


def test_no_equity_slot_at_the_weekend(config):
    schedule = build_schedule(config, utc(2026, 9, 26, 1, 0))   # Saturday
    assert not [s for s in schedule.slots if s.asset_classes == EQUITY_CLASSES]
    assert [s for s in schedule.slots if s.asset_classes == CRYPTO_CLASSES]


def test_a_stock_is_not_admitted_to_a_midnight_slot(config):
    """The whole point of slots: a US stock cannot be bought at 00:00 UTC."""
    schedule = build_schedule(config, utc(2026, 9, 21, 1, 0))
    midnight = utc(2026, 9, 21, 0, 30)
    assert schedule.admits("crypto_perp", midnight)
    assert not schedule.admits("equity", midnight)


def test_the_equity_slot_admits_only_equities(config):
    """The equity slot overlaps a crypto slot in wall-clock time, which is
    fine — but neither may admit the other's asset classes."""
    schedule = build_schedule(config, utc(2026, 9, 21, 1, 0))
    equity = [s for s in schedule.slots if s.asset_classes == EQUITY_CLASSES][0]
    assert equity.admits("equity") and equity.admits("etf")
    assert not equity.admits("crypto_perp")
    assert not equity.admits("crypto_spot")

    for slot in schedule.slots:
        if slot.asset_classes == CRYPTO_CLASSES:
            assert not slot.admits("equity")
            assert not slot.admits("etf")


def test_crypto_slots_never_overlap_each_other(config):
    schedule = build_schedule(config, utc(2026, 9, 21, 1, 0))
    crypto = sorted((s for s in schedule.slots if s.asset_classes == CRYPTO_CLASSES),
                    key=lambda s: s.start)
    for earlier, later in zip(crypto, crypto[1:]):
        assert earlier.end <= later.start, (
            "overlapping slots would consider the same bar twice"
        )


def test_slots_can_be_switched_off(config):
    config["session"]["entry_slots"] = {"enabled": False}
    schedule = build_schedule(config, utc(2026, 9, 21, 1, 0))
    assert len(schedule.slots) == 1
    assert schedule.slots[0].label == "day-open"


def test_later_slots_are_only_the_ones_still_ahead(config):
    schedule = build_schedule(config, utc(2026, 9, 21, 1, 0))
    at_nine = utc(2026, 9, 21, 9, 0)
    later = schedule.later_slots(at_nine)
    assert later, "there must be slots left at 09:00"
    assert all(s.start > at_nine for s in later)


def test_a_slot_with_no_classes_admits_everything():
    slot = EntrySlot(utc(2026, 9, 21), utc(2026, 9, 21, 2), "any")
    assert slot.admits("equity") and slot.admits("crypto_perp")
