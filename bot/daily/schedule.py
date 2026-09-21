"""Trading-day calendar.

Crypto never closes, so a "day" is a convention rather than a market fact.
We anchor it to the UTC midnight boundary that most exchanges use for their
own daily candles and funding schedule, and split it into the phases the
session walks through:

    BRIEFING  read news + data, build the plan      (day open)
    ENTRY     open planned trades                   (day open .. +entry_window)
    MANAGE    trail stops, honour exits             (until flatten time)
    FLATTEN   close intraday positions              (flatten_at)
    REPORT    write the day's record                (after flatten)

Perpetual funding settles at 00:00/08:00/16:00 UTC on most venues, and only
positions open at the settlement instant pay or receive it, so the session
also needs to know where the next settlement sits.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from enum import Enum

# Most venues settle perpetual funding every 8h on these UTC hours.
FUNDING_HOURS_UTC = (0, 8, 16)


class Phase(str, Enum):
    BRIEFING = "briefing"
    ENTRY = "entry"
    MANAGE = "manage"
    FLATTEN = "flatten"
    REPORT = "report"
    CLOSED = "closed"


def parse_hhmm(value: str) -> time:
    """Parse a 'HH:MM' config string into a time."""
    hh, _, mm = str(value).partition(":")
    return time(hour=int(hh), minute=int(mm or 0))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class DaySchedule:
    """Phase boundaries for a single trading day, all tz-aware UTC."""

    day: date
    open_at: datetime
    entry_until: datetime
    flatten_at: datetime
    close_at: datetime

    def phase_at(self, now: datetime) -> Phase:
        now = _as_utc(now)
        if now < self.open_at:
            return Phase.CLOSED
        if now < self.entry_until:
            # The briefing is the very first thing the session does; the
            # session itself decides when it is done and moves on to ENTRY.
            return Phase.ENTRY
        if now < self.flatten_at:
            return Phase.MANAGE
        if now < self.close_at:
            return Phase.FLATTEN
        return Phase.REPORT

    def seconds_until(self, moment: datetime, now: datetime | None = None) -> float:
        now = _as_utc(now or utcnow())
        return max(0.0, (_as_utc(moment) - now).total_seconds())

    def contains(self, now: datetime) -> bool:
        now = _as_utc(now)
        return self.open_at <= now < self.close_at


def _as_utc(dt: datetime) -> datetime:
    """Treat naive datetimes as UTC rather than silently using local time."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_schedule(config: dict, now: datetime | None = None) -> DaySchedule:
    """Build the schedule for the trading day that `now` falls inside."""
    now = _as_utc(now or utcnow())
    session = config.get("session", {})

    open_time = parse_hhmm(session.get("day_open", "00:00"))
    flatten_time = parse_hhmm(session.get("flatten_at", "23:30"))
    entry_window = int(session.get("entry_window_minutes", 90))

    open_at = datetime.combine(now.date(), open_time, tzinfo=timezone.utc)
    if now < open_at:
        # Before today's open we are still inside yesterday's session.
        open_at -= timedelta(days=1)

    flatten_at = datetime.combine(open_at.date(), flatten_time, tzinfo=timezone.utc)
    if flatten_at <= open_at:
        # flatten_at wraps past midnight relative to the open.
        flatten_at += timedelta(days=1)

    close_at = open_at + timedelta(days=1)
    if flatten_at > close_at:
        flatten_at = close_at

    entry_until = min(open_at + timedelta(minutes=entry_window), flatten_at)

    return DaySchedule(
        day=open_at.date(),
        open_at=open_at,
        entry_until=entry_until,
        flatten_at=flatten_at,
        close_at=close_at,
    )


def next_schedule(current: DaySchedule, config: dict) -> DaySchedule:
    """The schedule for the day after `current`."""
    return build_schedule(config, now=current.close_at + timedelta(seconds=1))


def next_funding_time(now: datetime | None = None) -> datetime:
    """Next 8-hourly perpetual funding settlement at or after `now`."""
    now = _as_utc(now or utcnow())
    for hour in FUNDING_HOURS_UTC:
        candidate = now.replace(hour=hour, minute=0, second=0, microsecond=0)
        if candidate > now:
            return candidate
    nxt = now + timedelta(days=1)
    return nxt.replace(hour=FUNDING_HOURS_UTC[0], minute=0, second=0, microsecond=0)


def minutes_to_funding(now: datetime | None = None) -> float:
    now = _as_utc(now or utcnow())
    return (next_funding_time(now) - now).total_seconds() / 60.0


def funding_events_between(start: datetime, end: datetime) -> list[datetime]:
    """Funding settlements strictly inside (start, end] — what a held
    position actually pays over its life."""
    start, end = _as_utc(start), _as_utc(end)
    events: list[datetime] = []
    cursor = start.replace(minute=0, second=0, microsecond=0)
    while cursor <= end:
        if cursor > start and cursor.hour in FUNDING_HOURS_UTC:
            events.append(cursor)
        cursor += timedelta(hours=1)
    return events
