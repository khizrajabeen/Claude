"""Trading-day calendar.

Crypto never closes, so a "day" is a convention rather than a market fact.
We anchor it to the UTC midnight boundary that most exchanges use for their
own daily candles and funding schedule, and split it into the phases the
session walks through:

    BRIEFING  read news + data, build the plan      (day open)
    ENTRY     open planned trades                   (inside an entry slot)
    MANAGE    trail stops, honour exits             (until flatten time)
    FLATTEN   close intraday positions              (flatten_at)
    REPORT    write the day's record                (after flatten)

**Entry slots.** A single morning window was right when the universe was
crypto only and the timeframe was fixed. It is wrong for two reasons now.

Crypto runs 24 hours, so a two-hour window at 00:00 UTC discards the other
twenty-two — every 4-hour bar that closes at 04:00, 08:00, 12:00, 16:00 and
20:00 is a setup the bot could not act on. And a US stock cannot be bought
at 00:00 UTC at all: its session is 13:30-20:00, so with one window at the
day open the equity half of the universe would never trade a single share.

So the day carries a list of slots, each naming the moment it opens, how
long it lasts, and which asset classes it admits. Crypto gets one at every
4-hour boundary; equities get one inside their own session, starting after
the opening auction has settled and closing well before the bell so a
position is not opened with minutes left to manage it.

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
class EntrySlot:
    """A window in which some asset classes may open new positions."""

    start: datetime
    end: datetime
    label: str
    asset_classes: frozenset = frozenset()   # empty admits everything

    def contains(self, now: datetime) -> bool:
        return self.start <= _as_utc(now) < self.end

    def admits(self, asset_class) -> bool:
        if not self.asset_classes:
            return True
        name = getattr(asset_class, "value", asset_class)
        return str(name) in self.asset_classes

    @property
    def minutes(self) -> float:
        return (self.end - self.start).total_seconds() / 60.0

    def __str__(self) -> str:
        classes = ",".join(sorted(self.asset_classes)) or "all"
        return (f"{self.label} {self.start.strftime('%H:%M')}-"
                f"{self.end.strftime('%H:%M')} [{classes}]")


@dataclass(frozen=True)
class DaySchedule:
    """Phase boundaries for a single trading day, all tz-aware UTC."""

    day: date
    open_at: datetime
    entry_until: datetime
    flatten_at: datetime
    close_at: datetime
    slots: tuple = ()

    def phase_at(self, now: datetime) -> Phase:
        now = _as_utc(now)
        if now < self.open_at:
            return Phase.CLOSED
        if now < self.entry_until:
            # The briefing is the very first thing the session does; the
            # session itself decides when it is done and moves on to ENTRY.
            return Phase.ENTRY
        if now < self.flatten_at:
            # A later slot is still an ENTRY phase: the book is managed
            # throughout, but new positions are permitted again.
            return Phase.ENTRY if self.slot_at(now) else Phase.MANAGE
        if now < self.close_at:
            return Phase.FLATTEN
        return Phase.REPORT

    # ── Entry slots ───────────────────────────────────────────

    def slot_at(self, now: datetime) -> EntrySlot | None:
        """The slot covering `now`, if any."""
        now = _as_utc(now)
        for slot in self.slots:
            if slot.contains(now):
                return slot
        return None

    def later_slots(self, after: datetime) -> list[EntrySlot]:
        """Slots that have not started yet, in order."""
        after = _as_utc(after)
        return [s for s in self.slots if s.start > after]

    def admits(self, asset_class, now: datetime) -> bool:
        """May this asset class open a position right now?"""
        slot = self.slot_at(now)
        return bool(slot and slot.admits(asset_class))

    def slot_end(self, now: datetime) -> datetime:
        """When the current permission to enter lapses."""
        slot = self.slot_at(now)
        return slot.end if slot else self.entry_until

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
    slots = build_slots(config, open_at, flatten_at, entry_window)
    if slots:
        # The first slot's end is what the legacy `entry_until` means: the
        # moment the day-open window lapses.
        entry_until = slots[0].end

    return DaySchedule(
        day=open_at.date(),
        open_at=open_at,
        entry_until=entry_until,
        flatten_at=flatten_at,
        close_at=close_at,
        slots=tuple(slots),
    )


CRYPTO_CLASSES = frozenset({"crypto_spot", "crypto_perp"})
EQUITY_CLASSES = frozenset({"equity", "etf"})


def build_slots(config: dict, open_at: datetime, flatten_at: datetime,
                entry_window: int) -> list[EntrySlot]:
    """The day's entry slots, in chronological order.

    Crypto slots land on 4-hour boundaries from the day open, because that
    is where its higher-timeframe bar closes and therefore where a trend
    filter actually changes its mind. The equity slot is carved out of the
    real US session, inset at both ends: the opening auction's first half
    hour is where the spread is widest and the overnight gap is still being
    priced, and the last stretch before the bell has no room left to manage
    a position in.
    """
    session = config.get("session", {})
    slots_cfg = session.get("entry_slots", {}) or {}
    if slots_cfg.get("enabled") is False:
        return [EntrySlot(open_at, min(open_at + timedelta(minutes=entry_window),
                                       flatten_at), "day-open")]

    every_hours = float(slots_cfg.get("crypto_every_hours", 4) or 4)
    open_delay = int(slots_cfg.get("equity_open_delay_minutes", 30))
    cutoff = int(slots_cfg.get("equity_entry_cutoff_minutes", 90))
    equity_window = int(slots_cfg.get("equity_window_minutes", 0) or 0)

    slots: list[EntrySlot] = []

    # ── Crypto: one slot per higher-timeframe boundary ──
    # The window is capped to the spacing so two slots never overlap; an
    # overlap would let one instrument be considered twice for the same bar.
    step = timedelta(hours=every_hours)
    window = timedelta(minutes=min(entry_window, every_hours * 60))
    cursor = open_at
    while cursor < flatten_at:
        end = min(cursor + window, flatten_at)
        if end > cursor:
            label = "day-open" if cursor == open_at else f"{cursor.strftime('%H:%M')}-bar"
            slots.append(EntrySlot(cursor, end, label, CRYPTO_CLASSES))
        cursor += step

    # ── Equities: inside their own session ──
    equity_slot = _equity_slot(open_at, flatten_at, open_delay, cutoff, equity_window)
    if equity_slot is not None:
        slots.append(equity_slot)

    slots.sort(key=lambda s: s.start)
    return slots


def _equity_slot(open_at: datetime, flatten_at: datetime, open_delay: int,
                 cutoff: int, window_minutes: int) -> EntrySlot | None:
    """The equity entry window for whichever US session falls in this day."""
    from bot.markets.calendar import calendar_for

    calendar = calendar_for("us_equity")
    # A UTC trading day that opens at midnight contains one US session; one
    # that opens later may straddle two, so check both dates it spans.
    for day in {open_at.date(), flatten_at.date()}:
        session = calendar.session(day)
        if session is None:
            continue
        market_open, market_close = session
        start = market_open + timedelta(minutes=open_delay)
        end = market_close - timedelta(minutes=cutoff)
        if window_minutes > 0:
            end = min(end, start + timedelta(minutes=window_minutes))
        # Clip to the bot's own day, and drop a slot that survives the
        # clipping with no usable width.
        start = max(start, open_at)
        end = min(end, flatten_at)
        if (end - start).total_seconds() >= 15 * 60:
            return EntrySlot(start, end, "us-session", EQUITY_CLASSES)
    return None


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
