"""When each market is actually open.

Crypto never closes. Equities trade a session, on weekdays, minus holidays,
and their overnight gap is not a tradeable move — it is a jump that happens
while you cannot act. Treating the two the same way produces two specific
bugs:

  * a stop "triggered" at 03:00 on a Sunday for a stock, filled at a price
    that no market was quoting;
  * an ATR computed across a weekend, which reads the gap as volatility and
    sizes the next position down for a risk that is not intraday risk.

So sessions are explicit. Every instrument carries a calendar, nothing is
planned for a market that is shut, and the session loop sleeps until the
next open rather than spinning.

US market holidays are listed rather than computed. A rule engine for
Good Friday and the Juneteenth observance is more code and more ways to be
subtly wrong than a table someone can check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone

# NYSE/Nasdaq full-day closures. Extend as needed; a missing holiday means
# the bot plans a day the market is shut and finds no data, which is safe
# but wasteful.
US_HOLIDAYS: set[date] = {
    date(2025, 1, 1), date(2025, 1, 9), date(2025, 1, 20), date(2025, 2, 17),
    date(2025, 4, 18), date(2025, 5, 26), date(2025, 6, 19), date(2025, 7, 4),
    date(2025, 9, 1), date(2025, 11, 27), date(2025, 12, 25),
    date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16), date(2026, 4, 3),
    date(2026, 5, 25), date(2026, 6, 19), date(2026, 7, 3), date(2026, 9, 7),
    date(2026, 11, 26), date(2026, 12, 25),
    date(2027, 1, 1), date(2027, 1, 18), date(2027, 2, 15), date(2027, 3, 26),
    date(2027, 5, 31), date(2027, 6, 18), date(2027, 7, 5), date(2027, 9, 6),
    date(2027, 11, 25), date(2027, 12, 24),
}

# Half days close at 18:00 UTC (13:00 ET) instead of 20:00.
US_HALF_DAYS: set[date] = {
    date(2025, 7, 3), date(2025, 11, 28), date(2025, 12, 24),
    date(2026, 11, 27), date(2026, 12, 24),
    date(2027, 11, 26),
}


@dataclass(frozen=True)
class TradingCalendar:
    """When a market accepts orders, in UTC."""

    name: str
    always_open: bool = False
    open_utc: time = time(13, 30)          # 09:30 ET during DST
    close_utc: time = time(20, 0)          # 16:00 ET during DST
    weekdays: tuple[int, ...] = (0, 1, 2, 3, 4)
    holidays: frozenset = field(default_factory=frozenset)
    half_days: frozenset = field(default_factory=frozenset)
    half_day_close_utc: time = time(18, 0)

    # ── Queries ───────────────────────────────────────────────

    def is_trading_day(self, day: date) -> bool:
        if self.always_open:
            return True
        return day.weekday() in self.weekdays and day not in self.holidays

    def session(self, day: date) -> tuple[datetime, datetime] | None:
        """(open, close) in UTC, or None when the market is shut."""
        if not self.is_trading_day(day):
            return None
        if self.always_open:
            start = datetime.combine(day, time(0, 0), tzinfo=timezone.utc)
            return start, start + timedelta(days=1)

        close = self.half_day_close_utc if day in self.half_days else self.close_utc
        return (datetime.combine(day, self.open_utc, tzinfo=timezone.utc),
                datetime.combine(day, close, tzinfo=timezone.utc))

    def is_open(self, moment: datetime) -> bool:
        moment = _as_utc(moment)
        window = self.session(moment.date())
        return bool(window) and window[0] <= moment < window[1]

    def next_open(self, moment: datetime) -> datetime:
        """The next moment this market accepts an order."""
        moment = _as_utc(moment)
        if self.always_open:
            return moment

        for offset in range(0, 12):
            day = (moment + timedelta(days=offset)).date()
            window = self.session(day)
            if window and moment < window[1]:
                return max(moment, window[0])
        # Twelve days of closure means the holiday table is wrong, not that
        # the market is gone.
        return moment + timedelta(days=1)

    def previous_close(self, moment: datetime) -> datetime | None:
        moment = _as_utc(moment)
        if self.always_open:
            return moment
        for offset in range(0, 12):
            day = (moment - timedelta(days=offset)).date()
            window = self.session(day)
            if window and window[1] <= moment:
                return window[1]
        return None

    def minutes_until_close(self, moment: datetime) -> float | None:
        """None when shut. Used to stop opening trades that cannot be
        managed before the bell."""
        moment = _as_utc(moment)
        window = self.session(moment.date())
        if not window or not (window[0] <= moment < window[1]):
            return None
        return (window[1] - moment).total_seconds() / 60.0

    def session_days(self, start: date, end: date) -> list[date]:
        days, cursor = [], start
        while cursor <= end:
            if self.is_trading_day(cursor):
                days.append(cursor)
            cursor += timedelta(days=1)
        return days


CRYPTO = TradingCalendar(name="crypto", always_open=True)

US_EQUITY = TradingCalendar(
    name="us_equity",
    open_utc=time(13, 30),
    close_utc=time(20, 0),
    holidays=frozenset(US_HOLIDAYS),
    half_days=frozenset(US_HALF_DAYS),
)

# US index futures trade nearly around the clock, pausing an hour a day and
# closing over the weekend. Modelled as a long weekday session so the
# maintenance break is not mistaken for a close.
US_FUTURES = TradingCalendar(
    name="us_futures",
    open_utc=time(23, 0),
    close_utc=time(22, 0),
    weekdays=(0, 1, 2, 3, 4, 6),
    holidays=frozenset(US_HOLIDAYS),
)

CALENDARS: dict[str, TradingCalendar] = {
    "crypto": CRYPTO,
    "us_equity": US_EQUITY,
    "us_futures": US_FUTURES,
}


def calendar_for(name: str) -> TradingCalendar:
    return CALENDARS.get(name, CRYPTO)


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)
