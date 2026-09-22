"""The trading day, as a state machine.

    open  →  BRIEFING  →  ENTRY  →  MANAGE  →  FLATTEN  →  REPORT  →  next day

Every day starts by reading yesterday's record off disk and the morning's
news off the wire, builds a plan, works that plan inside a bounded entry
window, manages the open book for the rest of the session, closes what it
said it would close, and writes the day down so tomorrow starts informed.

The clock is injected. Live trading passes the real one; tests and the
backtester pass a simulated one, so the code that runs against real money
is exactly the code that gets tested — there is no separate backtest engine
to drift out of sync with it.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from bot.daily.briefing import Briefing, BriefingBuilder
from bot.daily.journal import BotState, DaySummary, Journal
from bot.daily.plan import DayPlan, DayPlanner
from bot.daily.schedule import (
    DaySchedule,
    EntrySlot,
    Phase,
    build_schedule,
    next_schedule,
)
from bot.portfolio.allocator import StrategyAllocator
from bot.portfolio.autopilot import AutoPilot
from bot.portfolio.tracker import StrategyTracker
from bot.portfolio.voltarget import VolatilityTargeter
from bot.risk.budget import RiskBudget, SizedOrder
from bot.strategies import build_strategies
from bot.strategies.base import MarketContext
from bot.trading.broker import InsufficientFunds, PaperBroker
from bot.trading.models import Position, Trade

logger = logging.getLogger("trading_bot")


class Clock:
    """Wall-clock time and real sleeping."""

    def now(self) -> datetime:
        return datetime.now(timezone.utc)

    def sleep(self, seconds: float) -> None:
        import time as _time
        _time.sleep(max(0.0, seconds))


class SimulatedClock:
    """A clock that jumps forward instead of waiting — used by tests and
    the day-replay backtester."""

    def __init__(self, start: datetime):
        self._now = start if start.tzinfo else start.replace(tzinfo=timezone.utc)
        self.slept = 0.0

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        seconds = max(0.0, seconds)
        self._now += timedelta(seconds=seconds)
        self.slept += seconds

    def advance(self, seconds: float) -> None:
        self._now += timedelta(seconds=seconds)


@dataclass
class DayResult:
    """What one completed day produced."""

    summary: DaySummary
    plan: DayPlan | None = None
    briefing: Briefing | None = None
    opened: list = field(default_factory=list)
    closed: list = field(default_factory=list)


class DailySession:
    """Runs trading days end to end."""

    def __init__(
        self,
        config: dict,
        router,
        broker: PaperBroker,
        journal: Journal,
        state: BotState,
        clock: Clock | SimulatedClock | None = None,
        price_source: Callable[[str, datetime], dict] | None = None,
        universe: list | None = None,
    ):
        from bot.markets import build_universe

        self.config = config
        self.router = router
        self.universe = universe if universe is not None else build_universe(config)
        self.by_symbol = {i.symbol: i for i in self.universe}
        self.broker = broker
        self.journal = journal
        self.state = state
        self.clock = clock or Clock()

        self.risk = RiskBudget(config)
        self.briefing_builder = BriefingBuilder(config, router, universe=self.universe)
        self.strategies = build_strategies(config)
        self.allocator = StrategyAllocator(config)
        self.autopilot = AutoPilot(config)
        self.tracker = StrategyTracker(config, journal)
        self.vol_targeter = VolatilityTargeter(config)
        self.planner = DayPlanner(config, self.risk, self.strategies, self.allocator)

        session = config.get("session", {})
        self.manage_interval = int(session.get("manage_interval_seconds", 300))
        self.carry_overnight = bool(session.get("carry_overnight", False))
        self.max_holding_days = int(session.get("max_holding_days", 3))
        self.entry_stagger_seconds = int(session.get("entry_stagger_seconds", 0))
        self.allow_late_start = bool(session.get("allow_late_start", True))

        scaling = config.get("scaling", {})
        self.scale_out_levels = [
            (float(rung.get("at_r", 0)), float(rung.get("fraction", 0)))
            for rung in scaling.get("take_profit_at", []) or []
            if float(rung.get("at_r", 0)) > 0 and 0 < float(rung.get("fraction", 0)) < 1
        ]
        self.scale_out_levels.sort()
        self.scale_out_classes = set(scaling.get("scale_out_classes", []) or [])
        self.max_units = int(scaling.get("max_units", 1))
        self.pyramid_step_r = float(scaling.get("pyramid_step_r", 1.0))
        self.pyramid_size_factor = float(scaling.get("pyramid_size_factor", 0.5))
        self.pyramid_classes = set(scaling.get("pyramid_classes", []) or [])
        self._first_day = True

        self.symbols = [i.symbol for i in self.universe]
        self.timeframe = config.get("data", {}).get("timeframe", "1h")

        # Injectable so the backtester can feed historical bars through the
        # same code path the live session uses.
        self._price_source = price_source

        self.running = False
        self._market_limits: dict = {}
        self._funding_rates: dict = {}

    # ── Public entry points ───────────────────────────────────

    def run_forever(self, max_days: int | None = None) -> list[DayResult]:
        """Run day after day until stopped."""
        self.running = True
        results: list[DayResult] = []
        schedule = build_schedule(self.config, self.clock.now())

        while self.running:
            result = self.run_day(schedule)
            results.append(result)
            if max_days is not None and len(results) >= max_days:
                break
            schedule = next_schedule(schedule, self.config)
            wait = schedule.seconds_until(schedule.open_at, self.clock.now())
            if wait > 0:
                logger.info("Next session opens at %s (%.1f h)",
                            schedule.open_at.isoformat(), wait / 3600)
                self.clock.sleep(wait)
        return results

    def run_day(self, schedule: DaySchedule | None = None) -> DayResult:
        """Run one complete trading day."""
        schedule = schedule or build_schedule(self.config, self.clock.now())
        schedule = self._adjust_for_late_start(schedule)
        day = str(schedule.day)

        self._begin_day(day)
        briefing, plan = self.open_day(schedule)
        opened = self.execute_entries(plan, schedule)
        opened.extend(self.work_later_slots(schedule))
        self.manage_until(schedule.flatten_at, schedule)
        closed_at_flatten = self.flatten(schedule)
        result = self.close_day(schedule, briefing, plan, opened)
        result.closed.extend(closed_at_flatten)
        return result

    def stop(self) -> None:
        self.running = False

    def _adjust_for_late_start(self, schedule: DaySchedule) -> DaySchedule:
        """Give a mid-day start a slot of its own when it has landed between
        the day's scheduled ones.

        Entry slots already solve most of this: starting at 16:00 finds the
        16:00 slot live and needs no help. But a start at 02:30 sits in the
        gap between the day-open slot and the 04:00 one, and would trade
        nothing for ninety minutes despite having a current plan in hand.
        The plan is built at the current moment either way, so its levels
        are current — what is missing is only permission to act on them.
        """
        first_day, self._first_day = self._first_day, False
        if not (first_day and self.allow_late_start):
            return schedule

        now = self.clock.now()
        if schedule.slot_at(now) is not None:
            return schedule  # a slot is already live

        if now >= schedule.flatten_at:
            logger.info(
                "Started at %s, past today's flatten time — managing only; "
                "next session opens %s",
                now.strftime("%H:%M UTC"),
                schedule.close_at.strftime("%Y-%m-%d %H:%M UTC"),
            )
            return schedule

        window = timedelta(minutes=int(
            self.config.get("session", {}).get("entry_window_minutes", 120)
        ))
        end = min(now + window, schedule.flatten_at)
        # Do not run past the next scheduled slot; that one will handle it.
        upcoming = schedule.later_slots(now)
        if upcoming:
            end = min(end, upcoming[0].start)
        if end <= now:
            return schedule

        # Admit whatever is actually tradable right now. Naming the classes
        # explicitly rather than admitting everything keeps a stock out of
        # a window in which its market is shut.
        classes = frozenset(
            i.asset_class.value for i in self.universe if i.is_open(now)
        )
        if not classes:
            return schedule

        catch_up = EntrySlot(now, end, "late-start", classes)
        logger.info(
            "Late start at %s — opening a catch-up slot until %s (%s)",
            now.strftime("%H:%M UTC"), end.strftime("%H:%M UTC"),
            ",".join(sorted(classes)),
        )
        slots = tuple(sorted(schedule.slots + (catch_up,), key=lambda s: s.start))
        return DaySchedule(
            day=schedule.day,
            open_at=schedule.open_at,
            entry_until=max(schedule.entry_until, end),
            flatten_at=schedule.flatten_at,
            close_at=schedule.close_at,
            slots=slots,
        )

    # ── Phase 1: briefing + plan ──────────────────────────────

    def open_day(self, schedule: DaySchedule) -> tuple[Briefing, DayPlan]:
        now = self.clock.now()
        prices = self._latest_prices(self.symbols, now)
        equity = self.broker.equity(prices)

        rolling = self.journal.rolling_stats(
            window=int(self.config.get("journal", {}).get("rolling_window", 50))
        )
        yesterday = self.journal.last_day()

        briefing, frames = self.briefing_builder.build(
            symbols=self.symbols,
            day=str(schedule.day),
            equity=equity,
            cash=self.broker.cash,
            carried_positions=self.broker.positions,
            rolling_stats=rolling,
            yesterday=yesterday,
            now=now,
        )
        # The briefing's frames are kept only for reference; management
        # re-reads bars through _bars() so the tape is never stale.
        self._briefing_frames = frames
        self._refresh_market_limits(briefing)
        self._refresh_funding(briefing)
        # The risk gates need the same betas the briefing just measured;
        # without this they fall back to a static table that has no entry
        # for half the universe.
        self.risk.beta_book = self.briefing_builder.beta_book

        context = MarketContext(
            day=str(schedule.day),
            reads=briefing.symbols,
            frames=frames,
            funding=self._funding_rates,
            market_tone=float(briefing.market_tone.get("score", 0.0)),
            equity=equity,
            timeframe=self.timeframe,
        )

        weights, exposure = self._portfolio_state(equity)

        plan = self.planner.build(
            briefing=briefing,
            equity=equity,
            open_positions=self.broker.positions,
            peak_equity=max(self.state.peak_equity, equity),
            day_start_equity=self.state.day_start_equity or equity,
            context=context,
            strategy_weights=weights,
            exposure=exposure,
            market_limits=self._market_limits,
            consecutive_losses=self.state.consecutive_losses,
            cooldown_until=self._cooldown_until(),
            now=now,
            opened_today=self.state.trades_opened_today,
            opened_today_by_class=dict(self.state.opened_today_by_class or {}),
            entry_budget=self._entry_budget(schedule),
        )

        payload = briefing.to_dict()
        payload["plan"] = plan.to_dict()
        self.journal.save_briefing(schedule.day, payload)
        return briefing, plan

    def _portfolio_state(self, equity: float):
        """Strategy risk budget and the portfolio exposure multiplier.

        Both are derived from realised history: strategy weights from each
        driver's own return volatility and its correlation with the rest,
        the exposure multiplier from the account's daily return series.
        With too little history, both fall back to neutral rather than
        estimating a covariance matrix from a fortnight of noise.
        """
        names = [s.name for s in self.strategies]
        trades = self.journal.load_trades(limit=500)
        returns = self.tracker.daily_returns(trades, equity)

        volatilities = self.tracker.volatilities(returns)
        correlations = self.tracker.correlations(returns)
        weights = self.allocator.weights(names, volatilities, correlations)

        # The autopilot benches strategies that have earned their way out
        # and tilts the rest toward the prevailing tape, so the roster
        # manages itself rather than needing to be configured.
        roster = self.autopilot.decide(
            names, self.journal.strategy_stats(), regime=self._dominant_regime()
        )
        weights = self.autopilot.apply(weights, roster)
        self._roster = roster
        logger.info("  Strategy budget: %s", self.autopilot.explain(weights, roster))

        days = self.journal.load_days()
        daily_returns = [d.get("return_pct", 0.0) / 100 for d in days]
        drawdown = ((self.state.peak_equity - equity) / self.state.peak_equity * 100
                    if self.state.peak_equity > 0 else 0.0)
        exposure = self.vol_targeter.decide(daily_returns, drawdown_pct=drawdown)
        logger.info("  Exposure: %.2fx (%s)", exposure.scale, "; ".join(exposure.reasons))

        return weights, exposure

    # ── Phase 2: entries ──────────────────────────────────────

    def execute_entries(self, plan: DayPlan, schedule: DaySchedule) -> list[Position]:
        """Open the planned trades inside the entry window.

        Prices are re-checked at execution time: the plan was built on the
        open, and a symbol that has already run past its stop distance is no
        longer the trade that was planned.
        """
        opened: list[Position] = []
        if not plan.trades:
            logger.info("No entries to place.")
            return opened

        max_drift = float(self.config.get("session", {}).get("max_entry_drift_r", 0.5))

        for planned in plan.trades:
            now = self.clock.now()
            slot = schedule.slot_at(now)
            if slot is None:
                logger.info("Entry window closed — %s not taken", planned.symbol)
                break
            if not slot.admits(planned.asset_class):
                # A crypto slot must not open a stock, and vice versa —
                # the slot exists precisely because the two have different
                # hours.
                logger.debug("%s not admitted by %s", planned.symbol, slot.label)
                continue
            if self.state.halted_reason:
                logger.warning("Halted (%s) — no further entries", self.state.halted_reason)
                break

            price = self._price(planned.symbol, now)
            if price is None:
                logger.warning("No price for %s — skipping entry", planned.symbol)
                continue

            direction = 1 if planned.side == "long" else -1
            drift_r = (price - planned.entry_price) * direction / planned.r_distance \
                if planned.r_distance > 0 else 0.0
            if drift_r > max_drift:
                logger.info(
                    "%s moved %+.2fR since the plan — chasing it would take a worse "
                    "entry for the same stop, skipping", planned.symbol, drift_r,
                )
                continue

            order = SizedOrder(
                symbol=planned.symbol,
                side=planned.side,
                quantity=planned.quantity,
                entry_price=price,
                stop_price=price - direction * planned.r_distance,
                take_profit=price + direction * planned.r_distance
                * float(self.config.get("stops", {}).get("target_r_multiple", 2.0)),
                risk_usd=planned.quantity * planned.r_distance,
                notional=planned.quantity * price,
                leverage=1.0,
                atr=planned.atr,
                r_distance=planned.r_distance,
                asset_class=planned.asset_class,
            )

            try:
                position = self.broker.open(
                    order,
                    now=now,
                    day=str(schedule.day),
                    atr=planned.atr,
                    adv_notional=self._adv_notional(planned.symbol),
                    spread_bps=self._spread(planned.symbol),
                    daily_vol_bps=self._daily_vol_bps(planned.symbol),
                    strategy=planned.strategy or "blend",
                    asset_class=planned.asset_class,
                    venue=planned.venue,
                    timeframe=planned.timeframe,
                    entry_reason=planned.reason,
                    tags={"edge": planned.edge, "planned_entry": planned.entry_price},
                )
            except InsufficientFunds as e:
                logger.warning("Entry skipped: %s", e)
                continue

            opened.append(position)
            self.state.trades_opened_today += 1
            klass = str(planned.asset_class or "unknown")
            counts = self.state.opened_today_by_class or {}
            counts[klass] = counts.get(klass, 0) + 1
            self.state.opened_today_by_class = counts
            self._persist()

            if self.entry_stagger_seconds:
                self.clock.sleep(self.entry_stagger_seconds)

        return opened

    def work_later_slots(self, schedule: DaySchedule) -> list[Position]:
        """Manage the book to each remaining entry slot, then trade it.

        One window at the day open was enough when the universe was crypto
        only. It is not enough now: a US stock's session does not start
        until 13:30 UTC, so with a single window at midnight the equity
        half of the book would never open a position, and every 4-hour
        crypto bar after 02:00 would print unseen.

        Each slot gets a fresh briefing over only the instruments it
        admits, so the plan it acts on is built from the tape at that hour
        rather than from the morning's.
        """
        opened: list[Position] = []
        slots = schedule.later_slots(self.clock.now())
        if not slots:
            return opened

        logger.info("  %d further entry slot(s) today: %s",
                    len(slots), " | ".join(str(s) for s in slots))

        for slot in slots:
            if not self.running and self._first_day is False and self.state.halted_reason:
                break
            # Walk the book forward to the slot rather than jumping: stops
            # and targets between here and there must still be honoured.
            self.manage_until(slot.start, schedule)
            if self.state.halted_reason:
                logger.info("Halted (%s) — skipping %s",
                            self.state.halted_reason, slot.label)
                continue
            if self._day_entry_budget_spent():
                logger.info("Daily entry budget spent — skipping %s", slot.label)
                continue

            symbols = self._slot_symbols(slot)
            if not symbols:
                logger.info("  %s: no instrument open and admitted", slot)
                continue

            logger.info("─" * 62)
            logger.info("  ENTRY SLOT %s — %d instrument(s)", slot, len(symbols))
            plan = self._replan(schedule, symbols)
            opened.extend(self.execute_entries(plan, schedule))

        return opened

    def _slot_symbols(self, slot) -> list[str]:
        """Instruments this slot admits whose market is also open."""
        now = self.clock.now()
        return [i.symbol for i in self.universe
                if slot.admits(i.asset_class) and i.is_open(now)]

    def _day_entry_budget_spent(self) -> bool:
        cap = int(self.config.get("session", {}).get("max_new_positions_per_day", 3))
        return self.state.trades_opened_today >= cap

    def _entry_budget(self, schedule: DaySchedule) -> int:
        """The daily entry cap, less what later-opening classes are owed.

        Crypto's first slot opens at midnight and the US session not until
        13:30. Without a reservation crypto spends the whole day's budget
        before a stock can be looked at — which it did: a ten-day replay
        took thirty crypto trades and two equity ones, so the cross-asset
        comparison the reservation exists to make was crypto versus noise.

        So classes that still have a slot ahead of them today keep their
        per-class share in reserve, and the budget offered to the current
        slot is whatever is left over.
        """
        session = self.config.get("session", {})
        cap = int(session.get("max_new_positions_per_day", 3))
        per_class = int(session.get("max_new_positions_per_class", 0) or 0)
        if per_class <= 0:
            return cap

        now = self.clock.now()
        opened = dict(self.state.opened_today_by_class or {})
        current = schedule.slot_at(now)
        pending = set()
        for slot in schedule.later_slots(now):
            for instrument in self.universe:
                klass = instrument.asset_class.value
                if current is not None and current.admits(klass):
                    continue          # this class is being served right now
                if slot.admits(klass):
                    pending.add(klass)

        reserved = sum(max(0, per_class - opened.get(k, 0)) for k in pending)
        return max(1, cap - reserved)

    def _replan(self, schedule: DaySchedule, symbols: list[str]) -> DayPlan:
        """A plan for one slot, over the instruments it admits."""
        now = self.clock.now()
        prices = self._latest_prices(self.symbols, now)
        equity = self.broker.equity(prices)

        briefing, frames = self.briefing_builder.build(
            symbols=symbols,
            day=str(schedule.day),
            equity=equity,
            cash=self.broker.cash,
            carried_positions=self.broker.positions,
            rolling_stats=self.journal.rolling_stats(
                window=int(self.config.get("journal", {}).get("rolling_window", 50))
            ),
            yesterday=self.journal.last_day(),
            now=now,
        )
        self._refresh_market_limits(briefing)
        self._refresh_funding(briefing)
        self.risk.beta_book = self.briefing_builder.beta_book
        # Management re-reads bars on the timeframe the slot's briefing
        # chose, so a slot that stepped an instrument to a faster frame is
        # then managed on that frame too.
        reads = dict(getattr(self, "_briefing_reads", {}) or {})
        reads.update(briefing.symbols)
        self._briefing_reads = reads

        context = MarketContext(
            day=str(schedule.day),
            reads=briefing.symbols,
            frames=frames,
            funding=self._funding_rates,
            market_tone=float(briefing.market_tone.get("score", 0.0)),
            equity=equity,
            timeframe=self.timeframe,
        )
        weights, exposure = self._portfolio_state(equity)
        return self.planner.build(
            briefing=briefing,
            equity=equity,
            open_positions=self.broker.positions,
            peak_equity=max(self.state.peak_equity, equity),
            day_start_equity=self.state.day_start_equity or equity,
            context=context,
            strategy_weights=weights,
            exposure=exposure,
            market_limits=self._market_limits,
            consecutive_losses=self.state.consecutive_losses,
            cooldown_until=self._cooldown_until(),
            now=now,
            opened_today=self.state.trades_opened_today,
            opened_today_by_class=dict(self.state.opened_today_by_class or {}),
            entry_budget=self._entry_budget(schedule),
        )

    # ── Phase 3: management ───────────────────────────────────

    def manage_until(self, until: datetime, schedule: DaySchedule) -> list[Trade]:
        """Walk the open book until `until`, honouring stops and breakers."""
        closed: list[Trade] = []
        while True:
            now = self.clock.now()
            if now >= until:
                break
            closed.extend(self.manage_once(schedule))
            remaining = (until - self.clock.now()).total_seconds()
            if remaining <= 0:
                break
            self.clock.sleep(min(self.manage_interval, remaining))
        return closed

    def manage_once(self, schedule: DaySchedule) -> list[Trade]:
        """One management pass.

        Bar-driven rules (stop and target touches, the bar counter, the time
        stop) only fire on bars that have actually closed since the previous
        pass. Price-driven rules (marking, trailing) run every pass. Running
        the bar rules on every pass instead would re-judge the same candle
        repeatedly and burn the time stop in minutes.
        """
        now = self.clock.now()
        closed: list[Trade] = []

        if not self.broker.positions:
            self._check_breakers(now, schedule)
            return closed

        # Fetch once for the whole pass: the breakers need every symbol's
        # price and the marking loop needs the held ones, so asking for the
        # union here saves a second round of lookups.
        prices = self._latest_prices(self.symbols, now)

        last = getattr(self, "_last_mark_time", None) or now
        if now > last:
            self.broker.accrue_funding(last, now, self._funding_rates, prices)
        self._last_mark_time = now

        for position in list(self.broker.positions):
            price = prices.get(position.symbol)
            if price is None:
                continue
            self.broker.mark(position.symbol, price)

            # Bars closed since this position was last looked at.
            new_bars = self._new_bars(position.symbol, position.opened_at)
            exited = False
            for bar in new_bars:
                reason = self.broker.stop_or_target_hit(
                    position, float(bar["high"]), float(bar["low"])
                )
                if reason:
                    # Fill at the barrier itself: a stop fills where it sat,
                    # and assuming a better fill is how backtests flatter.
                    fill = (position.take_profit if reason == "take_profit"
                            else position.stop_price)
                    closed.append(self._close(position, fill, reason, now, schedule))
                    exited = True
                    break

                liq = self.broker.liquidation_price(position)
                if liq is not None and (
                    (position.direction == 1 and float(bar["low"]) <= liq)
                    or (position.direction == -1 and float(bar["high"]) >= liq)
                ):
                    closed.append(self._close(position, liq, "liquidation", now, schedule))
                    exited = True
                    break

                position.bars_held += 1
                if self.risk.time_stop_hit(position, position.bars_held):
                    closed.append(self._close(position, float(bar["close"]),
                                              "time_stop", now, schedule))
                    exited = True
                    break

            if exited:
                continue

            if self.risk.max_hold_exceeded(position, now, self.max_holding_days):
                closed.append(self._close(position, price, "max_hold", now, schedule))
                continue

            atr = self._current_atr(position.symbol) or position.atr_at_entry

            # Bank some of the open gain, then press the rest. Both are
            # the same idea as a trailing stop applied at different ends:
            # the trail protects a winner, scaling out converts part of it
            # to cash, and pyramiding presses it.
            self._book_profit(position, price, now)
            self._pyramid(position, price, atr, now, schedule)

            new_stop, stop_reason = self.risk.update_stop(position, price, atr)
            if stop_reason and new_stop != position.stop_price:
                logger.info(
                    "  %s stop %s: %.6f → %.6f (%.2fR open)",
                    position.symbol, stop_reason, position.stop_price, new_stop,
                    position.unrealized_r(price),
                )
                position.stop_price = new_stop
                if stop_reason == "breakeven":
                    position.moved_to_breakeven = True

        self._check_breakers(now, schedule, prices)
        self._persist()
        return closed

    def _book_profit(self, position: Position, price: float,
                     now: datetime) -> None:
        """Sell a slice at each profit rung the position has reached.

        A trend system's standing complaint is that it hands most of an
        open gain back waiting for the trailing stop to catch up. Taking a
        fixed fraction off at set multiples of risk converts some of that
        paper profit into cash while leaving the rest to run.

        Each rung fires once. The position keeps its original entry, stop
        and R measurement, so a trade that sold half at 2R and then
        stopped at breakeven is recorded as the winner it was.
        """
        if not self.scale_out_levels or position.quantity <= 0:
            return
        if self.scale_out_classes and \
                str(position.asset_class) not in self.scale_out_classes:
            return

        open_r = position.unrealized_r(price)
        for level, fraction in self.scale_out_levels:
            if open_r < level or level in position.scaled_out_at_levels:
                continue
            # Never sell the last of it here; the stop and target own the
            # exit, and a position closed by a scale-out would bypass the
            # trade record entirely.
            remaining = position.quantity * (1.0 - fraction)
            if remaining <= 0:
                continue
            self.broker.scale_out(
                position, price, fraction,
                now=now,
                adv_notional=self._adv_notional(position.symbol),
                spread_bps=self._spread(position.symbol),
                daily_vol_bps=self._daily_vol_bps(position.symbol),
            )
            position.scaled_out_at_levels.append(level)

    def _pyramid(self, position: Position, price: float, atr: float,
                 now: datetime, schedule: DaySchedule) -> None:
        """Add a unit to a position that is working.

        The entry-side counterpart of the trailing stop. Adding is gated
        on the same things a fresh entry is — the risk budget, the entry
        slot, the daily cap — because a pyramid unit is a new position in
        everything but name, and exempting it would let the book grow past
        limits it was told to respect.
        """
        if self.max_units <= 1 or position.units >= self.max_units:
            return
        if self.pyramid_classes and \
                str(position.asset_class) not in self.pyramid_classes:
            return
        if self.state.halted_reason:
            return
        if not schedule.admits(position.asset_class, now):
            return

        open_r = position.unrealized_r(price)
        if open_r < self.pyramid_step_r * position.units:
            return
        if atr <= 0:
            return

        equity = self.broker.equity(self._latest_prices(self.symbols, now))
        limits = self._market_limits.get(position.symbol, {})
        order = self.risk.size_order(
            symbol=position.symbol,
            side=position.side,
            entry_price=price,
            atr=atr,
            equity=equity,
            # Each unit risks less than the one before it, so a pyramid
            # cannot quietly turn a 0.75% trade into a 3% one.
            risk_pct=self.risk.risk_per_trade_pct() * self.pyramid_size_factor,
            min_qty=limits.get("min_qty", 0.0),
            qty_step=limits.get("qty_step", 0.0),
            min_notional=limits.get("min_notional", 0.0),
            asset_class=position.asset_class,
        )
        if not order.valid:
            return

        decision = self.risk.check_new_trade(
            order,
            [p for p in self.broker.positions if p is not position],
            equity=equity,
            day_start_equity=self.state.day_start_equity or equity,
            peak_equity=max(self.state.peak_equity, equity),
        )
        if not decision.allowed:
            logger.debug("Pyramid on %s refused: %s", position.symbol,
                         decision.reason)
            return

        self.broker.add_to(
            position, order, now=now,
            adv_notional=self._adv_notional(position.symbol),
            spread_bps=self._spread(position.symbol),
            daily_vol_bps=self._daily_vol_bps(position.symbol),
        )

    # ── Phase 4: flatten ──────────────────────────────────────

    def flatten(self, schedule: DaySchedule, force: bool = False) -> list[Trade]:
        """Close positions the day is not meant to carry."""
        closed: list[Trade] = []
        if not self.broker.positions:
            return closed

        now = self.clock.now()
        prices = self._latest_prices(
            sorted({p.symbol for p in self.broker.positions}), now
        )

        for position in list(self.broker.positions):
            if not force and self.carry_overnight and not self._must_close(position, now):
                logger.info(
                    "  Carrying %s %s into the next session (%.2fR open)",
                    position.side, position.symbol,
                    position.unrealized_r(prices.get(position.symbol, position.entry_price)),
                )
                continue
            price = prices.get(position.symbol)
            if price is None:
                logger.warning("No price to flatten %s — carrying it", position.symbol)
                continue
            reason = "end_of_day" if not force else "forced_flat"
            closed.append(self._close(position, price, reason, now, schedule))
        return closed

    def _must_close(self, position: Position, now: datetime) -> bool:
        """Even in carry mode, some positions have to go."""
        if self.state.halted_reason:
            return True
        return self.risk.max_hold_exceeded(position, now, self.max_holding_days)

    # ── Phase 5: report ───────────────────────────────────────

    def close_day(self, schedule: DaySchedule, briefing: Briefing | None,
                  plan: DayPlan | None, opened: list[Position]) -> DayResult:
        """Write the day down and roll state forward."""
        now = self.clock.now()
        prices = self._latest_prices(self.symbols, now)
        equity = self.broker.equity(prices)

        trades = list(self.broker.closed_today)
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))

        start_equity = self.state.day_start_equity or equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        drawdown = ((self.state.peak_equity - equity) / self.state.peak_equity * 100
                    if self.state.peak_equity > 0 else 0.0)

        summary = DaySummary(
            day=str(schedule.day),
            starting_equity=round(start_equity, 2),
            ending_equity=round(equity, 2),
            realized_pnl=round(sum(t.pnl for t in trades), 2),
            unrealized_pnl=round(self.broker.unrealized_pnl(prices), 2),
            return_pct=round((equity - start_equity) / start_equity * 100, 4)
            if start_equity > 0 else 0.0,
            trades_opened=len(opened),
            trades_closed=len(trades),
            wins=len(wins),
            losses=len(losses),
            win_rate=round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
            avg_r=round(sum(t.r_multiple for t in trades) / len(trades), 4) if trades else 0.0,
            gross_profit=round(gross_profit, 2),
            gross_loss=round(gross_loss, 2),
            profit_factor=round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0,
            fees=round(sum(t.fees for t in trades), 4),
            funding=round(sum(t.funding for t in trades), 4),
            max_drawdown_pct=round(drawdown, 3),
            peak_equity=round(self.state.peak_equity, 2),
            positions_carried=len(self.broker.positions),
            halted_reason=self.state.halted_reason or "",
            news_bias=float(briefing.market_tone.get("score", 0.0)) if briefing else 0.0,
            regime=_dominant_regime(briefing),
        )

        for trade in trades:
            self.journal.append_trade(trade)
        self.journal.append_day(summary)
        self._print_day_report(summary, trades)

        # Roll state into tomorrow.
        self.state.last_completed_day = str(schedule.day)
        self.state.current_day = None
        self.state.cash = self.broker.cash
        self.state.positions = list(self.broker.positions)
        self.state.trade_counter = self.broker.trade_counter
        self.state.symbol_stats = self.journal.symbol_stats()
        # The halt is for the day that hit it; tomorrow starts clean unless
        # the total-drawdown breaker is still tripped, which open_day
        # re-checks against live equity.
        self.state.halted_reason = None
        self.journal.save_state(self.state)

        result = DayResult(summary=summary, plan=plan, briefing=briefing,
                           opened=opened, closed=trades)
        self.broker.closed_today = []
        return result

    # ── Internals ─────────────────────────────────────────────

    def _begin_day(self, day: str) -> None:
        prices = self._latest_prices(self.symbols, self.clock.now())
        equity = self.broker.equity(prices)
        self.state.current_day = day
        self.state.day_start_equity = equity
        self.state.trades_opened_today = 0
        self.state.opened_today_by_class = {}
        self.state.trades_closed_today = 0
        self.state.realized_pnl_today = 0.0
        self.state.halted_reason = None
        if not self.state.initial_equity:
            self.state.initial_equity = equity
        self.state.peak_equity = max(self.state.peak_equity, equity)
        self.broker.closed_today = []
        self._last_mark_time = self.clock.now()
        self._bar_cache = {}
        self._last_bar_ts = {}
        logger.info("═" * 62)
        logger.info("  SESSION OPEN — %s | equity $%.2f", day, equity)
        logger.info("═" * 62)

    def _close(self, position: Position, price: float, reason: str,
               now: datetime, schedule: DaySchedule) -> Trade:
        trade = self.broker.close(
            position, price, reason, now=now, day=str(schedule.day),
            adv_notional=self._adv_notional(position.symbol),
            spread_bps=self._spread(position.symbol),
            daily_vol_bps=self._daily_vol_bps(position.symbol),
        )
        self.state.trades_closed_today += 1
        self.state.realized_pnl_today += trade.pnl

        if trade.pnl > 0:
            self.state.consecutive_wins += 1
            self.state.consecutive_losses = 0
        else:
            self.state.consecutive_losses += 1
            self.state.consecutive_wins = 0
            threshold = int(self.config.get("risk", {}).get("cooldown_after_losses", 2))
            if self.state.consecutive_losses >= threshold:
                minutes = int(self.config.get("risk", {}).get("cooldown_minutes", 120))
                until = now + timedelta(minutes=minutes)
                self.state.cooldown_until = until.isoformat()
                logger.info(
                    "  %d losses in a row — no new entries for %d minutes",
                    self.state.consecutive_losses, minutes,
                )
        self._persist()
        return trade

    def _check_breakers(self, now: datetime, schedule: DaySchedule,
                        prices: dict[str, float] | None = None) -> None:
        """Daily loss and total drawdown, measured on equity.

        Takes the prices the caller already fetched where it has them: a
        management pass otherwise asks for every symbol twice, which in a
        replay is thousands of redundant lookups per simulated day.
        """
        prices = prices if prices is not None else self._latest_prices(self.symbols, now)
        equity = self.broker.equity(prices)
        self.state.peak_equity = max(self.state.peak_equity, equity)

        risk = self.config.get("risk", {})
        start = self.state.day_start_equity or equity
        if start > 0:
            day_pct = (equity - start) / start * 100
            if day_pct <= -abs(float(risk.get("max_daily_loss_pct", 2.0))):
                if self.state.halted_reason != "daily_loss_limit":
                    logger.warning(
                        "DAILY LOSS LIMIT: %.2f%% — flattening and standing down",
                        day_pct,
                    )
                    self.state.halted_reason = "daily_loss_limit"
                    self.flatten(schedule, force=True)
                return

        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - equity) / self.state.peak_equity * 100
            if dd >= float(risk.get("max_total_drawdown_pct", 15.0)):
                if self.state.halted_reason != "max_drawdown_halt":
                    logger.error(
                        "MAX DRAWDOWN %.2f%% — halting and flattening", dd
                    )
                    self.state.halted_reason = "max_drawdown_halt"
                    self.flatten(schedule, force=True)

    def _dominant_regime(self) -> str:
        """The regime most of the universe is in, for the autopilot tilt."""
        reads = getattr(self, "_briefing_reads", {}) or {}
        counts: dict[str, int] = {}
        for read in reads.values():
            if read.tradable:
                counts[read.regime] = counts.get(read.regime, 0) + 1
        return max(counts, key=counts.get) if counts else ""

    def _cooldown_until(self) -> datetime | None:
        if not self.state.cooldown_until:
            return None
        try:
            return datetime.fromisoformat(self.state.cooldown_until)
        except ValueError:
            return None

    def _persist(self) -> None:
        self.state.cash = self.broker.cash
        self.state.positions = list(self.broker.positions)
        self.state.trade_counter = self.broker.trade_counter
        self.journal.save_state(self.state)

    # ── Market data access ────────────────────────────────────

    def _price(self, symbol: str, now: datetime) -> float | None:
        return self._latest_prices([symbol], now).get(symbol)

    def _latest_prices(self, symbols: list[str], now: datetime) -> dict[str, float]:
        if not symbols:
            return {}
        if self._price_source is not None:
            return self._price_source(symbols, now)

        prices: dict[str, float] = {}
        for symbol in symbols:
            instrument = self.by_symbol.get(symbol)
            if instrument is None or not instrument.is_open(now):
                continue
            try:
                price = self.router.price(instrument)
            except Exception as e:
                logger.debug("Price unavailable for %s: %s", symbol, e)
                continue
            if price:
                prices[symbol] = float(price)
        return prices

    def _bars(self, symbol: str, limit: int = 120):
        """Fresh bars for a symbol, cached for the current clock instant.

        Reading from the briefing snapshot instead — as an earlier version
        did — froze the tape at the morning's candle for the whole day.
        """
        now = self.clock.now()
        cache = getattr(self, "_bar_cache", None)
        if cache is None:
            cache = self._bar_cache = {}
        hit = cache.get(symbol)
        if hit and hit[0] == now:
            return hit[1]

        instrument = self.by_symbol.get(symbol)
        if instrument is None:
            return None
        read = getattr(self, "_briefing_reads", {}).get(symbol)
        timeframe = read.timeframe if read else instrument.timeframe
        try:
            df = self.router.bars(instrument, timeframe, limit)
            if df is not None and df.empty:
                df = None
        except Exception as e:
            logger.debug("Bars unavailable for %s: %s", symbol, e)
            df = None
        cache[symbol] = (now, df)
        return df

    def _new_bars(self, symbol: str, not_before: datetime) -> list:
        """Bars that closed since the last pass, newest last.

        Bars that closed before the position existed are skipped: a candle
        whose low pierced the stop level an hour before entry is not a
        stop-out of a trade that had not been placed.
        """
        df = self._bars(symbol)
        if df is None or df.empty:
            return []

        seen = getattr(self, "_last_bar_ts", None)
        if seen is None:
            seen = self._last_bar_ts = {}

        cutoff = seen.get(symbol)
        rows = []
        for ts, row in df.iterrows():
            if ts <= not_before:
                continue
            if cutoff is not None and ts <= cutoff:
                continue
            rows.append(row)
        if len(df):
            seen[symbol] = df.index[-1]
        return rows

    def _current_atr(self, symbol: str) -> float | None:
        df = self._bars(symbol)
        if df is None or len(df) < 20:
            return None
        from bot.analysis.indicators import atr, last_value
        period = int(self.config.get("stops", {}).get("atr_period", 14))
        return last_value(atr(df, period))

    def _adv_notional(self, symbol: str) -> float | None:
        read = getattr(self, "_briefing_reads", {}).get(symbol)
        if read and read.quote_volume_24h:
            return read.quote_volume_24h
        return None

    def _daily_vol_bps(self, symbol: str) -> float | None:
        """Daily volatility in bps, for the square-root impact model."""
        read = getattr(self, "_briefing_reads", {}).get(symbol)
        if read and read.annualized_vol:
            return float(read.annualized_vol) / (365 ** 0.5) * 10_000
        return None

    def _spread(self, symbol: str) -> float | None:
        read = getattr(self, "_briefing_reads", {}).get(symbol)
        if read and read.spread_bps:
            return read.spread_bps
        return None

    def _refresh_market_limits(self, briefing: Briefing) -> None:
        self._briefing_reads = briefing.symbols
        for symbol, read in briefing.symbols.items():
            if not read.tradable:
                continue
            instrument = self.by_symbol.get(symbol)
            if instrument is None:
                continue
            try:
                self._market_limits[symbol] = self.router.market_limits(instrument)
            except Exception:
                self._market_limits[symbol] = {}

    def _refresh_funding(self, briefing: Briefing) -> None:
        self._funding_rates = {
            symbol: read.funding_rate
            for symbol, read in briefing.symbols.items()
            if read.funding_rate
        }

    # ── Reporting ─────────────────────────────────────────────

    def _print_day_report(self, summary: DaySummary, trades: list[Trade]) -> None:
        logger.info("═" * 62)
        logger.info("  DAY REPORT — %s", summary.day)
        logger.info("═" * 62)
        logger.info("  Equity      : $%.2f → $%.2f (%+.2f%%)",
                    summary.starting_equity, summary.ending_equity, summary.return_pct)
        logger.info("  Trades      : %d opened / %d closed (%dW / %dL)",
                    summary.trades_opened, summary.trades_closed,
                    summary.wins, summary.losses)
        if trades:
            logger.info("  Avg R       : %+.2f | Profit factor %.2f",
                        summary.avg_r, summary.profit_factor)
            for t in trades:
                logger.info("    %-5s %-10s %+7.2fR  $%+8.2f  %s",
                            t.side.upper(), t.symbol, t.r_multiple, t.pnl, t.exit_reason)
        logger.info("  Costs       : fees $%.2f | funding $%+.4f",
                    summary.fees, summary.funding)
        logger.info("  Carried     : %d position(s)", summary.positions_carried)
        if summary.halted_reason:
            logger.info("  Halted      : %s", summary.halted_reason)
        logger.info("═" * 62)


def _dominant_regime(briefing: Briefing | None) -> str:
    if not briefing or not briefing.symbols:
        return ""
    counts: dict[str, int] = {}
    for read in briefing.symbols.values():
        counts[read.regime] = counts.get(read.regime, 0) + 1
    return max(counts, key=counts.get) if counts else ""
