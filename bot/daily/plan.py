"""Turning the morning briefing into a concrete plan for the day.

The plan is built in four steps, each a separate concern:

  1. **Strategies speak.** Every enabled strategy — trend, cross-sectional
     momentum, breakout, reversion, carry — sees the same market context
     and returns its own signals. None of them knows about the others.
  2. **The allocator resolves them.** Strategies are weighted by inverse
     volatility with a correlation haircut, then their signals are merged
     per symbol, so two strategies disagreeing cancel instead of opening
     two opposed positions in the same name.
  3. **News and the higher timeframe can veto.** Sentiment is a tilt and a
     veto, never a trigger: headline sentiment decays in hours and is a
     weak standalone predictor.
  4. **Risk sizes what survives.** Fixed dollar risk at an ATR stop, scaled
     by the portfolio's volatility-target multiplier, then run through the
     heat, correlation and breaker gates.

The reason this is structured as separate strategies rather than one
weighted formula is the evidence on managed futures: combining weakly
correlated return drivers is what shrinks drawdown. One formula with five
terms is still one return driver.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from bot.daily.briefing import Briefing, SymbolRead
from bot.portfolio.allocator import CombinedView, StrategyAllocator
from bot.portfolio.voltarget import ExposureDecision
from bot.risk.budget import RiskBudget, SizedOrder
from bot.strategies.base import MarketContext, StrategySignal

logger = logging.getLogger("trading_bot")

# Position-size multiplier by regime — press less when the tape is hostile.
REGIME_SIZE = {
    "trending_up": 1.0, "trending_down": 1.0, "ranging": 0.85,
    "volatile": 0.55, "quiet": 0.75,
}


@dataclass
class Candidate:
    """A resolved trade idea, before sizing."""

    symbol: str
    side: str
    edge: float
    components: dict = field(default_factory=dict)
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlannedTrade:
    """A candidate that has been sized and cleared the risk gates."""

    symbol: str
    side: str
    edge: float
    quantity: float
    entry_price: float
    stop_price: float
    take_profit: float
    risk_usd: float
    notional: float
    atr: float
    r_distance: float
    components: dict = field(default_factory=dict)
    reason: str = ""
    caps: list = field(default_factory=list)
    strategy: str = ""
    asset_class: str = "crypto_spot"
    venue: str = ""
    timeframe: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DayPlan:
    """The plan for one trading day."""

    day: str
    created_at: str
    trades: list = field(default_factory=list)          # PlannedTrade
    rejected: list = field(default_factory=list)        # [symbol, reason]
    considered: list = field(default_factory=list)      # Candidate
    risk_pct_per_trade: float = 0.0
    strategy_weights: dict = field(default_factory=dict)
    exposure: dict = field(default_factory=dict)
    signals: dict = field(default_factory=dict)         # strategy -> [signal dicts]
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "created_at": self.created_at,
            "risk_pct_per_trade": round(self.risk_pct_per_trade, 4),
            "strategy_weights": {k: round(v, 4) for k, v in self.strategy_weights.items()},
            "exposure": self.exposure,
            "trades": [t.to_dict() for t in self.trades],
            "considered": [c.to_dict() for c in self.considered],
            "signals": self.signals,
            "rejected": self.rejected,
            "notes": self.notes,
        }


class DayPlanner:
    """Runs the strategies, allocates between them, and sizes the result."""

    def __init__(self, config: dict, risk: RiskBudget,
                 strategies: list | None = None,
                 allocator: StrategyAllocator | None = None):
        self.config = config
        self.risk = risk

        from bot.strategies import build_strategies
        self.strategies = strategies if strategies is not None else build_strategies(config)
        self.allocator = allocator or StrategyAllocator(config)

        signals = config.get("signals", {})
        self.min_edge = float(signals.get("min_edge_score", 0.15))
        self.news_weight = float(signals.get("news_weight", 0.15))
        self.book_weight = float(signals.get("book_weight", 0.06))
        self.require_htf = bool(signals.get("require_htf_agreement", True))
        self.require_primary = bool(signals.get("require_primary_trend", True))
        # Only for measuring what the shorting rule is worth. Leaving it on
        # books positions that cannot be executed on a spot exchange.
        self.allow_spot_shorts = bool(signals.get("allow_spot_shorts", False))
        # Longs must rank in the top (1 - band) of their class, shorts in
        # the bottom. 0 disables the filter; 0.5 would admit only the
        # single strongest and weakest name.
        self.momentum_band = float(signals.get("momentum_band", 0.0) or 0.0)
        self.momentum_min_peers = int(signals.get("momentum_min_peers", 4))
        self.veto_on_news = bool(signals.get("news_can_veto", True))
        session = config.get("session", {})
        self.max_new = int(session.get("max_new_positions_per_day", 3))
        # 0 means "no per-class reservation" — the whole budget is shared.
        self.max_new_per_class = int(session.get("max_new_positions_per_class", 0) or 0)

    # ── Step 1-2: strategies and allocation ───────────────────

    def collect_signals(self, ctx: MarketContext) -> dict[str, list[StrategySignal]]:
        """Ask every strategy for its view. One failing must not stop the day."""
        out: dict[str, list[StrategySignal]] = {}
        for strategy in self.strategies:
            try:
                signals = strategy.generate(ctx)
            except Exception as e:
                logger.warning("Strategy '%s' failed: %s", strategy.name, e, exc_info=True)
                continue
            if signals:
                out[strategy.name] = signals
        return out

    def resolve(self, ctx: MarketContext, weights: dict[str, float]
                ) -> tuple[dict[str, CombinedView], dict[str, list[StrategySignal]]]:
        """Strategy signals in, one view per symbol out."""
        by_strategy = self.collect_signals(ctx)
        views = self.allocator.combine(by_strategy, weights)
        return views, by_strategy

    def apply_tilts(self, view: CombinedView, read: SymbolRead,
                    market_tone: float) -> float:
        """Nudge conviction with news, order book and the broad tape.

        These are modifiers on a decision the strategies already made. None
        of them can open a position on its own.
        """
        edge = view.direction * view.conviction
        edge += self.news_weight * read.news_tilt
        edge += self.book_weight * max(-1.0, min(1.0, read.book_imbalance * 2))
        edge += 0.04 * market_tone
        return max(-1.0, min(1.0, edge))

    # ── Step 3-4: filters and sizing ──────────────────────────

    def build(
        self,
        briefing: Briefing,
        equity: float,
        open_positions: list,
        peak_equity: float,
        day_start_equity: float,
        context: MarketContext | None = None,
        strategy_weights: dict[str, float] | None = None,
        exposure: ExposureDecision | None = None,
        market_limits: dict | None = None,
        consecutive_losses: int = 0,
        cooldown_until: datetime | None = None,
        now: datetime | None = None,
        opened_today: int = 0,
        opened_today_by_class: dict | None = None,
        entry_budget: int | None = None,
    ) -> DayPlan:
        """Score, rank, size and gate the day's trades."""
        now = now or datetime.now(timezone.utc)
        market_limits = market_limits or {}

        drawdown_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0.0
        risk_pct = self.risk.risk_per_trade_pct(
            rolling=briefing.rolling_stats,
            drawdown_pct=drawdown_pct,
            consecutive_losses=consecutive_losses,
        )

        # The portfolio's volatility target scales every position together,
        # which is where most of the drawdown control comes from.
        exposure_scale = exposure.scale if exposure else 1.0
        risk_pct *= exposure_scale

        plan = DayPlan(day=briefing.day, created_at=now.isoformat(),
                       risk_pct_per_trade=risk_pct,
                       strategy_weights=dict(strategy_weights or {}),
                       exposure=exposure.to_dict() if exposure else {})

        ctx = context or MarketContext(
            day=briefing.day, reads=briefing.symbols, frames={},
            market_tone=float(briefing.market_tone.get("score", 0.0)), equity=equity,
        )
        weights = strategy_weights or self.allocator.weights([s.name for s in self.strategies])
        views, by_strategy = self.resolve(ctx, weights)
        plan.signals = {name: [s.to_dict() for s in signals]
                        for name, signals in by_strategy.items()}

        tone = float(briefing.market_tone.get("score", 0.0))
        candidates: list[tuple[Candidate, SymbolRead, CombinedView]] = []

        for symbol, read in briefing.symbols.items():
            if not read.tradable:
                plan.rejected.append([symbol, read.skip_reason or "not tradable"])
                continue

            view = views.get(symbol)
            if view is None:
                plan.rejected.append([symbol, "no strategy has a view"])
                continue

            edge = self.apply_tilts(view, read, tone)
            side = "long" if edge > 0 else "short"
            candidate = Candidate(
                symbol=symbol, side=side, edge=round(edge, 4),
                components={
                    "strategies": {k: round(v, 4) for k, v in view.contributors.items()},
                    "agreement": round(view.agreement, 4),
                    "contrarian_share": round(view.contrarian_share, 4),
                    "news": round(read.news_tilt, 4),
                    "book": round(read.book_imbalance, 4),
                    "market_tone": round(tone, 4),
                },
                reason=self._describe(view, read, edge),
            )
            plan.considered.append(candidate)

            # The bar to clear differs by asset class. A perp is held on
            # leverage with a stop twice as wide as spot's, so each one
            # occupies the book for longer and ties up more of its risk
            # budget; it should be a position the day is confident about,
            # not one of a stream. Over 90 days the perp leg took 73
            # trades and lost on 47 of them.
            min_edge = self._min_edge(read)
            if abs(edge) < min_edge:
                plan.rejected.append(
                    [symbol, f"edge {edge:+.3f} below {min_edge:.2f} "
                             f"for {read.asset_class or 'this class'}"]
                )
                continue

            # A tilt must not be able to flip the strategies' direction.
            if view.direction != (1 if edge > 0 else -1):
                plan.rejected.append([symbol, "tilts flipped the signal — standing aside"])
                continue

            # Can this instrument actually be sold short? Spot crypto cannot,
            # without a margin facility this bot does not model. The
            # capability was declared on the asset class from the start but
            # never consulted, and a 90-day replay took 68 spot shorts out of
            # 100 spot trades — none of them executable. They were not merely
            # phantom P&L: they consumed the daily entry budget, portfolio
            # heat and cash that the executable half of the book needed.
            if side == "short" and not self.allow_spot_shorts \
                    and not self._can_short(read):
                plan.rejected.append(
                    [symbol, f"{read.asset_class or 'spot'} cannot be sold short"]
                )
                continue

            # The major trend, before anything faster. When it is
            # unambiguous, a trade against it is refused outright: over a
            # window where crypto rose 42-77% the bot took 120 shorts and
            # 67 longs, and the shorts lost. The 4h filter below cannot see
            # a move on that scale — 55 of its bars is nine days.
            if (self.require_primary and read.primary_trend
                    and not self._neutral_view(view)):
                wants_long = side == "long"
                if (wants_long and read.primary_trend < 0) or \
                        (not wants_long and read.primary_trend > 0):
                    plan.rejected.append([
                        symbol,
                        f"{side} against the {self.primary_label(read)} "
                        f"({read.primary_timeframe or 'primary'})",
                    ])
                    continue

            if self.require_htf and read.htf_trend and not self._neutral_view(view):
                wants_long = side == "long"
                if (wants_long and read.htf_trend < 0) or (not wants_long and read.htf_trend > 0):
                    plan.rejected.append([symbol, f"{side} against {self.htf_label(read)}"])
                    continue

            # Relative strength within the asset class. In a market where
            # everything rose 42-77% over three months, "it is trending up"
            # was true of the whole universe and picked nothing; the rank
            # is what separates the leaders from the laggards. Applied only
            # where there are enough peers for a rank to mean anything.
            if self.momentum_band > 0 and read.momentum_peers >= self.momentum_min_peers:
                if side == "long" and read.momentum_rank < self.momentum_band:
                    plan.rejected.append([
                        symbol,
                        f"long but ranks {read.momentum_rank:.0%} of "
                        f"{read.momentum_peers} in its class",
                    ])
                    continue
                if side == "short" and read.momentum_rank > 1.0 - self.momentum_band:
                    plan.rejected.append([
                        symbol,
                        f"short but ranks {read.momentum_rank:.0%} of "
                        f"{read.momentum_peers} in its class",
                    ])
                    continue

            if self.veto_on_news and read.news_tilt:
                against = (side == "long" and read.news_tilt < -0.3) or \
                          (side == "short" and read.news_tilt > 0.3)
                if against:
                    plan.rejected.append([symbol, f"news {read.news_tilt:+.2f} opposes {side}"])
                    continue

            candidates.append((candidate, read, view))

        # Rank by conviction, then liquidity: the same edge is worth more
        # where it can be executed cheaply.
        candidates.sort(key=lambda t: (abs(t[0].edge), t[1].quote_volume_24h), reverse=True)

        book = list(open_positions)
        # The daily budget is spent across every slot of the day, not per
        # plan: a day with seven entry slots must not take max_new trades
        # in each of them.
        per_class = dict(opened_today_by_class or {})
        taken = int(opened_today)
        budget = self.max_new if entry_budget is None else int(entry_budget)
        for candidate, read, view in candidates:
            if taken >= budget:
                plan.rejected.append([candidate.symbol, "daily new-position cap reached"])
                continue
            klass = str(read.asset_class or "unknown")
            if self.max_new_per_class and per_class.get(klass, 0) >= self.max_new_per_class:
                plan.rejected.append(
                    [candidate.symbol, f"{klass} daily cap reached"]
                )
                continue

            limits = market_limits.get(candidate.symbol, {})
            size_mult = REGIME_SIZE.get(read.regime, 0.8)
            size_mult *= self.risk.contrarian_haircut(view.contrarian_share)
            order = self.risk.size_order(
                symbol=candidate.symbol,
                side=candidate.side,
                entry_price=read.price,
                atr=read.atr,
                equity=equity,
                risk_pct=risk_pct * size_mult,
                annualized_vol=read.annualized_vol,
                min_qty=limits.get("min_qty", 0.0),
                qty_step=limits.get("qty_step", 0.0),
                min_notional=limits.get("min_notional", 0.0),
                asset_class=read.asset_class,
            )

            # Costs first: a trade whose expected move does not clear the
            # round trip is negative-expectancy before the market moves.
            clears, cost_detail = self.risk.clears_costs(
                order, abs(candidate.edge), self._round_trip_bps(read)
            )
            if not clears:
                plan.rejected.append([
                    candidate.symbol,
                    f"edge {cost_detail['expected_bps']:.0f}bps vs "
                    f"{cost_detail['cost_bps']:.0f}bps costs "
                    f"({cost_detail['ratio']:.1f}x < {cost_detail['required']}x)",
                ])
                continue

            decision = self.risk.check_new_trade(
                order=order, positions=book, equity=equity,
                day_start_equity=day_start_equity, peak_equity=peak_equity,
                consecutive_losses=consecutive_losses,
                cooldown_until=cooldown_until, now=now,
            )
            if not decision:
                plan.rejected.append([candidate.symbol, decision.reason])
                if decision.reason in ("daily_loss_limit", "max_drawdown_halt",
                                       "daily_profit_lock", "cooldown"):
                    plan.notes.append(f"Halted planning: {decision.reason}")
                    break
                continue

            plan.trades.append(PlannedTrade(
                symbol=candidate.symbol, side=candidate.side, edge=candidate.edge,
                quantity=order.quantity, entry_price=order.entry_price,
                stop_price=order.stop_price, take_profit=order.take_profit,
                risk_usd=order.risk_usd, notional=order.notional,
                atr=order.atr, r_distance=order.r_distance,
                components=candidate.components, reason=candidate.reason,
                caps=order.caps_applied, strategy=self._dominant(view),
                asset_class=read.asset_class, venue=read.venue,
                timeframe=read.timeframe,
            ))
            book.append(_ProvisionalPosition(plan.trades[-1]))
            taken += 1
            per_class[klass] = per_class.get(klass, 0) + 1

        self._log(plan)
        return plan

    # ── Helpers ───────────────────────────────────────────────

    def _min_edge(self, read: SymbolRead) -> float:
        """The conviction an instrument's class demands before trading."""
        from bot.markets.instrument import trading_profile

        profile = trading_profile(read.asset_class, self.config)
        return float(profile.get("min_edge") or self.min_edge)

    @staticmethod
    def _can_short(read: SymbolRead) -> bool:
        """Whether a short is executable on this instrument.

        An unrecognised asset class is refused rather than allowed: the
        cost of skipping a tradable short is one missed trade, and the
        cost of booking an unexecutable one is a position that does not
        exist holding real risk budget.
        """
        from bot.markets.instrument import AssetClass

        if not read.asset_class:
            return False
        try:
            return AssetClass(str(read.asset_class)).can_short
        except ValueError:
            return False

    def _round_trip_bps(self, read: SymbolRead) -> float:
        """Modelled cost of opening and closing one position, in bps.

        Taken from the instrument itself, because the number differs by an
        order of magnitude across asset classes: a crypto taker round trip
        is ~17bps while a liquid US equity is ~6bps. Charging the crypto
        figure to a stock would veto trades that comfortably clear their
        real costs.
        """
        base = float(read.round_trip_bps) if read.round_trip_bps else None
        if base is None:
            paper = self.config.get("paper", {})
            base = (float(paper.get("fee_taker_bps", 5.5))
                    + float(paper.get("slippage_bps", 3.0))) * 2
        return base + float(read.spread_bps or 0.0)

    @staticmethod
    def _dominant(view: CombinedView) -> str:
        """Which strategy drove this position, for attribution."""
        if not view.contributors:
            return "blend"
        return max(view.contributors, key=lambda k: abs(view.contributors[k]))

    def _neutral_view(self, view: CombinedView) -> bool:
        """Market-neutral strategies are exempt from the trend veto.

        Cross-sectional momentum is supposed to short the weakest name in a
        rising market — that is the whole construction, not a mistake.
        """
        neutral = {s.name for s in self.strategies if getattr(s, "market_neutral", False)}
        if not neutral or not view.contributors:
            return False
        dominant = self._dominant(view)
        return dominant in neutral

    @staticmethod
    def htf_label(read: SymbolRead) -> str:
        return "higher-TF uptrend" if read.htf_trend > 0 else "higher-TF downtrend"

    @staticmethod
    def primary_label(read: SymbolRead) -> str:
        return "primary uptrend" if read.primary_trend > 0 else "primary downtrend"

    def _describe(self, view: CombinedView, read: SymbolRead, edge: float) -> str:
        drivers = sorted(view.contributors.items(), key=lambda kv: -abs(kv[1]))[:2]
        driver_text = ", ".join(f"{name} {value:+.2f}" for name, value in drivers)
        parts = [f"edge {edge:+.3f}", driver_text,
                 f"agree {view.agreement:+.2f}", read.regime]
        if abs(read.news_tilt) > 0.05:
            parts.append(f"news {read.news_tilt:+.2f}")
        return " | ".join(p for p in parts if p)

    def _log(self, plan: DayPlan) -> None:
        logger.info("─" * 62)
        logger.info("  DAY PLAN — %s | risk/trade %.2f%%%s", plan.day,
                    plan.risk_pct_per_trade,
                    f" | exposure {plan.exposure.get('scale', 1):.2f}x"
                    if plan.exposure else "")
        logger.info("─" * 62)
        if plan.signals:
            counts = ", ".join(f"{k}:{len(v)}" for k, v in sorted(plan.signals.items()))
            logger.info("  Signals: %s", counts)
        if not plan.trades:
            logger.info("  No trades planned.")
        for trade in plan.trades:
            logger.info(
                "  %-5s %-10s qty %.6f @ %.4f | stop %.4f | tp %.4f | risk $%.2f | %s",
                trade.side.upper(), trade.symbol, trade.quantity, trade.entry_price,
                trade.stop_price, trade.take_profit, trade.risk_usd, trade.reason,
            )
        for symbol, reason in plan.rejected[:10]:
            logger.info("    skip %-10s %s", symbol, reason)


class _ProvisionalPosition:
    """Stands in for a not-yet-opened position so the risk gates see the
    cumulative effect of the plan while it is being built."""

    __slots__ = ("symbol", "side", "entry_price", "quantity", "stop_price", "direction")

    def __init__(self, trade: PlannedTrade):
        self.symbol = trade.symbol
        self.side = trade.side
        self.entry_price = trade.entry_price
        self.quantity = trade.quantity
        self.stop_price = trade.stop_price
        self.direction = 1 if trade.side == "long" else -1
