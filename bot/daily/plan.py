"""Turning the morning briefing into a concrete plan for the day.

The edge score is a regime-weighted blend of four views:

    trend        EMA structure and ADX/DI agreement, confirmed on the
                 higher timeframe
    momentum     rate of change and MACD histogram
    mean-revert  z-score and RSI stretch against the recent range
    news         sentiment tilt, scaled by consensus

Weights shift with the regime: a trending tape leans on trend and momentum,
a ranging tape on mean reversion. That matters because the same z-score of
-2 is a buy in a range and a falling knife in a downtrend.

News is only ever a tilt. Headline sentiment is a weak standalone predictor
and decays within hours, so it can tip a marginal setup or veto one that
fights the tape, but it cannot open a trade by itself.

Every candidate is then sized by the risk budget (fixed dollar risk at an
ATR stop) and run through the portfolio gates before it reaches the plan.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from bot.daily.briefing import Briefing, SymbolRead
from bot.risk.budget import RiskBudget, SizedOrder

logger = logging.getLogger("trading_bot")

# Regime -> (trend, momentum, mean_reversion) weights. News is added on top
# at its configured weight, then the whole thing is renormalised.
REGIME_WEIGHTS = {
    "trending_up":   (0.50, 0.30, -0.20),
    "trending_down": (0.50, 0.30, -0.20),
    "ranging":       (0.15, 0.10, 0.75),
    "volatile":      (0.30, 0.20, 0.50),
    "quiet":         (0.40, 0.35, 0.25),
}

# Position-size multiplier by regime — smaller when the tape is hostile.
REGIME_SIZE = {
    "trending_up": 1.0, "trending_down": 1.0, "ranging": 0.8,
    "volatile": 0.5, "quiet": 0.7,
}


@dataclass
class Candidate:
    """A scored trade idea, before sizing."""

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

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DayPlan:
    """The plan for one trading day."""

    day: str
    created_at: str
    trades: list = field(default_factory=list)          # PlannedTrade
    rejected: list = field(default_factory=list)        # (symbol, reason)
    considered: list = field(default_factory=list)      # Candidate
    risk_pct_per_trade: float = 0.0
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "day": self.day,
            "created_at": self.created_at,
            "risk_pct_per_trade": round(self.risk_pct_per_trade, 4),
            "trades": [t.to_dict() for t in self.trades],
            "considered": [c.to_dict() for c in self.considered],
            "rejected": self.rejected,
            "notes": self.notes,
        }


class DayPlanner:
    """Scores, ranks and sizes the day's trades."""

    def __init__(self, config: dict, risk: RiskBudget):
        self.config = config
        self.risk = risk
        signals = config.get("signals", {})
        self.min_edge = float(signals.get("min_edge_score", 0.18))
        self.news_weight = float(signals.get("news_weight", 0.20))
        self.book_weight = float(signals.get("book_weight", 0.08))
        self.require_htf = bool(signals.get("require_htf_agreement", True))
        self.veto_on_news = bool(signals.get("news_can_veto", True))
        self.max_new = int(config.get("session", {}).get("max_new_positions_per_day", 3))

    # ── Scoring ───────────────────────────────────────────────

    def score(self, read: SymbolRead, market_tone: float = 0.0) -> Candidate:
        """Blend the four views into a signed edge in roughly [-1, 1]."""
        trend = self._trend_score(read)
        momentum = self._momentum_score(read)
        mean_rev = self._mean_reversion_score(read)
        news = read.news_tilt

        w_trend, w_mom, w_mr = REGIME_WEIGHTS.get(read.regime, REGIME_WEIGHTS["ranging"])

        # A regime call we do not believe pulls the weights toward neutral.
        confidence = max(0.3, min(1.0, read.regime_confidence))
        w_trend *= confidence
        w_mom *= confidence

        # Resting-liquidity imbalance: a fast-decaying confirmation, given
        # a deliberately small weight because top-of-book depth is easy to
        # spoof and turns over in seconds.
        book = max(-1.0, min(1.0, read.book_imbalance * 2))

        raw = w_trend * trend + w_mom * momentum + w_mr * mean_rev
        raw += self.news_weight * news
        raw += self.book_weight * book
        raw += 0.05 * market_tone  # a light whole-market thumb on the scale

        total_weight = (abs(w_trend) + abs(w_mom) + abs(w_mr)
                        + self.news_weight + self.book_weight + 0.05)
        edge = raw / total_weight if total_weight else 0.0
        edge = max(-1.0, min(1.0, edge))

        components = {
            "trend": round(trend, 4),
            "momentum": round(momentum, 4),
            "mean_reversion": round(mean_rev, 4),
            "news": round(news, 4),
            "book": round(book, 4),
            "market_tone": round(market_tone, 4),
            "weights": {"trend": round(w_trend, 3), "momentum": round(w_mom, 3),
                        "mean_reversion": round(w_mr, 3), "news": self.news_weight},
        }

        side = "long" if edge > 0 else "short"
        reason = self._describe(read, components, edge)
        return Candidate(symbol=read.symbol, side=side, edge=round(edge, 4),
                         components=components, reason=reason)

    def _trend_score(self, read: SymbolRead) -> float:
        """EMA structure gated by ADX, in [-1, 1]."""
        if not read.ema_slow:
            return 0.0
        separation = (read.ema_fast - read.ema_slow) / read.ema_slow
        # Normalise separation by the asset's own volatility so a 1% gap
        # means something different in BTC than in a 200%-vol altcoin.
        vol_unit = max(1e-6, read.atr_pct / 100)
        direction = max(-1.0, min(1.0, separation / (vol_unit * 3)))

        # ADX below 20 means no trend worth trading; above 40 it is mature.
        strength = max(0.0, min(1.0, (read.adx - 18) / 22))

        di_agree = 0.0
        if read.plus_di or read.minus_di:
            total = read.plus_di + read.minus_di
            if total > 0:
                di_agree = (read.plus_di - read.minus_di) / total

        score = 0.6 * direction * strength + 0.4 * di_agree * strength
        return max(-1.0, min(1.0, score))

    def _momentum_score(self, read: SymbolRead) -> float:
        macd = 0.0
        if read.price and read.atr:
            # MACD histogram in ATR units — comparable across assets.
            macd = max(-1.0, min(1.0, read.macd_hist / (read.atr * 0.8)))
        overnight = max(-1.0, min(1.0, read.overnight_return_pct / max(0.5, read.atr_pct * 2)))
        return max(-1.0, min(1.0, 0.6 * macd + 0.4 * overnight))

    def _mean_reversion_score(self, read: SymbolRead) -> float:
        """Positive when price is stretched *down* — a fade-the-move buy."""
        z = max(-3.0, min(3.0, read.zscore))
        z_score = -z / 2.0

        rsi_score = 0.0
        if read.rsi >= 70:
            rsi_score = -(read.rsi - 70) / 30
        elif read.rsi <= 30:
            rsi_score = (30 - read.rsi) / 30

        # Donchian position: 1 at the range high (fade), 0 at the low (buy).
        range_score = (0.5 - read.donchian) * 2 if read.donchian is not None else 0.0

        return max(-1.0, min(1.0, 0.45 * z_score + 0.35 * rsi_score + 0.20 * range_score))

    def _describe(self, read: SymbolRead, components: dict, edge: float) -> str:
        parts = [f"{read.regime}", f"ADX {read.adx:.0f}", f"RSI {read.rsi:.0f}"]
        dominant = max(
            ("trend", "momentum", "mean_reversion"),
            key=lambda k: abs(components[k] * components["weights"][k]),
        )
        parts.append(f"driver={dominant}")
        if abs(components["news"]) > 0.05:
            parts.append(f"news {components['news']:+.2f}")
        return f"edge {edge:+.3f} | " + " | ".join(parts)

    # ── Plan construction ─────────────────────────────────────

    def build(
        self,
        briefing: Briefing,
        equity: float,
        open_positions: list,
        peak_equity: float,
        day_start_equity: float,
        market_limits: dict | None = None,
        consecutive_losses: int = 0,
        cooldown_until: datetime | None = None,
        now: datetime | None = None,
    ) -> DayPlan:
        """Score every symbol, rank them, size the best and gate them."""
        now = now or datetime.now(timezone.utc)
        market_limits = market_limits or {}

        drawdown_pct = ((peak_equity - equity) / peak_equity * 100) if peak_equity > 0 else 0.0
        risk_pct = self.risk.risk_per_trade_pct(
            rolling=briefing.rolling_stats,
            drawdown_pct=drawdown_pct,
            consecutive_losses=consecutive_losses,
        )

        plan = DayPlan(day=briefing.day, created_at=now.isoformat(),
                       risk_pct_per_trade=risk_pct)

        tone = float(briefing.market_tone.get("score", 0.0))
        candidates: list[tuple[Candidate, SymbolRead]] = []

        for symbol, read in briefing.symbols.items():
            if not read.tradable:
                plan.rejected.append([symbol, read.skip_reason or "not tradable"])
                continue

            candidate = self.score(read, tone)
            plan.considered.append(candidate)

            if abs(candidate.edge) < self.min_edge:
                plan.rejected.append([symbol, f"edge {candidate.edge:+.3f} below {self.min_edge}"])
                continue

            # Higher-timeframe veto: do not fight the bigger trend.
            if self.require_htf and read.htf_trend:
                wants_long = candidate.side == "long"
                if (wants_long and read.htf_trend < 0) or (not wants_long and read.htf_trend > 0):
                    plan.rejected.append([symbol, f"{candidate.side} against {self.htf_label(read)}"])
                    continue

            # News veto: a strong tilt against the setup kills it.
            if self.veto_on_news and read.news_tilt:
                against = (candidate.side == "long" and read.news_tilt < -0.3) or \
                          (candidate.side == "short" and read.news_tilt > 0.3)
                if against:
                    plan.rejected.append([symbol, f"news {read.news_tilt:+.2f} opposes {candidate.side}"])
                    continue

            candidates.append((candidate, read))

        # Rank by conviction, then by liquidity as the tie-break: the same
        # edge is worth more where it can be executed cheaply.
        candidates.sort(
            key=lambda pair: (abs(pair[0].edge), pair[1].quote_volume_24h),
            reverse=True,
        )

        # Positions are added one at a time against a running book so heat
        # and correlation limits see each prior fill.
        book = list(open_positions)
        for candidate, read in candidates:
            if len(plan.trades) >= self.max_new:
                plan.rejected.append([candidate.symbol, "daily new-position cap reached"])
                continue

            limits = market_limits.get(candidate.symbol, {})
            size_mult = REGIME_SIZE.get(read.regime, 0.8)
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
            )

            decision = self.risk.check_new_trade(
                order=order,
                positions=book,
                equity=equity,
                day_start_equity=day_start_equity,
                peak_equity=peak_equity,
                consecutive_losses=consecutive_losses,
                cooldown_until=cooldown_until,
                now=now,
            )
            if not decision:
                plan.rejected.append([candidate.symbol, decision.reason])
                if decision.reason in ("daily_loss_limit", "max_drawdown_halt",
                                       "daily_profit_lock", "cooldown"):
                    plan.notes.append(f"Halted planning: {decision.reason}")
                    break
                continue

            planned = PlannedTrade(
                symbol=candidate.symbol,
                side=candidate.side,
                edge=candidate.edge,
                quantity=order.quantity,
                entry_price=order.entry_price,
                stop_price=order.stop_price,
                take_profit=order.take_profit,
                risk_usd=order.risk_usd,
                notional=order.notional,
                atr=order.atr,
                r_distance=order.r_distance,
                components=candidate.components,
                reason=candidate.reason,
                caps=order.caps_applied,
            )
            plan.trades.append(planned)
            book.append(_ProvisionalPosition(planned))

        self._log(plan)
        return plan

    @staticmethod
    def htf_label(read: SymbolRead) -> str:
        return "higher-TF uptrend" if read.htf_trend > 0 else "higher-TF downtrend"

    def _log(self, plan: DayPlan) -> None:
        logger.info("─" * 62)
        logger.info("  DAY PLAN — %s | risk/trade %.2f%%", plan.day, plan.risk_pct_per_trade)
        logger.info("─" * 62)
        if not plan.trades:
            logger.info("  No trades planned. Rejections:")
        for t in plan.trades:
            logger.info(
                "  %-5s %-10s qty %.6f @ %.4f | stop %.4f | tp %.4f | risk $%.2f | %s",
                t.side.upper(), t.symbol, t.quantity, t.entry_price,
                t.stop_price, t.take_profit, t.risk_usd, t.reason,
            )
        for symbol, reason in plan.rejected[:12]:
            logger.info("    skip %-10s %s", symbol, reason)


class _ProvisionalPosition:
    """Stands in for a not-yet-opened position so the risk gates can see
    the cumulative effect of the plan while it is being built."""

    __slots__ = ("symbol", "side", "entry_price", "quantity", "stop_price", "direction")

    def __init__(self, trade: PlannedTrade):
        self.symbol = trade.symbol
        self.side = trade.side
        self.entry_price = trade.entry_price
        self.quantity = trade.quantity
        self.stop_price = trade.stop_price
        self.direction = 1 if trade.side == "long" else -1
