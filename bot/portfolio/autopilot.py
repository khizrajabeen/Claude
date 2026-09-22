"""Self-managing strategy roster.

The bot decides which strategies to fund from what they have actually
earned, so nothing has to be tuned by hand. Three inputs, in order of
authority:

  1. **Realised expectancy.** A strategy with a materially negative record
     over a real sample is benched. Not deleted — benched, with a small
     probe allocation kept alive so it can earn its way back. A strategy
     that is merely unlucky looks identical to one that is broken, and the
     only way to tell them apart is to keep measuring.
  2. **Regime fit.** Trend systems and reversal systems do not fail at
     random; they fail in each other's weather. A multiplier tilts the
     roster toward whichever suits the tape, bounded so it can never zero
     anything out on regime alone.
  3. **Risk parity.** The surviving weights go through the usual
     inverse-volatility and correlation machinery.

The obvious failure mode of any such scheme is overfitting: chase whatever
worked last month and you buy the top of every strategy's cycle. Three
guards against that — a minimum sample before anything is judged, a
threshold well below zero rather than at it, and the probe allocation that
makes benching reversible.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("trading_bot")

# Which strategies suit which tape. Values multiply a strategy's weight;
# 1.0 is neutral. Deliberately mild — this is a tilt, not a switch.
REGIME_FIT: dict[str, dict[str, float]] = {
    "trending_up":   {"trend": 1.30, "turtle": 1.25, "clenow": 1.25, "dualmom": 1.25,
                      "breakout": 1.15, "holygrail": 1.15, "xsmom": 1.00,
                      "reversion": 0.55, "carry": 0.90},
    "trending_down": {"trend": 1.25, "turtle": 1.20, "clenow": 1.20, "dualmom": 0.70,
                      "breakout": 1.10, "holygrail": 1.10, "xsmom": 1.05,
                      "reversion": 0.55, "carry": 0.90},
    "ranging":       {"trend": 0.65, "turtle": 0.70, "clenow": 0.70, "dualmom": 0.70,
                      "breakout": 0.85, "holygrail": 0.80, "xsmom": 1.15,
                      "reversion": 1.40, "carry": 1.15},
    "volatile":      {"trend": 0.85, "turtle": 0.85, "clenow": 0.90, "dualmom": 0.80,
                      "breakout": 1.10, "holygrail": 0.80, "xsmom": 0.95,
                      "reversion": 0.75, "carry": 0.80},
    "quiet":         {"trend": 0.85, "turtle": 0.90, "clenow": 0.90, "dualmom": 0.95,
                      "breakout": 1.30, "holygrail": 0.90, "xsmom": 1.00,
                      "reversion": 1.05, "carry": 1.10},
}


@dataclass
class RosterDecision:
    """What the autopilot decided, and why."""

    active: list = field(default_factory=list)
    benched: dict = field(default_factory=dict)     # strategy -> reason
    regime: str = ""
    multipliers: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "active": self.active,
            "benched": self.benched,
            "regime": self.regime,
            "multipliers": {k: round(v, 3) for k, v in self.multipliers.items()},
            "notes": self.notes,
        }


class AutoPilot:
    """Chooses the strategy roster and tilts its weights, unattended."""

    def __init__(self, config: dict):
        self.config = config
        auto = config.get("autopilot", {})
        self.enabled = bool(auto.get("enabled", True))
        self.min_trades = int(auto.get("min_trades_to_judge", 25))
        self.bench_expectancy_r = float(auto.get("bench_below_expectancy_r", -0.15))
        self.probe_weight = float(auto.get("probe_weight", 0.05))
        self.regime_tilt = bool(auto.get("regime_tilt", True))
        self.max_benched_fraction = float(auto.get("max_benched_fraction", 0.5))

    # ── Roster ────────────────────────────────────────────────

    def decide(self, strategies: list[str], per_strategy: dict,
               regime: str = "") -> RosterDecision:
        """Which strategies trade today, and how their weights are tilted.

        `per_strategy` maps a strategy name to its record — at minimum
        `trades` and `avg_r`, as the journal reports them.
        """
        decision = RosterDecision(active=list(strategies), regime=regime)
        if not self.enabled:
            decision.notes.append("autopilot disabled — every strategy funded equally")
            return decision

        judged = []
        for name in strategies:
            record = per_strategy.get(name) or {}
            trades = int(record.get("trades", 0))
            expectancy = float(record.get("avg_r", 0.0))

            if trades < self.min_trades:
                # Too early to judge. Saying so beats guessing.
                continue
            if expectancy < self.bench_expectancy_r:
                judged.append((name, trades, expectancy))

        # Never bench more than half the book at once. If most strategies
        # are losing, the problem is the market or the risk settings, and
        # concentrating into the survivors makes it worse.
        limit = max(1, int(len(strategies) * self.max_benched_fraction))
        judged.sort(key=lambda item: item[2])
        for name, trades, expectancy in judged[:limit]:
            decision.active.remove(name)
            decision.benched[name] = (
                f"{expectancy:+.2f}R over {trades} trades, below "
                f"{self.bench_expectancy_r:+.2f}R"
            )

        if len(judged) > limit:
            decision.notes.append(
                f"{len(judged)} strategies are below the bench threshold but only "
                f"{limit} were benched — when most are losing, the setting or the "
                f"market is the problem, not the roster"
            )

        if self.regime_tilt and regime:
            decision.multipliers = self.regime_multipliers(decision.active, regime)

        if decision.benched:
            logger.info(
                "  Autopilot benched: %s",
                "; ".join(f"{k} ({v})" for k, v in decision.benched.items()),
            )
        for note in decision.notes:
            logger.info("  Autopilot: %s", note)
        return decision

    def regime_multipliers(self, strategies: list[str], regime: str) -> dict[str, float]:
        """Weight tilt for the current tape."""
        table = REGIME_FIT.get(regime)
        if not table:
            return {name: 1.0 for name in strategies}
        return {name: float(table.get(name, 1.0)) for name in strategies}

    # ── Weight adjustment ─────────────────────────────────────

    def apply(self, weights: dict[str, float], decision: RosterDecision
              ) -> dict[str, float]:
        """Fold the roster decision into risk-parity weights.

        A benched strategy keeps a probe allocation rather than going to
        zero, because a strategy that is switched off produces no record
        and can therefore never be switched back on.
        """
        if not weights:
            return weights

        adjusted: dict[str, float] = {}
        for name, weight in weights.items():
            if name in decision.benched:
                adjusted[name] = self.probe_weight * weight
                continue
            adjusted[name] = weight * decision.multipliers.get(name, 1.0)

        total = sum(adjusted.values())
        if total <= 0:
            equal = 1.0 / len(weights)
            return {name: equal for name in weights}
        return {name: value / total for name, value in adjusted.items()}

    def explain(self, weights: dict[str, float], decision: RosterDecision) -> str:
        parts = []
        for name, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
            tag = " (probe)" if name in decision.benched else ""
            parts.append(f"{name} {weight * 100:.0f}%{tag}")
        return " | ".join(parts)
