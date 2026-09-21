"""How much risk each strategy gets, and how signals combine per symbol.

Two ideas, both from the risk-parity literature:

  * **Inverse volatility.** A strategy whose returns swing twice as hard
    gets half the risk budget, so no single driver dominates the book's
    variance just because it happens to be noisy. Equal *capital* across
    strategies is not equal *risk*, and it is risk that produces drawdown.
  * **A correlation haircut.** Equal-risk weighting still over-allocates to
    a cluster of strategies that all say the same thing. Each strategy's
    weight is divided by how correlated it is with the rest, which pushes
    capital toward the genuinely independent drivers — the ones that make
    the diversification real rather than nominal.

Until there is enough realised history to measure either, everything is
weighted equally. Estimating a covariance matrix from a fortnight of data
produces confident nonsense.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from bot.strategies.base import StrategySignal

logger = logging.getLogger("trading_bot")


@dataclass
class CombinedView:
    """Every strategy's opinion on one symbol, resolved into one position."""

    symbol: str
    direction: int
    conviction: float                    # 0..1, after weighting
    contributors: dict = field(default_factory=dict)   # strategy -> signed score
    agreement: float = 0.0               # -1 fully opposed, +1 unanimous
    coverage: float = 0.0                # share of strategy weight with a view
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "conviction": round(self.conviction, 4),
            "agreement": round(self.agreement, 4),
            "coverage": round(self.coverage, 4),
            "contributors": {k: round(v, 4) for k, v in self.contributors.items()},
            "reasons": self.reasons,
        }


class StrategyAllocator:
    """Weights strategies, then merges their signals per symbol."""

    def __init__(self, config: dict):
        self.config = config
        self.portfolio = config.get("portfolio", {})
        self.method = str(self.portfolio.get("weighting", "inverse_vol"))
        self.correlation_haircut = bool(self.portfolio.get("correlation_haircut", True))
        self.max_weight = float(self.portfolio.get("max_strategy_weight", 0.40))
        self.min_weight = float(self.portfolio.get("min_strategy_weight", 0.05))
        # Per-strategy caps. Inverse volatility over-funds strategies whose
        # risk is in the tail rather than in daily variance — carry is the
        # textbook case: it looks almost riskless right up until it doesn't.
        self.caps = dict(self.portfolio.get("max_weights", {}) or {})
        self.manual = self.portfolio.get("weights", {}) or {}
        self.require_agreement = float(self.portfolio.get("min_agreement", 0.0))
        self._warned_infeasible = False
        # How much a lone strategy keeps of its own conviction. At 1.0
        # breadth is ignored entirely; at 0 only unanimity counts.
        self.breadth_floor = float(self.portfolio.get("breadth_floor", 0.6))

    # ── Strategy weights ──────────────────────────────────────

    def weights(self, strategies: list[str],
                volatilities: dict[str, float] | None = None,
                correlations: pd.DataFrame | None = None) -> dict[str, float]:
        """Risk budget per strategy, summing to 1."""
        if not strategies:
            return {}

        if self.method == "manual" and self.manual:
            raw = {s: float(self.manual.get(s, 0.0)) for s in strategies}
            return self._normalise(raw, strategies)

        if self.method == "equal" or not volatilities:
            raw = {s: 1.0 for s in strategies}
            return self._normalise(raw, strategies)

        # Inverse volatility. A strategy with no measured history gets the
        # median vol, so it is neither punished nor favoured for being new.
        measured = [v for v in volatilities.values() if v > 0]
        fallback = float(np.median(measured)) if measured else 1.0

        raw = {}
        for name in strategies:
            vol = volatilities.get(name, fallback) or fallback
            raw[name] = 1.0 / vol

        if self.correlation_haircut and correlations is not None and not correlations.empty:
            raw = self._apply_haircut(raw, correlations, strategies)

        return self._normalise(raw, strategies)

    def _apply_haircut(self, raw: dict[str, float], corr: pd.DataFrame,
                       strategies: list[str]) -> dict[str, float]:
        """Shrink weights for strategies that duplicate the others."""
        adjusted = dict(raw)
        for name in strategies:
            if name not in corr.columns:
                continue
            others = [s for s in strategies if s != name and s in corr.columns]
            if not others:
                continue
            # Average absolute correlation with the rest of the book. A
            # strategy correlated 0.9 with everything is nearly a duplicate
            # and should not be funded as though it were independent.
            crowding = float(np.mean([abs(corr.loc[name, other]) for other in others]))
            adjusted[name] = raw[name] * (1.0 - 0.5 * crowding)
        return adjusted

    def _normalise(self, raw: dict[str, float], strategies: list[str]) -> dict[str, float]:
        """Clamp to the per-strategy bounds and renormalise to 1."""
        count = len(strategies)
        values = {s: max(0.0, float(raw.get(s, 0.0))) for s in strategies}
        total = sum(values.values())
        if total <= 0:
            equal = 1.0 / count
            return {s: equal for s in strategies}

        weights = {s: v / total for s, v in values.items()}

        # A global cap below 1/n cannot be satisfied by any allocation, and
        # clamping toward it just flattens everything to equal weights —
        # silently discarding the ranking it was given. Disable it instead
        # and say so, rather than pretending it applied.
        global_max = self.max_weight
        if count * global_max <= 1.0 + 1e-9:
            if not self._warned_infeasible:
                logger.warning(
                    "portfolio.max_strategy_weight (%.2f) cannot bind across %d "
                    "strategies — ignoring it; it would force equal weights",
                    global_max, count,
                )
                self._warned_infeasible = True
            global_max = 1.0

        def ceiling(name: str) -> float:
            cap = float(self.caps.get(name, global_max))
            # A per-strategy cap still has to leave room for the rest.
            return max(min(global_max, cap), 1e-6)

        floor = min(self.min_weight, 1.0 / count)

        # Clamping then renormalising can push another weight back over its
        # cap, so iterate rather than assuming one pass settles it.
        for _ in range(16):
            clamped = {s: min(ceiling(s), max(floor, w)) for s, w in weights.items()}
            total = sum(clamped.values())
            if total <= 0:
                break
            weights = {s: w / total for s, w in clamped.items()}
            if all(w <= ceiling(s) + 1e-9 for s, w in weights.items()):
                break
        return weights

    # ── Signal combination ────────────────────────────────────

    def combine(self, signals_by_strategy: dict[str, list[StrategySignal]],
                weights: dict[str, float]) -> dict[str, CombinedView]:
        """Merge per-strategy signals into one view per symbol.

        Contributions are summed with their strategy weight, so two
        strategies disagreeing cancel rather than producing two opposed
        positions in the same name.
        """
        by_symbol: dict[str, dict[str, StrategySignal]] = {}
        for name, signals in signals_by_strategy.items():
            for signal in signals:
                by_symbol.setdefault(signal.symbol, {})[name] = signal

        total_weight = sum(w for w in weights.values() if w > 0) or 1.0

        views: dict[str, CombinedView] = {}
        for symbol, per_strategy in by_symbol.items():
            contributors, reasons = {}, []
            net = 0.0
            gross = 0.0
            present_weight = 0.0
            for name, signal in per_strategy.items():
                weight = weights.get(name, 0.0)
                if weight <= 0:
                    continue
                contribution = weight * signal.signed
                contributors[name] = contribution
                net += contribution
                gross += weight * signal.strength
                present_weight += weight
                reasons.append(f"{name}: {signal.reason}")

            if gross <= 0 or present_weight <= 0:
                continue

            # Agreement is net over gross: +1 when every strategy points the
            # same way, near 0 when they cancel out.
            agreement = net / gross

            # Conviction has to satisfy three things at once, and the two
            # obvious formulas each fail one of them:
            #
            #   summing weighted contributions makes conviction depend on
            #   how many strategies happen to be enabled — with five, one
            #   firing alone scores a fifth of its own strength, so a fixed
            #   entry threshold means something different in every roster;
            #
            #   dividing by the weight actually present fixes that but
            #   makes weight irrelevant — worse, a strategy benched to a 5%
            #   probe then produces *more* conviction than a trusted one,
            #   because the small denominator inflates the mean. That
            #   defeats both risk parity and the autopilot.
            #
            # So: take the mean opinion, then scale it by how much weight
            # stands behind it relative to an equal share. A trusted
            # strategy firing alone keeps its conviction; a probe does not.
            mean_opinion = abs(net) / present_weight
            equal_share = total_weight / max(1, len(weights))
            standing = min(1.0, present_weight / equal_share) if equal_share > 0 else 0.0

            # Breadth is a bounded bonus on top, so unanimity beats a lone
            # voice without a lone voice being discounted to nothing.
            coverage = present_weight / total_weight
            breadth = self.breadth_floor + (1 - self.breadth_floor) * coverage
            conviction = min(1.0, mean_opinion * standing * breadth)

            if abs(agreement) < self.require_agreement:
                continue

            views[symbol] = CombinedView(
                symbol=symbol,
                direction=1 if net > 0 else -1,
                conviction=conviction,
                contributors=contributors,
                agreement=agreement,
                coverage=round(coverage, 4),
                reasons=reasons,
            )

        return views

    def log_weights(self, weights: dict[str, float],
                    volatilities: dict[str, float] | None = None) -> None:
        if not weights:
            return
        parts = []
        for name, weight in sorted(weights.items(), key=lambda kv: -kv[1]):
            vol = (volatilities or {}).get(name)
            parts.append(f"{name} {weight * 100:.0f}%" + (f" (vol {vol * 100:.2f}%)" if vol else ""))
        logger.info("  Strategy budget: %s", " | ".join(parts))
