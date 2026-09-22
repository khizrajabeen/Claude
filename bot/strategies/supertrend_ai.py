"""SuperTrend AI — LuxAlgo's clustering approach, implemented and measured.

The published idea: run SuperTrend at many multiplier settings at once,
score how each has actually performed, then use k-means to split those
settings into below-average, average and exceptional groups — and trade the
centroid of the best group. Instead of guessing a multiplier, the market
picks it.

The appeal is real: the right SuperTrend factor for a calm range is not the
right one for a violent trend, and this adapts without anyone touching a
setting. The risk is equally real, and is the reason this is measured
rather than trusted: choosing the setting that has worked best lately is
performance-chasing, which is exactly how a parameter sweep turns into
overfitting. The scoring window is deliberately long, and every choice is
made only on bars that had already printed.

K-means on a one-dimensional score is a sort and a few passes, so this is
implemented directly rather than pulling in scikit-learn — the daily
session should not need the ML extras installed to run.
"""

from __future__ import annotations

import logging

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal

logger = logging.getLogger("trading_bot")


class SuperTrendAI(BaseStrategy):
    """SuperTrend whose multiplier is chosen by clustering past performance."""

    name = "supertrend"

    def __init__(self, config: dict):
        super().__init__(config)
        self.atr_period = int(self._param("atr_period", 10))
        self.min_factor = float(self._param("min_factor", 1.0))
        self.max_factor = float(self._param("max_factor", 5.0))
        self.factor_step = float(self._param("factor_step", 0.5))
        self.performance_window = int(self._param("performance_window", 200))
        self.cluster_iterations = int(self._param("cluster_iterations", 20))
        self.min_signal_bars = int(self._param("min_signal_bars", 2))

    def required_bars(self) -> int:
        return self.performance_window + self.atr_period * 4 + 50

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        factors = np.arange(self.min_factor, self.max_factor + 1e-9, self.factor_step)
        if len(factors) < 3:
            logger.warning("SuperTrend AI needs at least three factors to cluster")
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            if df is None or len(df) < self.required_bars():
                continue

            returns = df["close"].pct_change().fillna(0.0).to_numpy(dtype=float)
            window = min(self.performance_window, len(df) - 1)

            performances, directions, lines = [], [], []
            for factor in factors:
                line, direction = ind.supertrend(df, self.atr_period, float(factor))
                signal = direction.to_numpy(dtype=float)
                # What this setting would have earned: its position on each
                # bar times the next bar's return. Shifted so the position
                # is the one it held *before* the move.
                earned = np.nansum(signal[-window - 1 : -1] * returns[-window:])
                performances.append(float(earned))
                directions.append(int(direction.iloc[-1]))
                lines.append(float(line.iloc[-1]))

            scores = np.array(performances, dtype=float)
            if not np.isfinite(scores).all():
                continue

            best_group = self._best_cluster(scores)
            if not best_group:
                continue

            # The chosen direction is the consensus of the best-performing
            # group; a split group means no signal, which is the point of
            # clustering rather than just taking the single best factor.
            group_directions = [directions[i] for i in best_group]
            consensus = float(np.mean(group_directions))
            if abs(consensus) < 0.6:
                continue

            side = 1 if consensus > 0 else -1
            chosen_factor = float(np.mean([factors[i] for i in best_group]))
            chosen_line = float(np.mean([lines[i] for i in best_group]))

            price = float(df["close"].iloc[-1])
            atr_now = ind.last_value(ind.atr(df, self.atr_period))
            if atr_now <= 0:
                continue

            # How far price sits beyond the trailing stop, in ATR. A fresh
            # flip is worth more than one that ran days ago.
            distance = abs(price - chosen_line) / atr_now
            freshness = float(np.clip(1.0 - distance / 4.0, 0.0, 1.0))

            # Separation between the best group and the rest: when every
            # setting performs alike, the clustering has found nothing.
            others = [s for i, s in enumerate(scores) if i not in best_group]
            spread = float(np.std(scores)) or 1.0
            edge = (float(np.mean(scores[best_group])) - float(np.mean(others))) / spread \
                if others else 0.0
            conviction = float(np.clip(np.tanh(edge), 0.0, 1.0))

            score = side * float(np.clip(0.30 + 0.40 * conviction + 0.30 * freshness,
                                         0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"SuperTrend AI: best cluster factor {chosen_factor:.1f} "
                       f"({len(best_group)}/{len(factors)} settings), "
                       f"{distance:.1f} ATR past the stop",
                horizon_bars=self.performance_window // 4,
                factor=round(chosen_factor, 2),
                cluster_size=len(best_group),
                cluster_edge=round(edge, 3),
                distance_atr=round(distance, 2),
            )
            if signal:
                signals.append(signal)

        return signals

    def _best_cluster(self, scores: np.ndarray) -> list[int]:
        """One-dimensional k-means into three groups; returns the best one.

        Centroids are seeded at the quartiles, which makes the result
        deterministic — a randomly seeded k-means would give a different
        answer on each run and turn the backtest into a lottery.
        """
        if len(scores) < 3:
            return list(range(len(scores)))

        centroids = np.percentile(scores, [25, 50, 75]).astype(float)
        assignment = np.zeros(len(scores), dtype=int)

        for _ in range(self.cluster_iterations):
            distances = np.abs(scores[:, None] - centroids[None, :])
            new_assignment = distances.argmin(axis=1)
            if np.array_equal(new_assignment, assignment):
                break
            assignment = new_assignment
            for k in range(3):
                members = scores[assignment == k]
                if len(members):
                    centroids[k] = float(members.mean())

        best = int(np.argmax(centroids))
        members = [i for i, k in enumerate(assignment) if k == best]
        return members or list(range(len(scores)))
