"""Lorentzian Classification — kNN over historical market states.

The most-used open-source machine-learning indicator on TradingView
(jdehorty), and genuinely different from everything else in this book:
where the trend systems apply a rule, this one asks "when the market last
looked like this, what happened next?" and lets the closest historical
neighbours vote.

The published construction, implemented as specified:

  Features      RSI(14), WaveTrend(10,11), CCI(20), ADX(20), RSI(9), each
                normalised on a trailing window
  Distance      Lorentzian, d = sum(log(1 + |x_i - y_i|)). The log matters:
                under Euclidean distance one volatility spike decides who
                counts as a neighbour
  Neighbours    the k closest, sampled every `spacing` bars so they cannot
                all come from one afternoon
  Vote          each neighbour votes the direction the market actually went
                over the following `horizon` bars
  Filters       volatility, regime and ADX gates before any signal is taken

Two things this implementation does that most do not. Every feature is
normalised on a *trailing* window rather than the whole series, because
rescaling against a future minimum is a leak that flatters the result. And
neighbours are drawn only from bars whose outcome had already resolved
`horizon` bars before the present, so no vote is cast by a bar that could
not yet have known its own answer.

Whether any of it works is what the out-of-sample bench is for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class LorentzianClassifier(BaseStrategy):
    """k-nearest-neighbour classification of the current market state."""

    name = "lorentzian"

    def __init__(self, config: dict):
        super().__init__(config)
        self.neighbours = int(self._param("neighbours", 8))
        self.horizon = int(self._param("horizon", 4))
        self.spacing = int(self._param("spacing", 4))
        self.history = int(self._param("history_bars", 500))
        self.normalise_window = int(self._param("normalise_window", 200))
        self.min_vote_ratio = float(self._param("min_vote_ratio", 0.6))
        self.use_volatility_filter = bool(self._param("volatility_filter", True))
        self.use_regime_filter = bool(self._param("regime_filter", True))
        self.adx_floor = float(self._param("adx_floor", 20.0))

    def required_bars(self) -> int:
        return self.history + self.normalise_window + self.horizon + 20

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            read = ctx.reads.get(symbol)
            if df is None or read is None or len(df) < self.required_bars():
                continue

            features = self._features(df)
            if features is None:
                continue

            prediction = self._classify(df, features)
            if prediction is None:
                continue

            votes, total, neighbours = prediction
            if total == 0:
                continue

            ratio = abs(votes) / total
            if ratio < self.min_vote_ratio:
                continue  # the neighbourhood disagrees; that is information

            side = 1 if votes > 0 else -1

            blocked = self._filters(df, read, side)
            if blocked:
                continue

            # Conviction is how lopsided the vote was, not how many
            # neighbours there were: eight unanimous neighbours say more
            # than sixteen split five ways.
            score = side * float(np.clip(0.35 + 0.65 * (ratio - self.min_vote_ratio)
                                         / max(1e-9, 1 - self.min_vote_ratio), 0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"Lorentzian kNN: {int(abs(votes))}/{total} of the closest "
                       f"historical states went {'up' if side > 0 else 'down'} "
                       f"over the next {self.horizon} bars",
                horizon_bars=self.horizon,
                vote_ratio=round(ratio, 3),
                neighbours=total,
                mean_distance=round(float(np.mean(neighbours)), 4) if neighbours else 0.0,
            )
            if signal:
                signals.append(signal)

        return signals

    # ── Features ──────────────────────────────────────────────

    def _features(self, df: pd.DataFrame) -> np.ndarray | None:
        """The published feature set, each normalised on a trailing window."""
        close = df["close"]
        window = self.normalise_window

        columns = [
            ind.normalise(ind.rsi(close, 14), window),
            ind.normalise(ind.wave_trend(df, 10, 11), window),
            ind.normalise(ind.cci(df, 20), window),
            ind.normalise(ind.adx(df, 20)[0], window),
            ind.normalise(ind.rsi(close, 9), window),
        ]
        matrix = pd.concat(columns, axis=1).to_numpy(dtype=float)
        if not np.isfinite(matrix[-1]).all():
            return None
        return matrix

    # ── Classification ────────────────────────────────────────

    def _classify(self, df: pd.DataFrame, features: np.ndarray):
        """Find the closest resolved historical states and count their votes."""
        close = df["close"].to_numpy(dtype=float)
        n = len(close)

        # A neighbour may only vote if its own outcome had resolved by the
        # time we are standing. Anything later is a bar voting on a future
        # it could not have seen.
        last_resolved = n - 1 - self.horizon
        first = max(self.normalise_window, last_resolved - self.history)
        if last_resolved <= first:
            return None

        # Chronological spacing: consecutive bars are near-identical, so an
        # unspaced kNN returns eight views of the same afternoon.
        candidates = np.arange(first, last_resolved + 1, self.spacing)
        if len(candidates) < self.neighbours * 2:
            return None

        history = features[candidates]
        finite = np.isfinite(history).all(axis=1)
        candidates, history = candidates[finite], history[finite]
        if len(candidates) < self.neighbours * 2:
            return None

        distances = ind.lorentzian_distance(features[-1], history)
        closest = np.argsort(distances)[: self.neighbours]

        votes = 0
        chosen_distances = []
        for index in closest:
            bar = int(candidates[index])
            future = close[bar + self.horizon]
            if not np.isfinite(future) or close[bar] <= 0:
                continue
            votes += 1 if future > close[bar] else -1
            chosen_distances.append(float(distances[index]))

        return votes, len(chosen_distances), chosen_distances

    # ── Filters ───────────────────────────────────────────────

    def _filters(self, df: pd.DataFrame, read, side: int) -> str | None:
        """The published gates. Returns a reason when the signal is blocked."""
        if self.use_adx_filter and read.adx < self.adx_floor:
            return "adx"

        if self.use_volatility_filter:
            # Recent range against its own longer-run average: a market that
            # has gone dead gives the classifier nothing to work with.
            true_range = ind.true_range(df)
            recent = float(true_range.tail(14).mean())
            baseline = float(true_range.tail(100).mean())
            if baseline > 0 and recent / baseline < 0.5:
                return "volatility"

        if self.use_regime_filter and read.htf_trend and side == -read.htf_trend:
            return "regime"

        return None

    @property
    def use_adx_filter(self) -> bool:
        return self.adx_floor > 0
