"""Short-term reversal — the other side of momentum, and its hedge.

Over days and weeks, returns trend. Over hours, they mean-revert: a sharp
move against no news tends to give part of itself back as liquidity
replenishes. That makes reversal the natural diversifier for a book full of
trend, because it makes money precisely in the chop that whipsaws trend
models.

It is also the fastest way to lose money in a real trend, so two guards
apply: the signal is suppressed when the higher timeframe is trending hard
(never catch a knife that is still falling), and it requires the move to be
statistically unusual rather than merely negative.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class ReversionStrategy(BaseStrategy):
    """Fade statistically stretched short-term moves in non-trending tape."""

    name = "reversion"

    def __init__(self, config: dict):
        super().__init__(config)
        self.window = int(self._param("zscore_window", 24))
        self.entry_z = float(self._param("entry_z", 1.6))
        self.max_adx = float(self._param("max_adx", 32.0))
        self.rsi_window = int(self._param("rsi_window", 14))
        self.require_volume = bool(self._param("require_volume_spike", False))

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            read = ctx.reads.get(symbol)
            if df is None or read is None or len(df) < self.window * 3:
                continue

            # A strong trend is the one environment where fading reliably
            # loses. Stand aside rather than size down.
            if read.adx > self.max_adx:
                continue

            z = ind.last_value(ind.zscore(df["close"], self.window))
            if abs(z) < self.entry_z:
                continue

            # Fade the move: a stretched-down price is a buy.
            side = -1 if z > 0 else 1

            rsi = ind.last_value(ind.rsi(df["close"], self.rsi_window), 50.0)
            rsi_agrees = (side == 1 and rsi < 40) or (side == -1 and rsi > 60)

            # Do not fade *into* the higher-timeframe trend.
            if read.htf_trend and side == -read.htf_trend:
                continue

            excess = (abs(z) - self.entry_z) / max(1.0, 3.0 - self.entry_z)
            score = side * float(np.clip(0.4 + 0.4 * excess + (0.2 if rsi_agrees else 0.0),
                                         0.0, 1.0))

            if self.require_volume:
                volume = df["volume"]
                spike = float(volume.iloc[-1] / max(1e-9, volume.tail(self.window).mean()))
                if spike < float(self._param("min_volume_spike", 1.3)):
                    continue

            signal = self.signal(
                symbol, score,
                reason=f"fade z={z:+.2f} (ADX {read.adx:.0f}, RSI {rsi:.0f})",
                horizon_bars=self.window,
                zscore=round(z, 3), rsi=round(rsi, 1),
            )
            if signal:
                signals.append(signal)

        return signals

    def required_bars(self) -> int:
        return self.window * 3 + 10
