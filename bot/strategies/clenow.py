"""Clenow's trend model, from *Following the Trend*, as published.

    Trend filter   EMA(50) above EMA(100) for longs, below for shorts.
    Entry          Price at a 100-period extreme in the filter's direction.
    Stop           3 ATR(20) against the position.
    Sizing         Risk a fixed small fraction of equity per position, the
                   ATR normalising size across markets.

It is deliberately close to the trend strategy already here, and that is
the point of including it: two respected trend formulations with different
parameters are a check on whether the result depends on the specific
lookbacks or on trend following as such. If they diverge sharply, the
signal is a parameter artefact.

The wider stop is the substantive difference. Three ATR gives a trade much
more room than the two ATR used elsewhere, which means fewer stop-outs and
smaller positions for the same dollar risk — a different trade-off, not a
better one.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class ClenowTrend(BaseStrategy):
    """EMA-filtered breakout at a long-lookback extreme."""

    name = "clenow"

    def __init__(self, config: dict):
        super().__init__(config)
        self.fast_ema = int(self._param("fast_ema", 50))
        self.slow_ema = int(self._param("slow_ema", 100))
        self.breakout = int(self._param("breakout_bars", 100))
        self.atr_period = int(self._param("atr_period", 20))
        self.stop_atr = float(self._param("stop_atr", 3.0))
        self.max_extension_atr = float(self._param("max_extension_atr", 2.0))

    def required_bars(self) -> int:
        return max(self.slow_ema * 2, self.breakout) + self.atr_period + 10

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            if df is None or len(df) < self.required_bars():
                continue

            close = df["close"]
            fast = ind.last_value(ind.ema(close, self.fast_ema))
            slow = ind.last_value(ind.ema(close, self.slow_ema))
            if not fast or not slow:
                continue

            bias = 1 if fast > slow else -1
            price = float(close.iloc[-1])

            window = df.iloc[-self.breakout - 1 : -1]
            extreme_high = float(window["high"].max())
            extreme_low = float(window["low"].min())

            # The filter and the breakout must agree. A new high in a
            # downtrend is a bounce, not an entry.
            if bias > 0 and price > extreme_high:
                level = extreme_high
            elif bias < 0 and price < extreme_low:
                level = extreme_low
            else:
                continue

            atr = ind.last_value(ctx.indicator(symbol, "atr", period=self.atr_period))
            if atr <= 0:
                continue

            extension = abs(price - level) / atr
            if extension > self.max_extension_atr:
                continue

            # Conviction rises with how cleanly the EMAs are separated,
            # normalised by volatility so it compares across markets.
            separation = abs(fast - slow) / max(1e-9, atr)
            strength = float(np.tanh(separation / 3.0))
            freshness = float(np.clip(1.0 - extension / self.max_extension_atr, 0.0, 1.0))
            score = bias * float(np.clip(0.35 + 0.40 * strength + 0.25 * freshness, 0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"Clenow EMA{self.fast_ema}/{self.slow_ema} "
                       f"{'up' if bias > 0 else 'down'} + {self.breakout}-bar "
                       f"{'high' if bias > 0 else 'low'} ({extension:.2f} ATR past)",
                horizon_bars=self.breakout,
                separation_atr=round(separation, 3),
                extension_atr=round(extension, 3),
                stop_atr=self.stop_atr,
            )
            if signal:
                signals.append(signal)

        return signals
