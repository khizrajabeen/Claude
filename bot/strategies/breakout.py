"""Donchian breakout — the Turtle rule, kept because it fails differently.

A breakout enters when price clears its own N-bar extreme. That overlaps
with trend following, but not completely: a breakout system is flat during
a steady grind that never makes a new high, and it is already positioned
when a range finally snaps. Blending it with a moving-average trend model
smooths the entry timing of the combined book.

Breakouts are filtered by volatility compression. A break out of a quiet
range has historically been worth more than a break out of an already
violent one, which is usually just noise clearing a level it will cross
back over within the hour.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class BreakoutStrategy(BaseStrategy):
    """Channel breakout with a volatility-compression filter."""

    name = "breakout"

    def __init__(self, config: dict):
        super().__init__(config)
        self.channel = int(self._param("channel_bars", 55))
        self.confirm_bars = int(self._param("confirm_bars", 2))
        self.compression_window = int(self._param("compression_window", 120))
        self.require_compression = bool(self._param("require_compression", True))
        self.max_extension_atr = float(self._param("max_extension_atr", 1.5))

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            if df is None or len(df) < max(self.channel, self.compression_window) + 5:
                continue

            close = float(df["close"].iloc[-1])
            # The channel excludes the current bar, otherwise price is
            # trivially inside its own high/low and nothing ever breaks.
            window = df.iloc[-self.channel - 1 : -1]
            upper = float(window["high"].max())
            lower = float(window["low"].min())
            if not np.isfinite(upper) or not np.isfinite(lower) or upper <= lower:
                continue

            atr = ind.last_value(ind.atr(df, 14))
            if atr <= 0:
                continue

            if close > upper:
                side, level, distance = 1, upper, (close - upper) / atr
            elif close < lower:
                side, level, distance = -1, lower, (lower - close) / atr
            else:
                continue

            # Already far past the level means the move happened without
            # us; entering here buys the same stop distance for a worse
            # price.
            if distance > self.max_extension_atr:
                continue

            # Volatility compression: is the recent range tight relative to
            # its own history?
            ratio = self._compression(df)
            if self.require_compression and ratio > float(self._param("max_ratio", 1.1)):
                continue

            # Conviction: strongest on a clean break out of a tight range.
            tightness = float(np.clip(1.2 - ratio, 0.0, 1.0))
            freshness = float(np.clip(1.0 - distance / max(self.max_extension_atr, 1e-9), 0.0, 1.0))
            score = side * float(np.clip(0.35 + 0.35 * tightness + 0.30 * freshness, 0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"{self.channel}-bar {'high' if side > 0 else 'low'} break "
                       f"({distance:.2f} ATR past {level:.4f}, compression {ratio:.2f})",
                horizon_bars=self.channel,
                level=round(level, 6), compression=round(ratio, 3),
                extension_atr=round(distance, 3),
            )
            if signal:
                signals.append(signal)

        return signals

    def _compression(self, df) -> float:
        """Recent true range against its longer-run average. Below 1 is a
        market that has gone quiet."""
        tr = ind.true_range(df)
        recent = float(tr.tail(self.confirm_bars * 10).mean())
        baseline = float(tr.tail(self.compression_window).mean())
        if not np.isfinite(baseline) or baseline <= 0:
            return 1.0
        return recent / baseline

    def required_bars(self) -> int:
        return max(self.channel, self.compression_window) + 10
