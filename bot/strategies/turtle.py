"""The original Turtle system (Dennis and Eckhardt, 1983), as published.

The rules fit on a page and are implemented here without improvement, so
that any comparison is against the real thing rather than a modernised
version of it:

  Entry     System 1: break of the 20-period Donchian channel.
            System 2: break of the 55-period channel, always taken.
  Filter    Skip an S1 breakout if the *previous* S1 breakout in that
            market would have won — even if it was never taken. After a
            winner the next breakout was judged less likely to run. The
            55-period failsafe is exempt, so no major trend is missed.
  N         The 20-period ATR. Everything is measured in N.
  Stop      2N from entry.
  Pyramid   Add a unit every 0.5N in favour, to a maximum of 4 units, each
            addition moving every stop to 2N below the newest entry.
  Exit      Opposite 10-period channel for S1, 20-period for S2.

Two honest notes. First, the Turtles traded a broad futures portfolio with
long holding periods; a 1h crypto timeframe is not the environment the
rules were written for, and the comparison should be read that way.
Second, pyramiding is expressed here as conviction rather than as literal
unit-stacking, because this bot holds one position per symbol and sizes it
through the risk layer. The signal is faithful; the plumbing differs.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class TurtleStrategy(BaseStrategy):
    """Donchian breakout with the original filter and N-based conviction."""

    name = "turtle"

    def __init__(self, config: dict):
        super().__init__(config)
        self.entry_s1 = int(self._param("entry_s1", 20))
        self.entry_s2 = int(self._param("entry_s2", 55))
        self.exit_s1 = int(self._param("exit_s1", 10))
        self.n_period = int(self._param("n_period", 20))
        self.stop_n = float(self._param("stop_n", 2.0))
        self.pyramid_step_n = float(self._param("pyramid_step_n", 0.5))
        self.max_units = int(self._param("max_units", 4))
        self.use_last_winner_filter = bool(self._param("last_winner_filter", True))

    def required_bars(self) -> int:
        # The filter needs history of previous breakouts, not just one channel.
        return self.entry_s2 * 6 + self.n_period + 10

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            if df is None or len(df) < self.required_bars():
                continue

            n = ind.last_value(ind.atr(df, self.n_period))
            if n <= 0:
                continue

            close = float(df["close"].iloc[-1])
            high, low = df["high"], df["low"]

            # Channels exclude the current bar; otherwise price is always
            # inside its own extreme and nothing ever breaks out.
            s1_high = float(high.iloc[-self.entry_s1 - 1 : -1].max())
            s1_low = float(low.iloc[-self.entry_s1 - 1 : -1].min())
            s2_high = float(high.iloc[-self.entry_s2 - 1 : -1].max())
            s2_low = float(low.iloc[-self.entry_s2 - 1 : -1].min())

            system, side, level = None, 0, 0.0
            if close > s2_high:
                system, side, level = "S2", 1, s2_high
            elif close < s2_low:
                system, side, level = "S2", -1, s2_low
            elif close > s1_high:
                system, side, level = "S1", 1, s1_high
            elif close < s1_low:
                system, side, level = "S1", -1, s1_low

            if system is None:
                continue

            skipped = False
            if system == "S1" and self.use_last_winner_filter:
                if self._last_breakout_won(df, side):
                    # The original rules skip it. The 55-period failsafe
                    # above is what stops a big trend being missed entirely.
                    skipped = True

            if skipped:
                continue

            # How far the move has already travelled, in N. The Turtles
            # added a unit every half-N, so progress through the pyramid is
            # the natural expression of conviction.
            advance = abs(close - level) / n
            units = min(self.max_units, 1 + int(advance / self.pyramid_step_n))

            # Do not chase: past the full pyramid the entry is worse for
            # the same 2N stop.
            if advance > self.max_units * self.pyramid_step_n:
                continue

            base = 0.55 if system == "S2" else 0.40
            score = side * float(np.clip(base + 0.12 * (units - 1), 0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"Turtle {system} {'high' if side > 0 else 'low'} break "
                       f"({advance:.2f}N past {level:.4f}, unit {units}/{self.max_units})",
                horizon_bars=self.entry_s2,
                system=system, n=round(n, 6), units=units,
                stop_distance_n=self.stop_n, advance_n=round(advance, 3),
            )
            if signal:
                signals.append(signal)

        return signals

    def _last_breakout_won(self, df, side: int) -> bool:
        """Would the previous S1 breakout in this direction have won?

        A breakout 'wins' if price reached 2N in favour before the opposite
        10-period channel stopped it out. Walked forward from the breakout
        bar, which is what the original rule describes.
        """
        high, low, close = df["high"], df["low"], df["close"]
        atr = ind.atr(df, self.n_period)
        window = min(len(df) - 2, self.entry_s2 * 5)

        for offset in range(2, window):
            i = len(df) - offset
            if i <= self.entry_s1 + self.n_period:
                break

            channel_high = float(high.iloc[i - self.entry_s1 : i].max())
            channel_low = float(low.iloc[i - self.entry_s1 : i].min())
            price = float(close.iloc[i])

            broke = (side > 0 and price > channel_high) or (side < 0 and price < channel_low)
            if not broke:
                continue

            n = float(atr.iloc[i])
            if not np.isfinite(n) or n <= 0:
                return False

            target = price + side * self.stop_n * n
            for j in range(i + 1, len(df)):
                bar_high, bar_low = float(high.iloc[j]), float(low.iloc[j])
                if side > 0:
                    if bar_high >= target:
                        return True
                    if bar_low <= float(low.iloc[max(0, j - self.exit_s1):j].min()):
                        return False
                else:
                    if bar_low <= target:
                        return True
                    if bar_high >= float(high.iloc[max(0, j - self.exit_s1):j].max()):
                        return False
            return False  # still open at the end of the data
        return False  # no prior breakout found
