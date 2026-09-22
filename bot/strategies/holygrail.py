"""Raschke and Connors' "Holy Grail" pullback, from *Street Smarts*.

    Filter   14-period ADX above 30 — a market already trending hard.
    Setup    Price retraces to the 20-period EMA.
    Entry    Above the trigger bar's extreme, in the trend's direction.
    Target   A retest of the most recent swing extreme.
    Stop     Beyond the swing point behind the entry.

The name was theirs and it was a joke about how simple the rules are, not
a claim about reliability. What it adds to this book is a *pullback*
entry: every other trend strategy here buys strength, which means they all
enter at the same moments and are effectively one bet. This one waits for
the retracement, so it takes the same view at a different price — and
sometimes does not get filled at all, which is itself a diversification.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class HolyGrailPullback(BaseStrategy):
    """Buy the pullback to the 20 EMA inside a strong trend."""

    name = "holygrail"
    stance = "confirmation"

    def __init__(self, config: dict):
        super().__init__(config)
        self.adx_period = int(self._param("adx_period", 14))
        self.adx_floor = float(self._param("adx_floor", 30.0))
        self.ema_period = int(self._param("ema_period", 20))
        self.touch_atr = float(self._param("touch_atr", 0.5))
        self.swing_lookback = int(self._param("swing_lookback", 20))
        self.max_bars_since_extreme = int(self._param("max_bars_since_extreme", 15))

    def required_bars(self) -> int:
        return max(self.ema_period, self.adx_period, self.swing_lookback) * 4 + 20

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            if df is None or len(df) < self.required_bars():
                continue

            adx_series, plus_di, minus_di = ind.adx(df, self.adx_period)
            adx = ind.last_value(adx_series, 0.0)
            if adx < self.adx_floor:
                continue  # not a strong enough trend to have a pullback in

            di_plus = ind.last_value(plus_di)
            di_minus = ind.last_value(minus_di)
            side = 1 if di_plus > di_minus else -1

            close = df["close"]
            ema = ind.last_value(ind.ema(close, self.ema_period))
            atr = ind.last_value(ind.atr(df, 14))
            if not ema or atr <= 0:
                continue

            price = float(close.iloc[-1])
            distance = abs(price - ema) / atr
            if distance > self.touch_atr:
                continue  # price has not come back to the average yet

            # The pullback must be *against* the trend, not a breakout that
            # happens to sit near the EMA.
            if side > 0 and price > ema + 0.1 * atr:
                continue
            if side < 0 and price < ema - 0.1 * atr:
                continue

            # There has to be a recent extreme to retest; otherwise this is
            # a range, not a pullback in a trend.
            recent = df.tail(self.swing_lookback)
            if side > 0:
                extreme = float(recent["high"].max())
                bars_since = len(recent) - 1 - int(np.argmax(recent["high"].to_numpy()))
                room = (extreme - price) / atr
            else:
                extreme = float(recent["low"].min())
                bars_since = len(recent) - 1 - int(np.argmin(recent["low"].to_numpy()))
                room = (price - extreme) / atr

            if bars_since > self.max_bars_since_extreme or room <= 0.3:
                continue

            trend_quality = float(np.clip((adx - self.adx_floor) / 25.0, 0.0, 1.0))
            proximity = float(np.clip(1.0 - distance / max(self.touch_atr, 1e-9), 0.0, 1.0))
            score = side * float(np.clip(0.40 + 0.30 * trend_quality + 0.30 * proximity,
                                         0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"Holy Grail: ADX {adx:.0f} pullback to EMA{self.ema_period} "
                       f"({distance:.2f} ATR away, {room:.1f} ATR to retest)",
                horizon_bars=self.swing_lookback,
                adx=round(adx, 1), distance_atr=round(distance, 3),
                target=round(extreme, 6), room_atr=round(room, 2),
            )
            if signal:
                signals.append(signal)

        return signals
