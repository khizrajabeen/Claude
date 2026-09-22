"""Nadaraya-Watson envelope — LuxAlgo's kernel-regression mean reversion.

Fit a kernel regression through price, band it by the mean absolute
deviation of the residuals, and fade the touches: price closing beyond the
upper band is a short, beyond the lower band a long.

**The repainting trap, which is the whole reason this needs care.** The
default mode recalculates the fit across the entire window on every bar, so
the line at any past point keeps moving as new data arrives. Backtested
naively it looks extraordinary — the "band" it touched is one drawn with
knowledge of what came next. LuxAlgo ships a non-repainting endpoint mode
precisely for this, and only that mode is used here.

The honest version is choppier and signals far less often. It is also the
only one that could have been traded, and the look-ahead test in the suite
exists to keep it that way.

As a strategy this is a mean-reversion cousin of the existing `reversion`
entry, differing in how the centre line is estimated: a kernel fit follows
curvature that a rolling mean lags. The same guard applies — never fade a
market that is genuinely trending.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class NadarayaWatsonEnvelope(BaseStrategy):
    """Fade closes beyond a kernel-regression envelope."""

    name = "nwenvelope"
    stance = "contrarian"

    def __init__(self, config: dict):
        super().__init__(config)
        self.bandwidth = float(self._param("bandwidth", 8.0))
        self.window = int(self._param("window", 200))
        self.multiplier = float(self._param("multiplier", 3.0))
        self.max_adx = float(self._param("max_adx", 35.0))
        self.max_excess_atr = float(self._param("max_excess_atr", 3.0))

    def required_bars(self) -> int:
        return self.window + 60

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            read = ctx.reads.get(symbol)
            if df is None or read is None or len(df) < self.required_bars():
                continue

            # Fading a strong trend is how mean-reversion books die.
            if read.adx > self.max_adx:
                continue

            close = df["close"]
            fit = ind.gaussian_kernel_regression(close, self.bandwidth, self.window)
            centre = ind.last_value(fit)
            if not centre:
                continue

            residuals = (close - fit).dropna()
            if len(residuals) < self.window // 2:
                continue

            deviation = float(residuals.abs().mean())
            if deviation <= 0:
                continue

            band = self.multiplier * deviation
            price = float(close.iloc[-1])
            excess = price - centre

            if abs(excess) < band:
                continue  # inside the envelope, nothing to fade

            atr_now = ind.last_value(ind.atr(df, 14))
            if atr_now <= 0:
                continue

            # A move far beyond the band is more likely a regime change
            # than a stretched rubber band.
            excess_atr = (abs(excess) - band) / atr_now
            if excess_atr > self.max_excess_atr:
                continue

            side = -1 if excess > 0 else 1

            # Do not fade into the higher timeframe's direction.
            if read.htf_trend and side == -read.htf_trend:
                continue

            stretch = float(np.clip((abs(excess) / band - 1.0) * 2, 0.0, 1.0))
            calm = float(np.clip(1.0 - read.adx / self.max_adx, 0.0, 1.0))
            score = side * float(np.clip(0.40 + 0.35 * stretch + 0.25 * calm, 0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"NW envelope: {abs(excess) / band:.2f}x band "
                       f"{'above' if side < 0 else 'below'} the fit "
                       f"(ADX {read.adx:.0f})",
                horizon_bars=self.window // 8,
                centre=round(centre, 6), band=round(band, 6),
                excess_atr=round(excess_atr, 2),
            )
            if signal:
                signals.append(signal)

        return signals
