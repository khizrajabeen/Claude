"""Cross-sectional momentum — rank the universe, buy the top, sell the bottom.

Where time-series momentum asks "is this asset rising?", cross-sectional
momentum asks "is it rising *more than its peers*?". That difference makes
the two genuinely distinct: in a market where everything rallies together,
TSMOM is fully long and XS momentum is close to flat, because being up 10%
when the median is up 12% is a short.

The practical consequence is that XS momentum is roughly market-neutral by
construction. It survives broad drawdowns that flatten a directional book,
which is exactly the property worth paying for.

Returns are skipped over the most recent bars before ranking, the standard
12-1 construction: the very latest move tends to reverse, so including it
contaminates a momentum signal with short-term reversal.
"""

from __future__ import annotations

import numpy as np

from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class CrossSectionalMomentum(BaseStrategy):
    """Rank-based long/short momentum across the universe."""

    name = "xsmom"
    market_neutral = True

    def __init__(self, config: dict):
        super().__init__(config)
        self.lookback = int(self._param("lookback_bars", 168))
        self.skip = int(self._param("skip_bars", 12))
        self.min_universe = int(self._param("min_universe", 4))
        self.quantile = float(self._param("quantile", 0.34))
        self.vol_adjust = bool(self._param("vol_adjust", True))

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        scores: dict[str, float] = {}
        for symbol in ctx.tradable():
            close = ctx.close(symbol)
            if close is None or len(close) < self.lookback + self.skip + 2:
                continue

            # 12-1 style: measure to `skip` bars ago, not to right now.
            end = close.iloc[-1 - self.skip]
            start = close.iloc[-1 - self.skip - self.lookback]
            if start <= 0:
                continue
            momentum = float(np.log(end / start))

            if self.vol_adjust:
                returns = np.log(close / close.shift(1))
                vol = float(returns.rolling(self.lookback).std().iloc[-1])
                if np.isfinite(vol) and vol > 0:
                    momentum /= vol * np.sqrt(self.lookback)
            scores[symbol] = momentum

        # A ranking needs something to rank. With a handful of assets the
        # top and bottom "deciles" are one name each and the signal is just
        # noise with extra steps.
        if len(scores) < self.min_universe:
            return []

        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        n = len(ranked)
        take = max(1, int(round(n * self.quantile)))

        median = float(np.median([v for _, v in ranked]))
        spread = float(np.std([v for _, v in ranked])) or 1.0

        signals: list[StrategySignal] = []
        for rank, (symbol, value) in enumerate(ranked):
            if rank < take:
                side = 1
            elif rank >= n - take:
                side = -1
            else:
                continue  # the middle of the pack has no opinion

            # Conviction scales with how far from the pack the name sits,
            # so a narrow spread produces small positions on both legs.
            z = (value - median) / spread
            score = side * min(1.0, abs(np.tanh(z)))

            signal = self.signal(
                symbol, score,
                reason=f"XS rank {rank + 1}/{n} (z={z:+.2f} vs peers)",
                horizon_bars=self.lookback // 2,
                rank=rank + 1, universe=n, raw=round(value, 4),
            )
            if signal:
                signals.append(signal)

        return signals

    def required_bars(self) -> int:
        return self.lookback + self.skip + 10
