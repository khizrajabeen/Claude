"""Time-series momentum — the managed-futures workhorse.

Each instrument is judged against its own past, not against its peers: hold
it long if its own trailing return is positive, short if negative. This is
the signal behind most CTA programs, and the reason they tend to make money
in exactly the months everything else loses it — a persistent move has to
happen for a crash to be a crash, and trend followers are positioned for
persistent moves.

Two implementation details that matter more than the entry rule:

  * **Multiple horizons.** A single lookback is a single bet on how fast
    the market mean-reverts. Fast and slow lookbacks are blended, which
    both diversifies that choice and blunts the whipsaw a lone fast signal
    suffers in chop.
  * **Volatility scaling.** The raw signal is the trailing return divided
    by trailing volatility, so a 5% move in a calm asset counts for more
    than a 5% move in a wild one and the signal is comparable across the
    universe.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class TrendStrategy(BaseStrategy):
    """Volatility-scaled time-series momentum across several horizons."""

    name = "trend"

    def __init__(self, config: dict):
        super().__init__(config)
        # Horizons in bars. Defaults are roughly 1 day / 3 days / 2 weeks on
        # an hourly timeframe — fast, medium and slow.
        self.horizons = list(self._param("horizons", [24, 72, 336]))
        self.horizon_weights = list(self._param("horizon_weights", [0.25, 0.40, 0.35]))
        self.vol_window = int(self._param("vol_window", 72))
        self.adx_floor = float(self._param("adx_floor", 18.0))
        self.squash = float(self._param("squash", 1.5))

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        weights = np.array(self.horizon_weights[: len(self.horizons)], dtype=float)
        if weights.sum() <= 0:
            weights = np.ones(len(self.horizons))
        weights = weights / weights.sum()

        for symbol in ctx.tradable():
            close = ctx.close(symbol)
            if close is None or len(close) < max(self.horizons) + self.vol_window:
                continue

            returns = np.log(close / close.shift(1))
            vol = float(returns.rolling(self.vol_window).std().iloc[-1])
            if not np.isfinite(vol) or vol <= 0:
                continue

            # Each horizon contributes its own vol-scaled trailing return.
            scores, detail = [], {}
            for horizon in self.horizons:
                if len(close) <= horizon:
                    scores.append(0.0)
                    continue
                trailing = float(np.log(close.iloc[-1] / close.iloc[-1 - horizon]))
                # Divide by the volatility the move *should* have had over
                # that many bars, which makes horizons comparable.
                expected = vol * np.sqrt(horizon)
                z = trailing / expected if expected > 0 else 0.0
                scores.append(z)
                detail[f"z_{horizon}"] = round(z, 3)

            blended = float(np.dot(weights[: len(scores)], np.array(scores)))
            # tanh keeps a violent move from dominating the book; past a
            # point, more trend is not more information.
            score = float(np.tanh(blended / self.squash))

            # A trend signal in a market with no trend is noise. ADX below
            # the floor scales the conviction down rather than vetoing, so
            # the allocator still sees a weak opinion instead of silence.
            read = ctx.reads.get(symbol)
            if read is not None and read.adx < self.adx_floor:
                score *= max(0.25, read.adx / self.adx_floor)

            signal = self.signal(
                symbol, score,
                reason=f"TSMOM {'/'.join(str(h) for h in self.horizons)} "
                       f"blended z={blended:+.2f}",
                horizon_bars=int(np.median(self.horizons)),
                **detail,
            )
            if signal:
                signals.append(signal)

        return signals

    def required_bars(self) -> int:
        return max(self.horizons) + self.vol_window + 10
