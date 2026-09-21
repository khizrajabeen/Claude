"""Antonacci's dual momentum, adapted to a crypto universe.

The rule has two gates and an asset only trades when both open:

    Relative momentum   it must be among the strongest in the universe
    Absolute momentum   its own trailing return must be positive

The second gate is the one that matters. Relative momentum alone always
holds *something*, which in a market-wide collapse means holding whatever
is falling least. Requiring positive absolute momentum moves the book to
cash instead. That is where the published drawdown reduction comes from —
roughly 23% maximum drawdown against 60% for the index over the same
period — not from better selection.

The crypto adaptation replaces the T-bill benchmark with a return floor:
there is no risk-free leg here, so "beat cash" becomes "be positive by
more than the cost of holding". When nothing clears both gates the
strategy returns no signals at all, which is the correct output and the
whole point of it being in the book.
"""

from __future__ import annotations

import numpy as np

from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class DualMomentum(BaseStrategy):
    """Relative strength, gated by absolute momentum."""

    name = "dualmom"
    market_neutral = False

    def __init__(self, config: dict):
        super().__init__(config)
        self.lookback = int(self._param("lookback_bars", 336))   # ~2 weeks on 1h
        self.top_n = int(self._param("top_n", 2))
        self.min_universe = int(self._param("min_universe", 3))
        # The hurdle the absolute gate must clear, in annualised percent.
        self.hurdle_annual_pct = float(self._param("hurdle_annual_pct", 4.0))
        self.allow_shorts = bool(self._param("allow_shorts", False))

    def required_bars(self) -> int:
        return self.lookback + 20

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        from bot.analysis.indicators import BARS_PER_YEAR

        bars_per_year = BARS_PER_YEAR.get(ctx.timeframe, 8_760)
        periods = bars_per_year / self.lookback
        hurdle = (1 + self.hurdle_annual_pct / 100) ** (1 / periods) - 1

        scored: dict[str, float] = {}
        for symbol in ctx.tradable():
            close = ctx.close(symbol)
            if close is None or len(close) < self.required_bars():
                continue
            start = float(close.iloc[-1 - self.lookback])
            if start <= 0:
                continue
            scored[symbol] = float(close.iloc[-1] / start - 1.0)

        if len(scored) < self.min_universe:
            return []

        ranked = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        leaders = ranked[: self.top_n]
        spread = float(np.std([v for _, v in ranked])) or 1.0

        signals: list[StrategySignal] = []
        for rank, (symbol, momentum) in enumerate(leaders, start=1):
            # The absolute gate. Failing it means cash, not the next name
            # down — that substitution is what the rule exists to prevent.
            if momentum <= hurdle:
                continue

            edge = (momentum - hurdle) / spread
            score = float(np.clip(0.45 + 0.55 * np.tanh(edge), 0.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason=f"dual momentum: rank {rank}/{len(ranked)}, "
                       f"{momentum * 100:+.1f}% over {self.lookback} bars "
                       f"(hurdle {hurdle * 100:+.2f}%)",
                horizon_bars=self.lookback // 2,
                rank=rank, momentum_pct=round(momentum * 100, 3),
                hurdle_pct=round(hurdle * 100, 4),
            )
            if signal:
                signals.append(signal)

        if self.allow_shorts:
            for rank, (symbol, momentum) in enumerate(reversed(ranked[-self.top_n:]), start=1):
                if momentum >= -hurdle:
                    continue
                edge = (abs(momentum) - hurdle) / spread
                score = -float(np.clip(0.45 + 0.55 * np.tanh(edge), 0.0, 1.0))
                signal = self.signal(
                    symbol, score,
                    reason=f"dual momentum short: bottom {rank}, "
                           f"{momentum * 100:+.1f}% over {self.lookback} bars",
                    horizon_bars=self.lookback // 2,
                    rank=len(ranked) - rank + 1, momentum_pct=round(momentum * 100, 3),
                )
                if signal:
                    signals.append(signal)

        if not signals:
            # Worth saying out loud: "no signal" here is a decision to hold
            # cash, not a failure to find one.
            pass
        return signals
