"""RSI (Relative Strength Index) Strategy.

Generates BUY when RSI drops below the oversold level,
and SELL when RSI rises above the overbought level.
"""

import pandas as pd
import ta

from bot.strategies.base import BaseStrategy, Signal


class RSIStrategy(BaseStrategy):
    """RSI mean-reversion strategy."""

    name = "RSI"

    def __init__(self, config: dict):
        super().__init__(config)
        params = config.get("strategies", {}).get("rsi", {})
        self.period = params.get("period", 14)
        self.overbought = params.get("overbought", 70)
        self.oversold = params.get("oversold", 30)

    def analyze(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.period + 2:
            return Signal.HOLD

        rsi = ta.momentum.rsi(df["close"], window=self.period)
        current_rsi = rsi.iloc[-1]

        if current_rsi <= self.oversold:
            return Signal.BUY
        if current_rsi >= self.overbought:
            return Signal.SELL
        return Signal.HOLD

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        rsi = ta.momentum.rsi(df["close"], window=self.period)
        return {
            "RSI": round(rsi.iloc[-1], 2),
            "overbought": self.overbought,
            "oversold": self.oversold,
            "close": round(df["close"].iloc[-1], 2),
        }
