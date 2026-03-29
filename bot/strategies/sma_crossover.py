"""Simple Moving Average Crossover Strategy.

Generates BUY when the fast SMA crosses above the slow SMA,
and SELL when the fast SMA crosses below the slow SMA.
"""

import pandas as pd
import ta

from bot.strategies.base import BaseStrategy, Signal


class SMACrossoverStrategy(BaseStrategy):
    """SMA crossover strategy."""

    name = "SMA Crossover"

    def __init__(self, config: dict):
        super().__init__(config)
        params = config.get("strategies", {}).get("sma_crossover", {})
        self.fast_period = params.get("fast_period", 10)
        self.slow_period = params.get("slow_period", 30)

    def analyze(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.slow_period + 2:
            return Signal.HOLD

        df = df.copy()
        df["sma_fast"] = ta.trend.sma_indicator(df["close"], window=self.fast_period)
        df["sma_slow"] = ta.trend.sma_indicator(df["close"], window=self.slow_period)

        prev_fast = df["sma_fast"].iloc[-2]
        prev_slow = df["sma_slow"].iloc[-2]
        curr_fast = df["sma_fast"].iloc[-1]
        curr_slow = df["sma_slow"].iloc[-1]

        # Bullish crossover
        if prev_fast <= prev_slow and curr_fast > curr_slow:
            return Signal.BUY

        # Bearish crossover
        if prev_fast >= prev_slow and curr_fast < curr_slow:
            return Signal.SELL

        return Signal.HOLD

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        df = df.copy()
        df["sma_fast"] = ta.trend.sma_indicator(df["close"], window=self.fast_period)
        df["sma_slow"] = ta.trend.sma_indicator(df["close"], window=self.slow_period)
        return {
            f"SMA({self.fast_period})": round(df["sma_fast"].iloc[-1], 2),
            f"SMA({self.slow_period})": round(df["sma_slow"].iloc[-1], 2),
            "close": round(df["close"].iloc[-1], 2),
        }
