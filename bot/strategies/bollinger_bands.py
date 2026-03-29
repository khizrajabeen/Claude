"""Bollinger Bands Strategy.

Generates BUY when price touches the lower band (oversold),
and SELL when price touches the upper band (overbought).
"""

import pandas as pd
import ta

from bot.strategies.base import BaseStrategy, Signal


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands mean-reversion strategy."""

    name = "Bollinger Bands"

    def __init__(self, config: dict):
        super().__init__(config)
        params = config.get("strategies", {}).get("bollinger_bands", {})
        self.period = params.get("period", 20)
        self.std_dev = params.get("std_dev", 2.0)

    def analyze(self, df: pd.DataFrame) -> Signal:
        if len(df) < self.period + 2:
            return Signal.HOLD

        bb = ta.volatility.BollingerBands(
            df["close"], window=self.period, window_dev=self.std_dev
        )
        upper = bb.bollinger_hband().iloc[-1]
        lower = bb.bollinger_lband().iloc[-1]
        close = df["close"].iloc[-1]

        if close <= lower:
            return Signal.BUY
        if close >= upper:
            return Signal.SELL
        return Signal.HOLD

    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        bb = ta.volatility.BollingerBands(
            df["close"], window=self.period, window_dev=self.std_dev
        )
        return {
            "BB_upper": round(bb.bollinger_hband().iloc[-1], 2),
            "BB_middle": round(bb.bollinger_mavg().iloc[-1], 2),
            "BB_lower": round(bb.bollinger_lband().iloc[-1], 2),
            "close": round(df["close"].iloc[-1], 2),
        }
