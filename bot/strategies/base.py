"""Base strategy interface."""

from abc import ABC, abstractmethod
from enum import Enum

import pandas as pd


class Signal(Enum):
    """Trading signal."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    def __init__(self, config: dict):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Signal:
        """Analyze candle data and return a trading signal.

        Args:
            df: OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            Signal: BUY, SELL, or HOLD.
        """

    @abstractmethod
    def get_indicator_values(self, df: pd.DataFrame) -> dict:
        """Return current indicator values for logging."""
