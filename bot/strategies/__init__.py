"""Trading strategies."""

from bot.strategies.base import BaseStrategy, Signal
from bot.strategies.sma_crossover import SMACrossoverStrategy
from bot.strategies.rsi_strategy import RSIStrategy
from bot.strategies.bollinger_bands import BollingerBandsStrategy

STRATEGIES = {
    "sma_crossover": SMACrossoverStrategy,
    "rsi": RSIStrategy,
    "bollinger_bands": BollingerBandsStrategy,
}


def get_strategy(name: str, config: dict) -> BaseStrategy:
    """Factory to get a strategy by name."""
    if name not in STRATEGIES:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGIES.keys())}"
        )
    return STRATEGIES[name](config)
