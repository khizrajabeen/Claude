"""Strategy registry.

Each entry is a distinct return driver. Adding one here and enabling it in
config is all it takes for the allocator to start funding it.
"""

from __future__ import annotations

import logging

from bot.strategies.base import BaseStrategy, MarketContext, Strategy, StrategySignal
from bot.strategies.breakout import BreakoutStrategy
from bot.strategies.carry import CarryStrategy
from bot.strategies.reversion import ReversionStrategy
from bot.strategies.trend import TrendStrategy
from bot.strategies.xsmom import CrossSectionalMomentum

logger = logging.getLogger("trading_bot")

REGISTRY: dict[str, type[BaseStrategy]] = {
    TrendStrategy.name: TrendStrategy,
    CrossSectionalMomentum.name: CrossSectionalMomentum,
    BreakoutStrategy.name: BreakoutStrategy,
    ReversionStrategy.name: ReversionStrategy,
    CarryStrategy.name: CarryStrategy,
}

DEFAULT_ENABLED = ["trend", "xsmom", "breakout", "reversion", "carry"]


def build_strategies(config: dict) -> list[BaseStrategy]:
    """Instantiate the strategies this config asks for."""
    requested = config.get("strategies", {}).get("enabled", DEFAULT_ENABLED)

    built: list[BaseStrategy] = []
    for name in requested:
        cls = REGISTRY.get(name)
        if cls is None:
            logger.warning("Unknown strategy '%s' — known: %s",
                           name, ", ".join(sorted(REGISTRY)))
            continue
        strategy = cls(config)
        if strategy.enabled:
            built.append(strategy)

    if not built:
        logger.warning("No strategies enabled — the bot will not trade")
    return built


__all__ = [
    "BaseStrategy", "MarketContext", "Strategy", "StrategySignal",
    "BreakoutStrategy", "CarryStrategy", "ReversionStrategy",
    "TrendStrategy", "CrossSectionalMomentum",
    "REGISTRY", "DEFAULT_ENABLED", "build_strategies",
]
