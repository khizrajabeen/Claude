"""Strategy registry.

Each entry is a distinct return driver. Adding one here and enabling it in
config is all it takes for the allocator to start funding it.
"""

from __future__ import annotations

import logging

from bot.strategies.base import BaseStrategy, MarketContext, Strategy, StrategySignal
from bot.strategies.breakout import BreakoutStrategy
from bot.strategies.carry import CarryStrategy
from bot.strategies.clenow import ClenowTrend
from bot.strategies.dualmom import DualMomentum
from bot.strategies.holygrail import HolyGrailPullback
from bot.strategies.nwenvelope import NadarayaWatsonEnvelope
from bot.strategies.reversion import ReversionStrategy
from bot.strategies.smc import SmartMoneyConcepts
from bot.strategies.supertrend_ai import SuperTrendAI
from bot.strategies.trend import TrendStrategy
from bot.strategies.turtle import TurtleStrategy
from bot.strategies.xsmom import CrossSectionalMomentum

logger = logging.getLogger("trading_bot")

REGISTRY: dict[str, type[BaseStrategy]] = {
    # Built here
    TrendStrategy.name: TrendStrategy,
    CrossSectionalMomentum.name: CrossSectionalMomentum,
    BreakoutStrategy.name: BreakoutStrategy,
    ReversionStrategy.name: ReversionStrategy,
    CarryStrategy.name: CarryStrategy,
    # Published systems, implemented to their stated rules so the
    # comparison is against the real thing rather than a paraphrase.
    TurtleStrategy.name: TurtleStrategy,          # Dennis & Eckhardt, 1983
    ClenowTrend.name: ClenowTrend,                # Clenow, Following the Trend
    HolyGrailPullback.name: HolyGrailPullback,    # Raschke & Connors, Street Smarts
    DualMomentum.name: DualMomentum,              # Antonacci, Dual Momentum Investing
    # LuxAlgo-style indicators, implemented mechanically and measured
    # rather than trusted — see each module for what the evidence says.
    SuperTrendAI.name: SuperTrendAI,              # clustered SuperTrend factors
    SmartMoneyConcepts.name: SmartMoneyConcepts,  # structure, sweeps, imbalances
    NadarayaWatsonEnvelope.name: NadarayaWatsonEnvelope,  # kernel-regression fade
}

DEFAULT_ENABLED = [
    "trend", "xsmom", "breakout", "reversion", "carry",
    "turtle", "clenow", "holygrail", "dualmom",
    "supertrend", "smc", "nwenvelope",
]

# The three published systems that showed positive expectancy on the first
# 150-day bench. Selected *from* that sample, so they need out-of-sample
# testing before the selection means anything — see the bench's OOS mode.
SELECTIVE = ["clenow", "turtle", "holygrail"]
LUXALGO = ["supertrend", "smc", "nwenvelope"]


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
    "TurtleStrategy", "ClenowTrend", "HolyGrailPullback", "DualMomentum",
    "SuperTrendAI", "SmartMoneyConcepts", "NadarayaWatsonEnvelope",
    "SELECTIVE", "LUXALGO",
    "REGISTRY", "DEFAULT_ENABLED", "build_strategies",
]
