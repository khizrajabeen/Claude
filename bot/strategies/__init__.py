"""Strategy registry.

Two families, kept because they earned their place on measured results
rather than on reputation:

  **selective** — `clenow`, `turtle`, `holygrail`. Published trend systems
  that trade rarely and were the only group to survive an out-of-sample
  split, ranking first in both halves of a 300-day test.

  **lux** — `supertrend`, `smc`, `nwenvelope`, `lorentzian`. Indicator-style
  signals of the kind LuxAlgo and its peers publish, implemented
  mechanically so the measurement decides rather than the marketing.

Six earlier strategies were removed after measuring poorly across repeated
benches: a house trend model, cross-sectional momentum, a Donchian
breakout, short-term reversion, funding carry and dual momentum. Their
removal is recorded here rather than silently, because "we tried it and it
did not work" is information the next person needs.
"""

from __future__ import annotations

import logging

from bot.strategies.base import BaseStrategy, MarketContext, Strategy, StrategySignal
from bot.strategies.clenow import ClenowTrend
from bot.strategies.holygrail import HolyGrailPullback
from bot.strategies.lorentzian import LorentzianClassifier
from bot.strategies.nwenvelope import NadarayaWatsonEnvelope
from bot.strategies.smc import SmartMoneyConcepts
from bot.strategies.supertrend_ai import SuperTrendAI
from bot.strategies.turtle import TurtleStrategy

logger = logging.getLogger("trading_bot")

REGISTRY: dict[str, type[BaseStrategy]] = {
    # Published systems, implemented to their stated rules.
    TurtleStrategy.name: TurtleStrategy,          # Dennis & Eckhardt, 1983
    ClenowTrend.name: ClenowTrend,                # Clenow, Following the Trend
    HolyGrailPullback.name: HolyGrailPullback,    # Raschke & Connors, Street Smarts
    # Indicator-style signals, measured rather than trusted.
    SuperTrendAI.name: SuperTrendAI,              # clustered SuperTrend factors
    SmartMoneyConcepts.name: SmartMoneyConcepts,  # structure, sweeps, imbalances
    NadarayaWatsonEnvelope.name: NadarayaWatsonEnvelope,  # kernel-regression fade
    LorentzianClassifier.name: LorentzianClassifier,      # kNN on market state
}

# The three published systems that survived the out-of-sample split.
SELECTIVE = ["clenow", "turtle", "holygrail"]
# Indicator-style signals in the LuxAlgo mould.
LUX = ["supertrend", "smc", "nwenvelope", "lorentzian"]

DEFAULT_ENABLED = SELECTIVE + LUX


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
    "TurtleStrategy", "ClenowTrend", "HolyGrailPullback",
    "SuperTrendAI", "SmartMoneyConcepts", "NadarayaWatsonEnvelope",
    "LorentzianClassifier",
    "REGISTRY", "DEFAULT_ENABLED", "SELECTIVE", "LUX", "build_strategies",
]
