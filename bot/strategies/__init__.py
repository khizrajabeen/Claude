"""Strategy registry.

The roster was finalised on a solo bench — each strategy run alone over
the same 90 days of crypto history, on one shared download, so the only
thing differing between runs was the strategy:

    strategy      return%  trades  expect R  maxDD%   win%     PF
    clenow         +14.91      39    +0.581    1.33   69.2   5.58
    turtle         +10.77      69    +0.255    1.29   60.9   2.09
    lorentzian      +3.80      50    -0.002    4.02   48.0   1.37
    supertrend      -1.17     214    -0.070    5.58   42.1   1.01
    nwenvelope      -3.23      19    -0.308    3.25   31.6   0.20
    holygrail       -3.26      55    -0.206    5.14   30.9   0.76
    smc             -6.45     162    -0.321    9.40   30.9   0.65

**Enabled by default**: `clenow`, `turtle`.

The two published trend systems carry the book, and do it at roughly a
fifth of the drawdown the full roster suffered. The four dropped ones
traded about 450 times between them to lose money; `smc` and `supertrend`
alone accounted for 376 of those trades, which is the same lesson every
measurement in this project has returned — trading less is the only
durable edge found so far.

`lorentzian` was kept briefly and then dropped. Run together the three
rosters compared as:

    roster           return%  trades  expect R  maxDD%   win%    PF  Sharpe
    clenow+turtle     +15.66      64    +0.400    1.23   65.6  2.98    9.00
    +lorentzian       +16.20      90    +0.283    1.77   57.8  2.29    6.17
    old seven          +3.58     144    -0.039    4.55   43.1  1.22    1.75

It buys 0.54 points of return and costs 0.54 points of drawdown, 26
trades, $64 of fees and a third of the Sharpe. Adding a third driver for
diversification is a good instinct, but one whose per-trade expectancy is
zero diversifies nothing except the fee bill.

Read that Sharpe of 9.00 with suspicion rather than pleasure. It is not a
number a real edge produces over 90 days; it is what a lucky window looks
like, and the roster was chosen by looking at that window.

Everything dropped stays in the registry and keeps its tests. They are
still reachable by name in config and still have bench variants, because
"we measured it and it lost money" is a fact about one 90-day window on
one universe, not a proof — and deleting the code would make that
finding unrepeatable.

Six earlier strategies were removed outright after measuring poorly
across repeated benches: a house trend model, cross-sectional momentum, a
Donchian breakout, short-term reversion, funding carry and dual momentum.

  **selective** — `clenow`, `turtle`, `holygrail`. Published trend systems.
  **lux** — `supertrend`, `smc`, `nwenvelope`, `lorentzian`. Indicator-style
    signals of the kind LuxAlgo and its peers publish, implemented
    mechanically so the measurement decides rather than the marketing.
  **news** — `news`. Per-company sentiment for equities. Not in the default
    roster because the equity leg is off; it costs nothing to re-enable
    alongside the instruments.
"""

from __future__ import annotations

import logging

from bot.strategies.base import BaseStrategy, MarketContext, Strategy, StrategySignal
from bot.strategies.clenow import ClenowTrend
from bot.strategies.holygrail import HolyGrailPullback
from bot.strategies.lorentzian import LorentzianClassifier
from bot.strategies.newsdriven import NewsDriven
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
    # Single-name news, for instruments whose tape is too slow to signal.
    NewsDriven.name: NewsDriven,
}

# The three published systems that survived the out-of-sample split.
SELECTIVE = ["clenow", "turtle", "holygrail"]
# Indicator-style signals in the LuxAlgo mould.
LUX = ["supertrend", "smc", "nwenvelope", "lorentzian"]
# Driven by the wire rather than the chart. Equities print one bar a day
# and get one entry slot, so over a 90-day replay the equity leg managed
# nine trades — there were barely any setups to have an opinion about.
NEWS = ["news"]

# What actually gets run. See the module docstring for the bench that
# chose it.
PROFITABLE = ["clenow", "turtle"]

DEFAULT_ENABLED = PROFITABLE


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
    "LorentzianClassifier", "NewsDriven",
    "REGISTRY", "DEFAULT_ENABLED", "PROFITABLE", "SELECTIVE", "LUX", "NEWS",
    "build_strategies",
]
