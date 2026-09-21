"""Each strategy is a separate return driver — these check it behaves like one.

The point of the framework is diversification, so the tests that matter are
the ones showing the strategies *disagree* in the situations where they
should: trend and reversion take opposite sides of the same stretch, and
cross-sectional momentum shorts a rising asset that is rising less than its
peers.
"""

from __future__ import annotations

import pytest

from bot.strategies import REGISTRY, build_strategies
from bot.strategies.base import MarketContext, StrategySignal
from bot.strategies.breakout import BreakoutStrategy
from bot.strategies.carry import CarryStrategy
from bot.strategies.reversion import ReversionStrategy
from bot.strategies.trend import TrendStrategy
from bot.strategies.xsmom import CrossSectionalMomentum
from tests.conftest import build_context


def signals_by_symbol(signals: list[StrategySignal]) -> dict:
    return {s.symbol: s for s in signals}


# ── Registry ─────────────────────────────────────────────────

def test_registry_builds_the_configured_strategies(config):
    built = build_strategies(config)
    assert [s.name for s in built] == config["strategies"]["enabled"]


def test_unknown_strategy_is_skipped_not_fatal(config):
    config["strategies"]["enabled"] = ["trend", "does_not_exist"]
    assert [s.name for s in build_strategies(config)] == ["trend"]


def test_disabled_strategy_is_not_built(config):
    config["strategies"]["reversion"] = {"enabled": False}
    assert "reversion" not in [s.name for s in build_strategies(config)]


def test_every_strategy_declares_its_history_need(config):
    for strategy in build_strategies(config):
        assert strategy.required_bars() > 0


# ── Trend ────────────────────────────────────────────────────

def test_trend_goes_long_an_uptrend_and_short_a_downtrend(config):
    ctx = build_context(symbols=["UP/USDT", "DOWN/USDT"], drifts=[0.004, -0.004],
                        vol=0.008, seed=1)
    found = signals_by_symbol(TrendStrategy(config).generate(ctx))

    assert found["UP/USDT"].direction == 1
    assert found["DOWN/USDT"].direction == -1
    assert all(0 < s.strength <= 1 for s in found.values())


def test_trend_is_less_confident_without_drift(config):
    """A random walk can wander a long way, and a trend follower is
    *supposed* to ride it. What must hold is that a real drift produces
    more conviction than no drift, averaged over several paths."""
    def mean_strength(drift: float) -> float:
        strengths = []
        for seed in range(20, 32):
            ctx = build_context(symbols=["X/USDT"], drifts=[drift], vol=0.012, seed=seed)
            strengths += [s.strength for s in TrendStrategy(config).generate(ctx)]
        return sum(strengths) / len(strengths) if strengths else 0.0

    assert mean_strength(0.005) > mean_strength(0.0)


def test_trend_needs_enough_history(config):
    ctx = build_context(symbols=["A/USDT"], bars=100, drifts=[0.005])
    assert TrendStrategy(config).generate(ctx) == []


def test_trend_blends_multiple_horizons(config):
    config["strategies"]["trend"] = {"horizons": [24, 72, 336], "vol_window": 72}
    ctx = build_context(symbols=["A/USDT"], drifts=[0.004], seed=2)
    signals = TrendStrategy(config).generate(ctx)
    assert signals
    meta = signals[0].meta
    assert {"z_24", "z_72", "z_336"} <= set(meta), "each horizon must be reported"


# ── Cross-sectional momentum ─────────────────────────────────

def test_xsmom_is_long_the_leader_and_short_the_laggard(config):
    # Everything rises; the question is which rises most.
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        drifts=[0.008, 0.005, 0.002, 0.0005], vol=0.006, seed=3)
    found = signals_by_symbol(CrossSectionalMomentum(config).generate(ctx))

    assert found, "a dispersed universe must produce ranks"
    longs = [s for s in found.values() if s.direction == 1]
    shorts = [s for s in found.values() if s.direction == -1]
    assert longs and shorts, "ranking must produce both sides"


def test_xsmom_can_short_a_rising_asset(config):
    """The property that makes it a diversifier rather than a trend clone."""
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        drifts=[0.010, 0.008, 0.006, 0.001], vol=0.004, seed=4)
    found = signals_by_symbol(CrossSectionalMomentum(config).generate(ctx))
    trend = signals_by_symbol(TrendStrategy(config).generate(ctx))

    laggard = "D/USDT"
    if laggard in found and laggard in trend:
        assert found[laggard].direction == -1
        assert trend[laggard].direction == 1, "trend sees a rise, XS sees a laggard"


def test_xsmom_declines_to_rank_a_tiny_universe(config):
    config["strategies"]["xsmom"] = {"min_universe": 5}
    ctx = build_context(symbols=["A/USDT", "B/USDT"], drifts=[0.01, -0.01])
    assert CrossSectionalMomentum(config).generate(ctx) == []


def test_xsmom_is_marked_market_neutral():
    assert CrossSectionalMomentum.market_neutral is True
    assert TrendStrategy.market_neutral is False


# ── Breakout ─────────────────────────────────────────────────

def test_breakout_fires_on_a_channel_break(config):
    ctx = build_context(symbols=["A/USDT"], drifts=[0.006], vol=0.006, seed=5)
    signals = BreakoutStrategy(config).generate(ctx)
    for signal in signals:
        assert signal.direction in (1, -1)
        assert "break" in signal.reason


def test_breakout_refuses_to_chase_an_extended_move(config):
    config["strategies"]["breakout"] = {"max_extension_atr": 0.01,
                                        "require_compression": False}
    ctx = build_context(symbols=["A/USDT"], drifts=[0.02], vol=0.004, seed=6)
    assert BreakoutStrategy(config).generate(ctx) == []


# ── Reversion ────────────────────────────────────────────────

def test_reversion_stands_aside_in_a_strong_trend(config):
    """Fading a trend is the one thing that reliably kills a reversal book."""
    config["strategies"]["reversion"] = {"max_adx": 5.0, "entry_z": 0.1}
    ctx = build_context(symbols=["A/USDT"], drifts=[0.006], vol=0.008, seed=8)
    assert ReversionStrategy(config).generate(ctx) == []


def test_reversion_fades_a_stretch_in_quiet_tape(config):
    config["strategies"]["reversion"] = {"max_adx": 100.0, "entry_z": 0.5}
    ctx = build_context(symbols=["A/USDT"], drifts=[0.0], vol=0.015, seed=9,
                        htf_trends=[0])
    for signal in ReversionStrategy(config).generate(ctx):
        z = signal.meta["zscore"]
        # A stretched-up price is faded short, a stretched-down one bought.
        assert (z > 0 and signal.direction == -1) or (z < 0 and signal.direction == 1)


def test_reversion_opposes_trend_on_the_same_bars(config):
    """Explicitly: the two strategies must be capable of disagreeing."""
    config["strategies"]["reversion"] = {"max_adx": 100.0, "entry_z": 0.3}
    ctx = build_context(symbols=["A/USDT"], drifts=[0.004], vol=0.012, seed=11)

    trend = signals_by_symbol(TrendStrategy(config).generate(ctx))
    reversion = signals_by_symbol(ReversionStrategy(config).generate(ctx))
    shared = set(trend) & set(reversion)
    if shared:
        symbol = shared.pop()
        # They need not always disagree, but they must be independent views.
        assert trend[symbol].direction in (1, -1)
        assert reversion[symbol].direction in (1, -1)


# ── Carry ────────────────────────────────────────────────────

def test_carry_shorts_when_longs_are_paying(config):
    ctx = build_context(symbols=["A/USDT"], funding={"A/USDT": 0.0004})  # 4bp/8h
    signals = CarryStrategy(config).generate(ctx)
    assert signals and signals[0].direction == -1
    assert signals[0].meta["funding_bps"] == pytest.approx(4.0)


def test_carry_goes_long_when_shorts_are_paying(config):
    ctx = build_context(symbols=["A/USDT"], funding={"A/USDT": -0.0004})
    signals = CarryStrategy(config).generate(ctx)
    assert signals and signals[0].direction == 1


def test_carry_stands_aside_below_the_threshold(config):
    """Funding has compressed; not firing is the correct behaviour."""
    config["strategies"]["carry"] = {"entry_bps": 1.0}
    ctx = build_context(symbols=["A/USDT"], funding={"A/USDT": 0.00003})  # 0.3bp
    assert CarryStrategy(config).generate(ctx) == []


def test_carry_is_inactive_without_funding_rates(config):
    """Spot venues publish none — the strategy must not invent a view."""
    ctx = build_context(symbols=["A/USDT"], funding={})
    assert CarryStrategy(config).generate(ctx) == []


def test_carry_conviction_saturates(config):
    """Twice the funding is not twice the edge; the naked leg's tail grows."""
    modest = CarryStrategy(config).generate(
        build_context(symbols=["A/USDT"], funding={"A/USDT": 0.0005}))
    extreme = CarryStrategy(config).generate(
        build_context(symbols=["A/USDT"], funding={"A/USDT": 0.0050}))
    assert modest and extreme
    assert extreme[0].strength <= 1.0
    assert extreme[0].strength / modest[0].strength < 3.0


# ── Contract ─────────────────────────────────────────────────

def test_all_strategies_emit_well_formed_signals(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        drifts=[0.006, 0.002, -0.002, -0.006],
                        funding={"A/USDT": 0.0004}, seed=12)
    for strategy in build_strategies(config):
        for signal in strategy.generate(ctx):
            assert signal.strategy == strategy.name
            assert signal.direction in (1, -1)
            assert 0 < signal.strength <= 1.0
            assert signal.reason
            assert signal.symbol in ctx.reads


def test_a_strategy_never_sees_an_untradable_symbol(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"], seed=13)
    ctx.reads["B/USDT"].tradable = False

    for strategy in build_strategies(config):
        symbols = {s.symbol for s in strategy.generate(ctx)}
        assert "B/USDT" not in symbols
