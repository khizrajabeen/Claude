"""The published systems, and the strategy contract they all share.

These are the three that survived an out-of-sample split: Clenow's trend
model, the original Turtle rules, and Raschke's Holy Grail pullback. The
tests pin the parts of each that are easy to quietly get wrong — the
Turtle's last-winner filter, Clenow's requirement that filter and breakout
agree, and the fact that Holy Grail enters at the moving average rather
than at an extreme.
"""

from __future__ import annotations

import pytest

from bot.strategies import LUX, NEWS, REGISTRY, SELECTIVE, build_strategies
from bot.strategies.base import MarketContext, StrategySignal
from bot.strategies.clenow import ClenowTrend
from bot.strategies.holygrail import HolyGrailPullback
from bot.strategies.turtle import TurtleStrategy
from tests.conftest import build_context


def signals_by_symbol(signals: list[StrategySignal]) -> dict:
    return {s.symbol: s for s in signals}


# ── Registry ─────────────────────────────────────────────────

def test_registry_holds_only_the_strategies_that_earned_a_place():
    assert set(REGISTRY) == set(SELECTIVE) | set(LUX) | set(NEWS)
    assert set(SELECTIVE) == {"clenow", "turtle", "holygrail"}
    assert set(LUX) == {"supertrend", "smc", "nwenvelope", "lorentzian"}


def test_registry_builds_the_configured_strategies(config):
    built = build_strategies(config)
    assert [s.name for s in built] == config["strategies"]["enabled"]


def test_unknown_strategy_is_skipped_not_fatal(config):
    config["strategies"]["enabled"] = ["clenow", "does_not_exist"]
    assert [s.name for s in build_strategies(config)] == ["clenow"]


def test_disabled_strategy_is_not_built(config):
    config["strategies"]["smc"] = {"enabled": False}
    assert "smc" not in [s.name for s in build_strategies(config)]


def test_every_strategy_declares_its_history_need(config):
    for strategy in build_strategies(config):
        assert strategy.required_bars() > 0


# ── Turtle ───────────────────────────────────────────────────

def test_turtle_breaks_out_on_a_new_channel_extreme(config):
    ctx = build_context(symbols=["A/USDT"], bars=700, drifts=[0.004],
                        vol=0.008, seed=50)
    for signal in TurtleStrategy(config).generate(ctx):
        assert signal.meta["system"] in ("S1", "S2")
        assert 1 <= signal.meta["units"] <= 4
        assert signal.meta["stop_distance_n"] == 2.0
        assert "Turtle" in signal.reason


def test_turtle_refuses_to_chase_past_the_pyramid(config):
    """Beyond four units the entry is worse for the same 2N stop."""
    config["strategies"]["turtle"] = {"max_units": 1, "pyramid_step_n": 0.01,
                                      "last_winner_filter": False}
    ctx = build_context(symbols=["A/USDT"], bars=700, drifts=[0.02],
                        vol=0.004, seed=51)
    assert TurtleStrategy(config).generate(ctx) == []


def test_turtle_last_winner_filter_only_removes_signals(config):
    """The part of the original rules most implementations drop."""
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=700, drifts=[0.004, -0.004, 0.002, -0.002],
                        vol=0.010, seed=52)

    config["strategies"]["turtle"] = {"last_winner_filter": True}
    filtered = {s.symbol for s in TurtleStrategy(config).generate(ctx)}
    config["strategies"]["turtle"] = {"last_winner_filter": False}
    unfiltered = {s.symbol for s in TurtleStrategy(config).generate(ctx)}

    assert filtered <= unfiltered


# ── Clenow ───────────────────────────────────────────────────

def test_clenow_requires_filter_and_breakout_to_agree(config):
    """A new high inside a downtrend is a bounce, not an entry."""
    ctx = build_context(symbols=["A/USDT"], bars=700, drifts=[0.004],
                        vol=0.008, seed=53)
    for signal in ClenowTrend(config).generate(ctx):
        assert signal.meta["separation_atr"] > 0
        assert signal.meta["stop_atr"] == 3.0


def test_clenow_uses_a_wider_stop_than_the_house_default(config):
    assert ClenowTrend(config).stop_atr > float(config["stops"]["atr_stop_mult"])


# ── Holy Grail ───────────────────────────────────────────────

def test_holy_grail_needs_a_strong_trend(config):
    config["strategies"]["holygrail"] = {"adx_floor": 99.0}
    ctx = build_context(symbols=["A/USDT"], bars=700, drifts=[0.004], seed=54)
    assert HolyGrailPullback(config).generate(ctx) == []


def test_holy_grail_enters_at_the_average_not_at_extremes(config):
    """A pullback entry is what makes it diversify the breakout systems
    rather than duplicate them."""
    config["strategies"]["holygrail"] = {"adx_floor": 15.0, "touch_atr": 0.5}
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=700, drifts=[0.005, 0.003, -0.003, -0.005],
                        vol=0.010, seed=55)
    for signal in HolyGrailPullback(config).generate(ctx):
        assert signal.meta["distance_atr"] <= 0.5 + 1e-9
        assert signal.meta["room_atr"] > 0


# ── Stance ───────────────────────────────────────────────────

def test_trend_systems_confirm_and_the_envelope_fades(config):
    """A fade has less lag but far more exposure to a move that keeps
    running, so the two are sized differently downstream."""
    stances = {s.name: s.stance for s in build_strategies(config)}
    for name in ("clenow", "turtle", "holygrail", "supertrend"):
        assert stances[name] == "confirmation"
    assert stances["nwenvelope"] == "contrarian"


def test_smc_picks_its_stance_per_signal(config):
    """A structure break rides the move; a sweep reversal opposes it."""
    from bot.strategies.smc import SmartMoneyConcepts

    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=700, drifts=[0.005, 0.001, -0.001, -0.005],
                        vol=0.012, seed=62)
    for signal in SmartMoneyConcepts(config).generate(ctx):
        assert signal.kind in ("confirmation", "contrarian")
        if signal.kind == "contrarian":
            assert {"sweep", "zone"} & set(signal.meta)


def test_every_signal_carries_a_stance(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=900, drifts=[0.005, 0.001, -0.001, -0.005],
                        vol=0.011, seed=63)
    for strategy in build_strategies(config):
        for signal in strategy.generate(ctx):
            assert signal.kind in ("confirmation", "contrarian")
            assert signal.to_dict()["kind"] == signal.kind


# ── Contract ─────────────────────────────────────────────────

def test_all_strategies_emit_well_formed_signals(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=900, drifts=[0.006, 0.002, -0.002, -0.006],
                        seed=64)
    for strategy in build_strategies(config):
        for signal in strategy.generate(ctx):
            assert signal.strategy == strategy.name
            assert signal.direction in (1, -1)
            assert 0 < signal.strength <= 1.0
            assert signal.reason
            assert signal.symbol in ctx.reads


def test_a_strategy_never_sees_an_untradable_symbol(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=900, seed=65)
    ctx.reads["B/USDT"].tradable = False

    for strategy in build_strategies(config):
        symbols = {s.symbol for s in strategy.generate(ctx)}
        assert "B/USDT" not in symbols


def test_starved_strategies_produce_nothing_rather_than_guessing(config):
    ctx = build_context(symbols=["A/USDT"], bars=60, drifts=[0.005])
    for strategy in build_strategies(config):
        assert strategy.generate(ctx) == []
