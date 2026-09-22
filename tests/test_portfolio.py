"""Portfolio construction: risk budgeting and exposure control.

These test the two levers the research says actually move drawdown —
equalising risk across weakly-correlated drivers, and scaling the whole
book by realised volatility.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest

from bot.portfolio.allocator import StrategyAllocator
from bot.portfolio.tracker import StrategyTracker
from bot.portfolio.voltarget import VolatilityTargeter
from bot.strategies.base import StrategySignal
from bot.trading.models import Trade

T0 = datetime(2026, 6, 1, tzinfo=timezone.utc)
NAMES = ["trend", "xsmom", "breakout", "reversion", "carry"]


def signal(strategy, symbol, direction, strength=0.8):
    return StrategySignal(strategy=strategy, symbol=symbol, direction=direction,
                          strength=strength, reason=f"{strategy} view")


def trade(strategy, day, pnl):
    return Trade(
        id=f"{strategy}-{day}", symbol="BTC/USDT", side="long", entry_price=100,
        exit_price=101, quantity=1, leverage=1,
        opened_at=T0, closed_at=T0 + timedelta(hours=4), pnl=pnl, pnl_pct=pnl / 100,
        r_multiple=pnl / 50, fees=0.1, funding=0, slippage_cost=0,
        exit_reason="take_profit" if pnl > 0 else "stop_loss",
        risk_usd=50, strategy=strategy, closed_on_day=day,
    )


# ── Weighting ────────────────────────────────────────────────

def test_weights_always_sum_to_one(config):
    allocator = StrategyAllocator(config)
    for vols in ({}, {n: 0.01 for n in NAMES},
                 {"trend": 0.03, "xsmom": 0.005, "breakout": 0.02}):
        weights = allocator.weights(NAMES, vols)
        assert sum(weights.values()) == pytest.approx(1.0)
        assert set(weights) == set(NAMES)


def test_a_noisier_strategy_gets_less_risk(config):
    """Equal capital is not equal risk, and it is risk that makes drawdown."""
    allocator = StrategyAllocator(config)
    weights = allocator.weights(
        ["calm", "wild"], {"calm": 0.005, "wild": 0.020}
    )
    assert weights["calm"] > weights["wild"]


def test_no_history_means_equal_weights(config):
    """Estimating covariance from a fortnight of noise is worse than not."""
    config["portfolio"]["max_weights"] = {}
    weights = StrategyAllocator(config).weights(NAMES, volatilities={})
    assert all(w == pytest.approx(1 / len(NAMES)) for w in weights.values())


def test_an_infeasible_cap_is_ignored_not_silently_flattened(config):
    """max_strategy_weight below 1/n cannot bind; clamping toward it would
    discard the ranking entirely."""
    config["portfolio"]["max_strategy_weight"] = 0.40   # impossible for n=2
    config["portfolio"]["max_weights"] = {}
    weights = StrategyAllocator(config).weights(
        ["calm", "wild"], {"calm": 0.005, "wild": 0.020}
    )
    assert weights["calm"] > weights["wild"] + 0.1
    assert sum(weights.values()) == pytest.approx(1.0)


def test_correlated_strategies_are_haircut(config):
    """Five copies of one signal is one bet, not five."""
    allocator = StrategyAllocator(config)
    vols = {n: 0.01 for n in NAMES}

    plain = allocator.weights(NAMES, vols)
    corr = pd.DataFrame(np.eye(len(NAMES)), index=NAMES, columns=NAMES)
    corr.loc["trend", "breakout"] = corr.loc["breakout", "trend"] = 0.95
    haircut = allocator.weights(NAMES, vols, corr)

    assert haircut["trend"] < plain["trend"]
    assert haircut["breakout"] < plain["breakout"]
    assert haircut["xsmom"] > plain["xsmom"], "the independent driver gains"


def test_per_strategy_caps_are_respected(config):
    """Carry has tiny daily vol and a fat tail — inverse vol over-funds it."""
    config["portfolio"]["max_weights"] = {"carry": 0.10}
    weights = StrategyAllocator(config).weights(
        NAMES, {"trend": 0.02, "xsmom": 0.015, "breakout": 0.025,
                "reversion": 0.018, "carry": 0.001},
    )
    assert weights["carry"] <= 0.10 + 1e-9
    assert sum(weights.values()) == pytest.approx(1.0)


def test_manual_weights_are_honoured(config):
    config["portfolio"]["weighting"] = "manual"
    config["portfolio"]["weights"] = {"trend": 3, "xsmom": 1}
    weights = StrategyAllocator(config).weights(["trend", "xsmom"])
    assert weights["trend"] > weights["xsmom"]
    assert sum(weights.values()) == pytest.approx(1.0)


# ── Combination ──────────────────────────────────────────────

def test_agreeing_strategies_compound_conviction(config):
    allocator = StrategyAllocator(config)
    weights = {n: 0.2 for n in NAMES}

    one = allocator.combine({"trend": [signal("trend", "BTC/USDT", 1)]}, weights)
    three = allocator.combine({
        "trend": [signal("trend", "BTC/USDT", 1)],
        "breakout": [signal("breakout", "BTC/USDT", 1)],
        "xsmom": [signal("xsmom", "BTC/USDT", 1)],
    }, weights)

    assert three["BTC/USDT"].conviction > one["BTC/USDT"].conviction
    assert three["BTC/USDT"].agreement == pytest.approx(1.0)


def test_opposed_strategies_cancel_rather_than_both_trading(config):
    """Never hold two opposite positions in one name."""
    allocator = StrategyAllocator(config)
    views = allocator.combine({
        "trend": [signal("trend", "BTC/USDT", 1, 0.8)],
        "reversion": [signal("reversion", "BTC/USDT", -1, 0.8)],
    }, {"trend": 0.5, "reversion": 0.5})

    view = views["BTC/USDT"]
    assert view.conviction < 0.1
    assert abs(view.agreement) < 0.1


def test_the_stronger_side_wins_a_disagreement(config):
    allocator = StrategyAllocator(config)
    views = allocator.combine({
        "trend": [signal("trend", "BTC/USDT", 1, 0.9)],
        "reversion": [signal("reversion", "BTC/USDT", -1, 0.3)],
    }, {"trend": 0.6, "reversion": 0.4})
    assert views["BTC/USDT"].direction == 1


def test_a_minimum_agreement_filters_contested_names(config):
    config["portfolio"]["min_agreement"] = 0.5
    allocator = StrategyAllocator(config)
    views = allocator.combine({
        "trend": [signal("trend", "BTC/USDT", 1, 0.6)],
        "reversion": [signal("reversion", "BTC/USDT", -1, 0.5)],
    }, {"trend": 0.5, "reversion": 0.5})
    assert "BTC/USDT" not in views


def test_zero_weight_strategies_are_ignored(config):
    allocator = StrategyAllocator(config)
    views = allocator.combine(
        {"carry": [signal("carry", "BTC/USDT", -1)]}, {"carry": 0.0}
    )
    assert views == {}


# ── Attribution ──────────────────────────────────────────────

def test_returns_are_attributed_per_strategy(config):
    tracker = StrategyTracker(config)
    trades = [trade("trend", "2026-06-01", 100), trade("xsmom", "2026-06-01", -40),
              trade("trend", "2026-06-02", -20), trade("xsmom", "2026-06-02", 60)]

    returns = tracker.daily_returns(trades, equity=10_000)
    assert list(returns.columns) == ["trend", "xsmom"]
    assert returns.loc["2026-06-01", "trend"] == pytest.approx(0.01)
    assert returns.loc["2026-06-02", "xsmom"] == pytest.approx(0.006)


def test_a_day_a_strategy_sat_out_is_flat_not_missing(config):
    tracker = StrategyTracker(config)
    trades = [trade("trend", "2026-06-01", 100), trade("xsmom", "2026-06-02", 50)]
    returns = tracker.daily_returns(trades, equity=10_000)
    assert returns.loc["2026-06-02", "trend"] == 0.0
    assert not returns.isna().any().any()


def test_no_trades_means_no_measurements(config):
    tracker = StrategyTracker(config)
    assert tracker.daily_returns([], 10_000).empty
    assert tracker.volatilities(pd.DataFrame()) == {}
    assert tracker.correlations(pd.DataFrame()).empty


# ── Volatility targeting ─────────────────────────────────────

def test_calm_markets_lever_up_and_wild_ones_lever_down(config):
    rng = np.random.default_rng(0)
    calm = list(rng.normal(0, 0.003, 40))
    wild = list(rng.normal(0, 0.030, 40))

    targeter = VolatilityTargeter(config)
    calm_scale = targeter.decide(calm).scale
    targeter.reset()
    wild_scale = targeter.decide(wild).scale

    assert calm_scale > 1.0 > wild_scale


def test_exposure_stays_inside_its_bounds(config):
    rng = np.random.default_rng(1)
    targeter = VolatilityTargeter(config)
    for sigma in (1e-9, 0.001, 0.05, 0.5):
        targeter.reset()
        decision = targeter.decide(list(rng.normal(0, sigma, 40)))
        assert config["vol_target"]["min_scale"] <= decision.scale \
            <= config["vol_target"]["max_scale"]


def test_too_little_history_holds_exposure_at_one(config):
    decision = VolatilityTargeter(config).decide([0.001, -0.002, 0.003])
    assert decision.scale == pytest.approx(1.0)
    assert "history" in " ".join(decision.reasons)


def test_de_risking_is_faster_than_re_risking(config):
    """Volatility spikes arrive faster than they decay."""
    rng = np.random.default_rng(2)
    calm = list(rng.normal(0, 0.003, 40))
    wild = list(rng.normal(0, 0.030, 40))

    targeter = VolatilityTargeter(config)
    targeter.decide(calm)                      # settle high
    after_shock = targeter.decide(wild).scale  # must drop at once
    recovering = targeter.decide(calm).scale   # must climb back slowly

    assert after_shock < 1.0
    assert recovering > after_shock
    assert recovering < 1.4, "restoring exposure should be gradual"


def test_drawdown_throttles_exposure_further(config):
    rng = np.random.default_rng(3)
    returns = list(rng.normal(0, 0.008, 40))

    targeter = VolatilityTargeter(config)
    flat = targeter.decide(returns, drawdown_pct=0.0).scale
    targeter.reset()
    sunk = targeter.decide(returns, drawdown_pct=10.0).scale

    assert sunk < flat
    assert any("drawdown" in r for r in
               VolatilityTargeter(config).decide(returns, drawdown_pct=10.0).reasons)


def test_targeting_can_be_switched_off(config):
    config["vol_target"]["enabled"] = False
    decision = VolatilityTargeter(config).decide([0.05] * 40)
    assert decision.scale == 1.0


def test_a_benched_strategy_cannot_shout_louder_than_a_trusted_one(config):
    """Regression: dividing conviction by the weight actually present made
    a 5% probe produce *more* conviction than a full allocation, because
    the small denominator inflated the mean — defeating both risk parity
    and the autopilot."""
    allocator = StrategyAllocator(config)
    names = [f"s{i}" for i in range(9)]

    equal = {name: 1 / 9 for name in names}
    probe = dict(equal)
    probe["s0"] = 0.005
    total = sum(probe.values())
    probe = {k: v / total for k, v in probe.items()}

    trusted = allocator.combine({"s0": [signal("s0", "BTC/USDT", 1)]}, equal)
    benched = allocator.combine({"s0": [signal("s0", "BTC/USDT", 1)]}, probe)

    assert benched["BTC/USDT"].conviction < trusted["BTC/USDT"].conviction / 5


def test_conviction_barely_moves_with_roster_size(config):
    """A fixed entry threshold must mean the same thing whatever is enabled."""
    allocator = StrategyAllocator(config)
    convictions = []
    for size in (3, 5, 9):
        names = [f"s{i}" for i in range(size)]
        weights = {name: 1 / size for name in names}
        view = allocator.combine({"s0": [signal("s0", "BTC/USDT", 1)]}, weights)
        convictions.append(view["BTC/USDT"].conviction)

    assert max(convictions) - min(convictions) < 0.15


def test_unanimity_still_beats_a_lone_voice(config):
    allocator = StrategyAllocator(config)
    names = [f"s{i}" for i in range(9)]
    weights = {name: 1 / 9 for name in names}

    alone = allocator.combine({"s0": [signal("s0", "BTC/USDT", 1)]}, weights)
    everyone = allocator.combine(
        {name: [signal(name, "BTC/USDT", 1)] for name in names}, weights
    )
    assert everyone["BTC/USDT"].conviction > alone["BTC/USDT"].conviction
