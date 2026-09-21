"""The day plan: strategies in, sized and gated trades out."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.config_loader import validate_config
from bot.daily.briefing import Briefing
from bot.daily.plan import DayPlanner
from bot.portfolio.allocator import StrategyAllocator
from bot.portfolio.voltarget import VolatilityTargeter
from bot.risk.budget import RiskBudget
from bot.strategies import build_strategies
from tests.conftest import build_context

NOW = datetime(2026, 7, 1, 0, 30, tzinfo=timezone.utc)


def planner(config):
    return DayPlanner(config, RiskBudget(config), build_strategies(config),
                      StrategyAllocator(config))


def briefing_from(ctx, tone=0.0, equity=10_000.0):
    return Briefing(
        day=ctx.day, generated_at=NOW.isoformat(), equity=equity, cash=equity,
        symbols=ctx.reads,
        market_tone={"score": tone, "articles": 10, "tone": "neutral"},
        rolling_stats={},
    )


def build_plan(config, ctx, equity=10_000.0, positions=None, tone=0.0,
               exposure=None, **kwargs):
    p = planner(config)
    weights = p.allocator.weights([s.name for s in p.strategies])
    return p.build(
        briefing=briefing_from(ctx, tone, equity), equity=equity,
        open_positions=positions or [], peak_equity=kwargs.pop("peak", equity),
        day_start_equity=kwargs.pop("day_start", equity),
        context=ctx, strategy_weights=weights, exposure=exposure, now=NOW, **kwargs,
    )


# ── End to end ───────────────────────────────────────────────

def test_a_trending_universe_produces_trades(config):
    ctx = build_context(drifts=[0.006, 0.004, -0.005, -0.003],
                        htf_trends=[1, 1, -1, -1], seed=20)
    plan = build_plan(config, ctx)

    assert plan.trades, "a clearly trending market should produce trades"
    assert plan.signals, "the plan must record what each strategy said"
    assert plan.strategy_weights
    for trade in plan.trades:
        assert trade.quantity > 0 and trade.risk_usd > 0
        assert trade.strategy, "every trade must be attributed to a strategy"


def test_trades_are_attributed_to_the_driving_strategy(config):
    ctx = build_context(drifts=[0.008, 0.005, -0.006, -0.004],
                        htf_trends=[1, 1, -1, -1], seed=21)
    plan = build_plan(config, ctx)
    names = {s.name for s in build_strategies(config)}
    for trade in plan.trades:
        assert trade.strategy in names | {"blend"}


def test_stops_sit_on_the_correct_side(config):
    ctx = build_context(drifts=[0.007, -0.007, 0.005, -0.005],
                        htf_trends=[1, -1, 1, -1], seed=22)
    plan = build_plan(config, ctx)
    assert plan.trades
    for trade in plan.trades:
        if trade.side == "long":
            assert trade.stop_price < trade.entry_price < trade.take_profit
        else:
            assert trade.take_profit < trade.entry_price < trade.stop_price


def test_a_symbol_nobody_has_a_view_on_is_skipped(config):
    ctx = build_context(symbols=["A/USDT"], bars=100, drifts=[0.0])
    plan = build_plan(config, ctx)
    assert plan.trades == []
    assert any("no strategy has a view" in reason for _, reason in plan.rejected)


def test_untradable_symbols_never_reach_the_plan(config):
    ctx = build_context(drifts=[0.006, 0.004, -0.005, -0.003], seed=23)
    for read in ctx.reads.values():
        read.tradable = False
        read.skip_reason = "thin book"

    plan = build_plan(config, ctx)
    assert plan.trades == []
    assert all(reason == "thin book" for _, reason in plan.rejected)


# ── Tilts ────────────────────────────────────────────────────

def test_news_cannot_open_a_position_on_its_own(config):
    """Sentiment is a weak, fast-decaying predictor. It tilts, never triggers."""
    ctx = build_context(symbols=["A/USDT"], bars=100, drifts=[0.0])
    for read in ctx.reads.values():
        read.news_tilt = 1.0
        read.news_articles = 40

    plan = build_plan(config, ctx)
    assert plan.trades == [], "max bullish news with no strategy view must not trade"


def test_news_veto_kills_a_setup_it_opposes(config):
    ctx = build_context(drifts=[0.008, 0.006, 0.005, 0.004],
                        htf_trends=[1, 1, 1, 1], seed=24)
    without = build_plan(config, ctx)
    if not without.trades:
        pytest.skip("no trades in this synthetic sample")

    opposed = build_context(drifts=[0.008, 0.006, 0.005, 0.004],
                            htf_trends=[1, 1, 1, 1], seed=24)
    for read in opposed.reads.values():
        read.news_tilt = -0.9

    plan = build_plan(config, opposed)
    assert any("opposes" in reason for _, reason in plan.rejected)


def test_a_tilt_cannot_flip_the_strategies_direction(config):
    ctx = build_context(drifts=[0.006, 0.004, -0.005, -0.003],
                        htf_trends=[1, 1, -1, -1], seed=25)
    plan = build_plan(config, ctx)
    for candidate in plan.considered:
        view_direction = 1 if sum(candidate.components["strategies"].values()) > 0 else -1
        traded = [t for t in plan.trades if t.symbol == candidate.symbol]
        for trade in traded:
            assert (1 if trade.side == "long" else -1) == view_direction


def test_higher_timeframe_veto_exempts_market_neutral_strategies(config):
    """Cross-sectional momentum shorting a laggard in a rising market is the
    construction, not a mistake."""
    config["strategies"]["enabled"] = ["xsmom"]
    ctx = build_context(drifts=[0.010, 0.008, 0.006, 0.001], vol=0.004,
                        htf_trends=[1, 1, 1, 1], seed=26)
    plan = build_plan(config, ctx)

    shorts = [t for t in plan.trades if t.side == "short"]
    vetoes = [r for _, r in plan.rejected if "higher-TF" in r]
    assert shorts or not vetoes, "a neutral strategy must not be vetoed by the trend"


# ── Portfolio controls ───────────────────────────────────────

def test_vol_target_scales_every_position_together(config):
    ctx = build_context(drifts=[0.006, 0.004, -0.005, -0.003],
                        htf_trends=[1, 1, -1, -1], seed=27)

    full = build_plan(config, ctx)
    if not full.trades:
        pytest.skip("no trades in this synthetic sample")

    halved = VolatilityTargeter(config).decide([0.0] * 3)   # neutral, then force
    halved.scale = 0.5
    reduced = build_plan(config, ctx, exposure=halved)

    assert reduced.risk_pct_per_trade == pytest.approx(full.risk_pct_per_trade * 0.5)
    if reduced.trades:
        assert sum(t.risk_usd for t in reduced.trades) < sum(t.risk_usd for t in full.trades)


def test_plan_respects_the_daily_new_position_cap(config):
    config["session"]["max_new_positions_per_day"] = 1
    ctx = build_context(drifts=[0.008, 0.006, -0.007, -0.005],
                        htf_trends=[1, 1, -1, -1], seed=28)
    plan = build_plan(config, ctx)
    assert len(plan.trades) <= 1


def test_plan_stays_inside_the_heat_budget(config):
    config["session"]["max_new_positions_per_day"] = 10
    config["risk"]["max_correlated_positions"] = 10
    ctx = build_context(symbols=[f"C{i}/USDT" for i in range(6)],
                        drifts=[0.007] * 6, htf_trends=[1] * 6, seed=29)
    plan = build_plan(config, ctx)

    total_risk_pct = sum(t.risk_usd for t in plan.trades) / 10_000 * 100
    assert total_risk_pct <= config["risk"]["max_portfolio_heat_pct"] + 1e-6


def test_planning_stops_when_a_breaker_has_fired(config):
    ctx = build_context(drifts=[0.008, 0.006, -0.007, -0.005],
                        htf_trends=[1, 1, -1, -1], seed=30)
    plan = build_plan(config, ctx, equity=9_700, peak=10_000, day_start=10_000)
    assert plan.trades == []
    assert any("daily_loss_limit" in note for note in plan.notes)


def test_a_failing_strategy_does_not_stop_the_day(config):
    class Exploding:
        name = "exploding"
        market_neutral = False

        def generate(self, ctx):
            raise RuntimeError("boom")

    ctx = build_context(drifts=[0.006, 0.004, -0.005, -0.003],
                        htf_trends=[1, 1, -1, -1], seed=31)
    strategies = build_strategies(config) + [Exploding()]
    p = DayPlanner(config, RiskBudget(config), strategies, StrategyAllocator(config))
    weights = p.allocator.weights([s.name for s in strategies])

    plan = p.build(briefing=briefing_from(ctx), equity=10_000, open_positions=[],
                   peak_equity=10_000, day_start_equity=10_000, context=ctx,
                   strategy_weights=weights, now=NOW)
    assert "exploding" not in plan.signals
    assert plan.considered, "the working strategies must still be heard"


# ── Config validation ────────────────────────────────────────

def test_config_rejects_risk_above_the_heat_budget(config):
    config["risk"]["risk_per_trade_pct"] = 10.0
    with pytest.raises(ValueError, match="exceeds"):
        validate_config(config)


def test_config_rejects_a_daily_stop_above_the_hard_halt(config):
    config["risk"]["max_daily_loss_pct"] = 20.0
    with pytest.raises(ValueError, match="below"):
        validate_config(config)


def test_config_rejects_a_malformed_symbol(config):
    config["data"]["symbols"] = ["BTCUSDT"]
    with pytest.raises(ValueError, match="BASE/QUOTE"):
        validate_config(config)


def test_config_rejects_history_too_short_for_a_strategy(config):
    """A starved strategy returns nothing and looks merely opinionless."""
    config["data"]["history_bars"] = 150
    with pytest.raises(ValueError, match="history_bars"):
        validate_config(config)


def test_a_valid_config_passes(config):
    validate_config(config)
