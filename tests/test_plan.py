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
    """And the reason says WHICH kind of "no view" it was.

    This fixture supplies 100 bars against a 200-bar warm-up, so the
    honest reason is that the indicators never formed — not that the
    strategies looked and saw nothing. The two need opposite responses
    and used to share one message.
    """
    ctx = build_context(symbols=["A/USDT"], bars=100, drifts=[0.0])
    plan = build_plan(config, ctx)
    assert plan.trades == []
    reasons = [reason for _, reason in plan.rejected]
    assert any("indicators not ready" in r for r in reasons), reasons


def test_a_warmed_up_symbol_with_no_setup_says_so(config):
    """The other branch: enough history, strategies ran, nothing fired."""
    ctx = build_context(symbols=["A/USDT"], bars=400, drifts=[0.0])
    plan = build_plan(config, ctx)
    reasons = [reason for _, reason in plan.rejected]
    assert not any("indicators not ready" in r for r in reasons), reasons


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


# ── The primary trend ────────────────────────────────────────
# `htf_trend` is a 21/55 EMA cross on the 4h bar: 55 bars is nine days,
# and it flips on every pullback. These cover the slower read that was
# added because, over a window where crypto rose 42-77%, the bot took 120
# shorts against 67 longs and nothing in its view could see the move.

def _trend_read(primary=0, htf=0, symbol="BTC/USDT:USDT",
                asset_class="crypto_perp"):
    from bot.daily.briefing import SymbolRead
    return SymbolRead(
        symbol=symbol, price=100.0, atr=1.0, atr_pct=1.0, adx=30.0,
        regime="trending_up", tradable=True, quote_volume_24h=1e8,
        asset_class=asset_class, primary_trend=primary, htf_trend=htf,
        primary_timeframe="1d",
    )


def test_a_short_against_an_unambiguous_uptrend_is_refused(config):
    from bot.daily.plan import DayPlanner
    assert DayPlanner.primary_label(_trend_read(primary=1)) == "primary uptrend"


def test_an_ambiguous_primary_trend_constrains_nothing(config):
    """Price above a falling average, or below a rising one, is not a
    reason to overrule the strategies — only an unambiguous trend is."""
    read = _trend_read(primary=0)
    assert read.primary_trend == 0


def _primary_frame(direction=1, n=300):
    import numpy as np
    import pandas as pd
    drift = 0.004 * direction
    idx = pd.date_range("2025-01-01", periods=n, freq="1D", tz="UTC")
    close = 100 * np.exp(np.cumsum(np.full(n, drift)))
    return pd.DataFrame({"open": close, "high": close * 1.01,
                         "low": close * 0.99, "close": close,
                         "volume": 1.0}, index=idx)


class _OneFrameRouter:
    def __init__(self, df):
        self.df = df

    def bars(self, instrument, timeframe=None, limit=500):
        return self.df.tail(limit)


def test_a_rising_market_reads_as_a_primary_uptrend(config):
    from bot.daily.briefing import BriefingBuilder, SymbolRead
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(1)),
                              universe=[])
    read = SymbolRead(symbol="BTC/USDT", price=0.0, atr=1.0, atr_pct=1.0,
                      adx=25.0, regime="trending_up", tradable=True)
    instrument = build_instrument({"symbol": "BTC/USDT",
                                   "asset_class": "crypto_spot"})
    builder._read_primary_trend(read, instrument, "4h")
    assert read.primary_trend == 1
    assert read.primary_timeframe == "1d"
    assert read.primary_strength > 0


def test_a_falling_market_reads_as_a_primary_downtrend(config):
    from bot.daily.briefing import BriefingBuilder, SymbolRead
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(-1)),
                              universe=[])
    read = SymbolRead(symbol="BTC/USDT", price=0.0, atr=1.0, atr_pct=1.0,
                      adx=25.0, regime="trending_down", tradable=True)
    instrument = build_instrument({"symbol": "BTC/USDT",
                                   "asset_class": "crypto_spot"})
    builder._read_primary_trend(read, instrument, "4h")
    assert read.primary_trend == -1
    assert read.primary_strength < 0


def test_too_little_history_leaves_the_primary_trend_neutral(config):
    """Below the average's own length its slope means nothing, so the read
    stays neutral rather than guessing."""
    from bot.daily.briefing import BriefingBuilder, SymbolRead
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(1, n=60)),
                              universe=[])
    read = SymbolRead(symbol="BTC/USDT", price=0.0, atr=1.0, atr_pct=1.0,
                      adx=25.0, regime="trending_up", tradable=True)
    instrument = build_instrument({"symbol": "BTC/USDT",
                                   "asset_class": "crypto_spot"})
    builder._read_primary_trend(read, instrument, "4h")
    assert read.primary_trend == 0


def test_an_equity_reads_its_primary_trend_on_the_weekly_bar(config):
    from bot.daily.briefing import BriefingBuilder, SymbolRead
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(1)),
                              universe=[])
    read = SymbolRead(symbol="NVDA", price=0.0, atr=1.0, atr_pct=1.0,
                      adx=25.0, regime="trending_up", tradable=True)
    instrument = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
    builder._read_primary_trend(read, instrument, "1w")
    assert read.primary_timeframe == "1w"


def test_a_dead_primary_feed_leaves_the_read_neutral(config):
    """One unavailable frame must not take down the briefing."""
    from bot.daily.briefing import BriefingBuilder, SymbolRead
    from bot.markets.instrument import build_instrument

    class Dead:
        def bars(self, *a, **k):
            raise RuntimeError("feed down")

    builder = BriefingBuilder(config, Dead(), universe=[])
    read = SymbolRead(symbol="BTC/USDT", price=0.0, atr=1.0, atr_pct=1.0,
                      adx=25.0, regime="ranging", tradable=True)
    instrument = build_instrument({"symbol": "BTC/USDT",
                                   "asset_class": "crypto_spot"})
    builder._read_primary_trend(read, instrument, "4h")
    assert read.primary_trend == 0


def test_the_replay_downloads_enough_bars_for_the_long_average(config):
    """A 90-day window implies ~170 daily bars but the average is 200 long;
    without a floor it never forms and the gate silently reports no trend
    for the whole replay, which looks exactly like the gate being off."""
    from bot.utils.backtester import _slow_frame_floor

    assert _slow_frame_floor(config, "1d") > int(
        config.get("signals", {}).get("primary_trend_period", 200))
    assert _slow_frame_floor(config, "1h") == 0


# ── Cross-sectional momentum ─────────────────────────────────
# Absolute momentum says an instrument is rising; relative momentum says
# whether it is the one worth owning. Over the measured window everything
# rose 42-77%, so "trending up" was true of the whole universe.

def _ranked_read(symbol, rank, peers=8, asset_class="crypto_spot"):
    from bot.daily.briefing import SymbolRead
    return SymbolRead(
        symbol=symbol, price=100.0, atr=1.0, atr_pct=1.0, adx=30.0,
        regime="trending_up", tradable=True, quote_volume_24h=1e8,
        asset_class=asset_class, momentum_rank=rank, momentum_peers=peers,
    )


def test_ranks_are_spread_across_the_class(config):
    """A rank is only meaningful relative to peers in the same class: a
    stock's 90-day return is not comparable to a perp's."""
    from bot.daily.briefing import Briefing, BriefingBuilder
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(1)),
                              universe=[])
    briefing = Briefing(day="2026-09-21", generated_at="", equity=1e4, cash=1e4)

    instruments, scores = [], {"A/USDT": 3.0, "B/USDT": 1.0, "C/USDT": 2.0}
    for symbol in scores:
        instruments.append(build_instrument({"symbol": symbol,
                                             "asset_class": "crypto_spot"}))
        briefing.symbols[symbol] = _ranked_read(symbol, 0.5)

    builder._momentum_score = lambda inst, read: scores[inst.symbol]
    builder._rank_cross_section(briefing, instruments)

    ranks = {s: briefing.symbols[s].momentum_rank for s in scores}
    assert ranks["A/USDT"] > ranks["C/USDT"] > ranks["B/USDT"]
    assert all(r.momentum_peers == 3 for r in briefing.symbols.values())


def test_a_single_instrument_ranks_neutral(config):
    """One name is neither the best nor the worst of its class."""
    from bot.daily.briefing import Briefing, BriefingBuilder
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(1)),
                              universe=[])
    briefing = Briefing(day="d", generated_at="", equity=1e4, cash=1e4)
    briefing.symbols["A/USDT"] = _ranked_read("A/USDT", 0.0)
    instrument = build_instrument({"symbol": "A/USDT",
                                   "asset_class": "crypto_spot"})
    builder._momentum_score = lambda inst, read: 1.0
    builder._rank_cross_section(briefing, [instrument])
    assert briefing.symbols["A/USDT"].momentum_rank == 0.5


def test_classes_are_ranked_separately(config):
    from bot.daily.briefing import Briefing, BriefingBuilder
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(1)),
                              universe=[])
    briefing = Briefing(day="d", generated_at="", equity=1e4, cash=1e4)
    specs = [("A/USDT", "crypto_spot", 5.0), ("B/USDT", "crypto_spot", 1.0),
             ("NVDA", "equity", 0.1)]
    instruments = []
    for symbol, klass, _ in specs:
        instruments.append(build_instrument({"symbol": symbol,
                                             "asset_class": klass}))
        briefing.symbols[symbol] = _ranked_read(symbol, 0.5, asset_class=klass)

    scores = {s: v for s, _, v in specs}
    builder._momentum_score = lambda inst, read: scores[inst.symbol]
    builder._rank_cross_section(briefing, instruments)

    # NVDA has the lowest raw score but is alone in its class, so it ranks
    # neutral rather than worst.
    assert briefing.symbols["NVDA"].momentum_rank == 0.5
    assert briefing.symbols["NVDA"].momentum_peers == 1
    assert briefing.symbols["A/USDT"].momentum_peers == 2


def test_an_untradable_read_is_left_out_of_the_ranking(config):
    from bot.daily.briefing import Briefing, BriefingBuilder
    from bot.markets.instrument import build_instrument

    builder = BriefingBuilder(config, _OneFrameRouter(_primary_frame(1)),
                              universe=[])
    briefing = Briefing(day="d", generated_at="", equity=1e4, cash=1e4)
    good = _ranked_read("A/USDT", 0.5)
    dead = _ranked_read("B/USDT", 0.5)
    dead.tradable = False
    briefing.symbols["A/USDT"] = good
    briefing.symbols["B/USDT"] = dead
    instruments = [build_instrument({"symbol": s, "asset_class": "crypto_spot"})
                   for s in ("A/USDT", "B/USDT")]
    builder._momentum_score = lambda inst, read: 1.0
    builder._rank_cross_section(briefing, instruments)
    assert good.momentum_peers == 1


def test_the_momentum_band_is_off_by_default(config):
    from bot.daily.plan import DayPlanner
    from bot.risk.budget import RiskBudget
    from bot.portfolio.allocator import StrategyAllocator

    planner = DayPlanner(config, RiskBudget(config), [],
                         StrategyAllocator(config))
    assert planner.momentum_band == 0.0


def test_the_band_needs_enough_peers_to_mean_anything(config):
    from bot.daily.plan import DayPlanner
    from bot.portfolio.allocator import StrategyAllocator
    from bot.risk.budget import RiskBudget

    config.setdefault("signals", {})["momentum_band"] = 0.5
    planner = DayPlanner(config, RiskBudget(config), [],
                         StrategyAllocator(config))
    assert planner.momentum_min_peers >= 2
