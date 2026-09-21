"""Signal blending, filters and the shape of a day plan."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.config_loader import validate_config
from bot.daily.briefing import Briefing, SymbolRead
from bot.daily.plan import DayPlanner
from bot.risk.budget import RiskBudget

NOW = datetime(2026, 7, 1, 0, 30, tzinfo=timezone.utc)


def read(symbol="BTC/USDT", **overrides):
    defaults = dict(
        symbol=symbol, price=100.0, atr=1.0, atr_pct=1.0, annualized_vol=0.6,
        adx=15.0, plus_di=20.0, minus_di=20.0, rsi=50.0, zscore=0.0,
        donchian=0.5, macd_hist=0.0, ema_fast=100.0, ema_slow=100.0,
        htf_trend=0, htf_strength=0.0, overnight_return_pct=0.0,
        regime="ranging", regime_confidence=0.6, quote_volume_24h=50_000_000.0,
        spread_bps=2.0, book_imbalance=0.0, tradable=True, bars=400,
    )
    defaults.update(overrides)
    return SymbolRead(**defaults)


def briefing(reads, tone=0.0, equity=10_000.0):
    return Briefing(
        day="2026-07-01", generated_at=NOW.isoformat(), equity=equity, cash=equity,
        symbols={r.symbol: r for r in reads},
        market_tone={"score": tone, "articles": 10, "tone": "neutral"},
        rolling_stats={},
    )


def planner(config):
    return DayPlanner(config, RiskBudget(config))


# ── Scoring ──────────────────────────────────────────────────

def test_an_uptrend_scores_long(config):
    score = planner(config).score(read(
        regime="trending_up", regime_confidence=0.9, adx=40, plus_di=35, minus_di=10,
        ema_fast=105, ema_slow=100, macd_hist=0.8, overnight_return_pct=1.5,
    ))
    assert score.side == "long"
    assert score.edge > 0.2


def test_a_downtrend_scores_short(config):
    score = planner(config).score(read(
        regime="trending_down", regime_confidence=0.9, adx=40, plus_di=10, minus_di=35,
        ema_fast=95, ema_slow=100, macd_hist=-0.8, overnight_return_pct=-1.5,
    ))
    assert score.side == "short"
    assert score.edge < -0.2


def test_the_same_stretch_is_read_differently_by_regime(config):
    """A z-score of -2 is a buy in a range and a falling knife in a trend."""
    stretched = dict(zscore=-2.2, rsi=25.0, donchian=0.05)
    ranging = planner(config).score(read(regime="ranging", regime_confidence=0.8, **stretched))
    trending = planner(config).score(read(
        regime="trending_down", regime_confidence=0.9, adx=40,
        plus_di=8, minus_di=38, ema_fast=95, ema_slow=100, **stretched,
    ))
    assert ranging.edge > trending.edge


def test_a_flat_market_produces_no_conviction(config):
    assert abs(planner(config).score(read()).edge) < config["signals"]["min_edge_score"]


def test_news_tilts_but_cannot_create_a_signal(config):
    """Sentiment is a weak, fast-decaying predictor; it must not trade alone."""
    quiet = read(news_tilt=1.0, news_score=1.0, news_articles=20)
    edge = planner(config).score(quiet).edge
    assert abs(edge) < config["signals"]["min_edge_score"], (
        "maximum bullish news on a flat tape must not reach the entry threshold"
    )


def test_news_moves_a_marginal_setup(config):
    base = read(regime="trending_up", regime_confidence=0.7, adx=28,
                plus_di=26, minus_di=16, ema_fast=102, ema_slow=100)
    without = planner(config).score(base).edge
    with_news = planner(config).score(read(
        regime="trending_up", regime_confidence=0.7, adx=28, plus_di=26,
        minus_di=16, ema_fast=102, ema_slow=100, news_tilt=0.8,
    )).edge
    assert with_news > without


def test_book_imbalance_is_only_a_nudge(config):
    base = read(regime="trending_up", adx=30, plus_di=28, minus_di=14,
                ema_fast=103, ema_slow=100)
    confirmed = planner(config).score(read(
        regime="trending_up", adx=30, plus_di=28, minus_di=14,
        ema_fast=103, ema_slow=100, book_imbalance=0.5,
    ))
    assert confirmed.edge > planner(config).score(base).edge
    assert confirmed.edge - planner(config).score(base).edge < 0.15


# ── Filters and vetoes ───────────────────────────────────────

def test_untradable_symbols_never_reach_the_plan(config):
    plan = planner(config).build(
        briefing([read("THIN/USDT", tradable=False, skip_reason="thin book",
                       regime="trending_up", adx=40, ema_fast=110, ema_slow=100)]),
        equity=10_000, open_positions=[], peak_equity=10_000,
        day_start_equity=10_000, now=NOW,
    )
    assert plan.trades == []
    assert plan.rejected[0][1] == "thin book"


def test_higher_timeframe_vetoes_a_counter_trend_entry(config):
    strong_long = dict(regime="trending_up", regime_confidence=0.9, adx=45,
                       plus_di=40, minus_di=8, ema_fast=108, ema_slow=100,
                       macd_hist=1.0, overnight_return_pct=2.0)

    with_veto = planner(config).build(
        briefing([read("BTC/USDT", htf_trend=-1, **strong_long)]),
        10_000, [], 10_000, 10_000, now=NOW,
    )
    assert with_veto.trades == []
    assert "higher-TF" in with_veto.rejected[0][1]

    config["signals"]["require_htf_agreement"] = False
    without_veto = planner(config).build(
        briefing([read("BTC/USDT", htf_trend=-1, **strong_long)]),
        10_000, [], 10_000, 10_000, now=NOW,
    )
    assert without_veto.trades


def test_news_can_veto_a_setup_it_opposes(config):
    """A setup strong enough to clear the threshold even after the tilt is
    applied still gets killed by the veto."""
    overwhelming = dict(
        regime="trending_up", regime_confidence=1.0, adx=60, plus_di=50,
        minus_di=3, ema_fast=125.0, ema_slow=100.0, macd_hist=2.0,
        overnight_return_pct=5.0, htf_trend=1, book_imbalance=0.5,
    )
    # Without the veto it trades.
    config["signals"]["news_can_veto"] = False
    allowed = planner(config).build(
        briefing([read("BTC/USDT", news_tilt=-0.9, **overwhelming)]),
        10_000, [], 10_000, 10_000, now=NOW,
    )
    assert allowed.trades, "setup must be strong enough for the veto to be the cause"

    config["signals"]["news_can_veto"] = True
    vetoed = planner(config).build(
        briefing([read("BTC/USDT", news_tilt=-0.9, **overwhelming)]),
        10_000, [], 10_000, 10_000, now=NOW,
    )
    assert vetoed.trades == []
    assert "opposes" in vetoed.rejected[0][1]


# ── Plan construction ────────────────────────────────────────

def test_plan_respects_the_daily_new_position_cap(config):
    config["session"]["max_new_positions_per_day"] = 2
    reads = [
        read(f"C{i}/USDT", regime="trending_up", regime_confidence=0.9, adx=45,
             plus_di=40, minus_di=8, ema_fast=108, ema_slow=100, htf_trend=1,
             macd_hist=1.0, overnight_return_pct=2.0)
        for i in range(5)
    ]
    plan = planner(config).build(briefing(reads), 10_000, [], 10_000, 10_000, now=NOW)
    assert len(plan.trades) == 2
    assert any("cap reached" in reason for _, reason in plan.rejected)


def test_plan_stays_inside_the_heat_budget(config):
    config["session"]["max_new_positions_per_day"] = 10
    config["risk"]["max_correlated_positions"] = 10
    reads = [
        read(f"C{i}/USDT", regime="trending_up", regime_confidence=0.9, adx=45,
             plus_di=40, minus_di=8, ema_fast=108, ema_slow=100, htf_trend=1,
             macd_hist=1.0, overnight_return_pct=2.0)
        for i in range(8)
    ]
    plan = planner(config).build(briefing(reads), 10_000, [], 10_000, 10_000, now=NOW)

    total_risk_pct = sum(t.risk_usd for t in plan.trades) / 10_000 * 100
    assert total_risk_pct <= config["risk"]["max_portfolio_heat_pct"] + 1e-6


def test_conviction_decides_the_ordering(config):
    config["session"]["max_new_positions_per_day"] = 1
    weak = read("WEAK/USDT", regime="trending_up", regime_confidence=0.7, adx=26,
                plus_di=24, minus_di=16, ema_fast=102, ema_slow=100, htf_trend=1)
    strong = read("STRONG/USDT", regime="trending_up", regime_confidence=0.95, adx=50,
                  plus_di=45, minus_di=6, ema_fast=112, ema_slow=100, htf_trend=1,
                  macd_hist=1.2, overnight_return_pct=2.5)

    plan = planner(config).build(briefing([weak, strong]), 10_000, [], 10_000, 10_000, now=NOW)
    assert plan.trades and plan.trades[0].symbol == "STRONG/USDT"


def test_planning_stops_when_a_breaker_has_fired(config):
    reads = [read(f"C{i}/USDT", regime="trending_up", regime_confidence=0.9, adx=45,
                  plus_di=40, minus_di=8, ema_fast=108, ema_slow=100, htf_trend=1)
             for i in range(3)]
    # Equity 3% below the day's open, against a 2% limit.
    plan = planner(config).build(briefing(reads), equity=9_700, open_positions=[],
                                 peak_equity=10_000, day_start_equity=10_000, now=NOW)
    assert plan.trades == []
    assert any("daily_loss_limit" in note for note in plan.notes)


def test_every_planned_trade_has_a_stop_on_the_right_side(config):
    reads = [
        read("UP/USDT", regime="trending_up", regime_confidence=0.9, adx=45,
             plus_di=40, minus_di=8, ema_fast=108, ema_slow=100, htf_trend=1),
        read("DOWN/USDT", regime="trending_down", regime_confidence=0.9, adx=45,
             plus_di=8, minus_di=40, ema_fast=92, ema_slow=100, htf_trend=-1),
    ]
    plan = planner(config).build(briefing(reads), 10_000, [], 10_000, 10_000, now=NOW)
    assert plan.trades
    for trade in plan.trades:
        if trade.side == "long":
            assert trade.stop_price < trade.entry_price < trade.take_profit
        else:
            assert trade.take_profit < trade.entry_price < trade.stop_price
        assert trade.risk_usd > 0
        assert trade.quantity > 0


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


def test_a_valid_config_passes(config):
    validate_config(config)
