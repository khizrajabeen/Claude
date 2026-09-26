"""Historical replay.

The property that matters: the replay must run the live session's code and
must never let it see a bar that had not printed yet.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from bot.daily.session import SimulatedClock
from bot.utils.backtester import DailyReplay, ReplayExchange, _replay_config
from bot.utils.metrics import summarize_trades
from tests.conftest import make_ohlcv

START = datetime(2026, 2, 1, tzinfo=timezone.utc)


@pytest.fixture
def frames():
    out = {}
    for i, symbol in enumerate(["BTC/USDT", "ETH/USDT", "SOL/USDT"]):
        out[symbol] = {
            "1h": make_ohlcv(bars=1200, start_price=100 * (i + 1), drift=0.0020,
                             vol=0.010, seed=10 + i, start=START),
            "4h": make_ohlcv(bars=300, start_price=100 * (i + 1), drift=0.0080,
                             vol=0.020, seed=10 + i, start=START, freq_hours=4),
        }
    return out


def test_replay_exchange_never_shows_the_future(frames):
    clock = SimulatedClock(START + timedelta(hours=500))
    exchange = ReplayExchange(frames, clock)

    visible = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=1000)
    assert visible.index.max() <= clock.now()

    clock.advance(24 * 3600)
    later = exchange.fetch_ohlcv("BTC/USDT", "1h", limit=1000)
    assert later.index.max() > visible.index.max()
    assert later.index.max() <= clock.now()


def test_replay_price_is_the_last_visible_close(frames):
    clock = SimulatedClock(START + timedelta(hours=500))
    exchange = ReplayExchange(frames, clock)
    expected = frames["BTC/USDT"]["1h"].loc[
        frames["BTC/USDT"]["1h"].index <= clock.now() - timedelta(hours=1), "close"
    ].iloc[-1]
    assert exchange.get_current_price("BTC/USDT") == pytest.approx(expected)


def test_replay_disables_news_and_isolates_the_journal(config):
    replayed = _replay_config(config)
    assert replayed["news"]["enabled"] is False
    assert replayed["journal"]["dir"] != config["journal"]["dir"], (
        "a replay must never overwrite live records"
    )


def test_replay_runs_days_and_produces_a_report(config, frames):
    config["data"]["symbols"] = list(frames)
    replay = DailyReplay(config)
    report = replay.run(days=5, frames=frames)

    assert report["days"] == 5
    assert report["starting_equity"] == config["paper"]["initial_balance"]
    assert len(report["daily"]) == 5
    assert report["caveats"], "limitations must be stated in the report"

    equities = [d["ending_equity"] for d in report["daily"]]
    assert all(e > 0 for e in equities)


def test_replay_days_chain_their_equity(config, frames):
    config["data"]["symbols"] = list(frames)
    report = DailyReplay(config).run(days=6, frames=frames)

    daily = report["daily"]
    for previous, following in zip(daily, daily[1:]):
        assert following["starting_equity"] == pytest.approx(
            previous["ending_equity"], rel=1e-6
        ), "each day must open where the last one closed"


def test_replay_trades_are_recorded_with_r_multiples(config, frames):
    config["data"]["symbols"] = list(frames)
    report = DailyReplay(config).run(days=8, frames=frames)

    summary = report["trades"]
    assert summary.get("total_trades"), "a trending synthetic market must produce trades"
    assert summary["wins"] + summary["losses"] == summary["total_trades"]
    assert "exit_reasons" in summary
    assert summary["total_fees"] >= 0


def test_drawdown_is_measured_against_equity(config):
    """Guards the '431% drawdown' bug: cumulative PnL is the wrong base."""
    from bot.trading.models import Trade

    def trade(pnl):
        return Trade(
            id="x", symbol="BTC/USDT", side="long", entry_price=100, exit_price=101,
            quantity=1, leverage=1, opened_at=START, closed_at=START,
            pnl=pnl, pnl_pct=pnl, r_multiple=pnl / 50, fees=0, funding=0,
            slippage_cost=0, exit_reason="signal", risk_usd=50,
        )

    trades = [trade(100), trade(-300), trade(50)]
    summary = summarize_trades(trades, starting_equity=10_000)
    assert 0 <= summary["max_drawdown_pct"] <= 100
    assert summary["max_drawdown_pct"] < 5


# ── Statistical significance ─────────────────────────────────

def make_trades(n, mean_r, sd, seed):
    from datetime import timedelta

    import numpy as np

    from bot.trading.models import Trade

    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        r = float(rng.normal(mean_r, sd))
        out.append(Trade(
            id=str(i), symbol="BTC/USDT", side="long", entry_price=100,
            exit_price=100 + r, quantity=1, leverage=1,
            opened_at=START + timedelta(hours=i),
            closed_at=START + timedelta(hours=i + 1),
            pnl=r * 50, pnl_pct=r, r_multiple=r, fees=0.5, funding=0,
            slippage_cost=0, exit_reason="signal", risk_usd=50,
            closed_on_day=(START + timedelta(days=i // 3)).date().isoformat(),
        ))
    return out


def test_a_great_looking_result_on_few_trades_is_not_significant():
    """The trap the bench exists to expose: a Sharpe of 13 on 13 trades is
    a small sample, not a good strategy."""
    summary = summarize_trades(make_trades(14, 0.35, 1.4, 3), starting_equity=10_000)
    assert summary["avg_r"] > 0
    assert not summary["significant"], "14 trades cannot establish an edge"
    assert summary["trades_for_significance"] > 14


def test_a_genuine_edge_over_a_real_sample_is_significant():
    summary = summarize_trades(make_trades(500, 0.25, 1.2, 4), starting_equity=10_000)
    assert summary["significant"]
    assert summary["t_stat"] > 2


def test_standard_error_shrinks_with_the_square_root_of_trades():
    small = summarize_trades(make_trades(50, 0.1, 1.5, 5), starting_equity=10_000)
    large = summarize_trades(make_trades(800, 0.1, 1.5, 5), starting_equity=10_000)
    assert large["expectancy_se"] < small["expectancy_se"]
    # Sixteen times the trades should roughly quarter the error.
    assert large["expectancy_se"] == pytest.approx(small["expectancy_se"] / 4, rel=0.4)


def test_a_flat_strategy_is_reported_as_not_significant():
    summary = summarize_trades(make_trades(300, 0.0, 1.3, 6), starting_equity=10_000)
    assert not summary["significant"]
    assert abs(summary["t_stat"]) < 2


def test_single_trade_does_not_crash_the_statistics():
    summary = summarize_trades(make_trades(1, 0.5, 1.0, 7), starting_equity=10_000)
    assert summary["total_trades"] == 1
    assert summary["expectancy_se"] == 0.0
    assert summary["significant"] is False


# ── Warm-up is capped by what the feed can supply ────────────
# The deepest strategy wants 724 bars: 30 calendar days of a 1-hour
# crypto series but 1,048 of an equity daily one, against a feed that
# serves about 1,150 days in total. Demanding the full warm-up left the
# equity leg with roughly a hundred replayable days, and any study that
# shortened the history produced zero trades on it.

def test_warmup_is_capped_so_there_is_something_left_to_replay(config):
    from bot.markets.instrument import build_instrument
    from bot.utils.backtester import DailyReplay

    replay = DailyReplay(config)
    stock = build_instrument({"symbol": "NVDA", "asset_class": "equity"})

    uncapped = replay._warmup_span(stock)
    capped = replay._warmup_span(stock, available_bars=300, days=60)
    assert capped < uncapped
    assert capped.days >= 0


def test_a_generous_feed_keeps_the_full_warmup(config):
    from bot.markets.instrument import build_instrument
    from bot.utils.backtester import DailyReplay

    replay = DailyReplay(config)
    perp = build_instrument({"symbol": "BTC/USDT:USDT",
                             "asset_class": "crypto_perp"})
    full = replay._warmup_span(perp)
    plenty = replay._warmup_span(perp, available_bars=20_000, days=60)
    assert plenty == full


def test_crypto_needs_far_less_calendar_warmup_than_equities(config):
    """The same 724 bars is a month of hourly bars and three years of
    daily ones."""
    from bot.markets.instrument import build_instrument
    from bot.utils.backtester import DailyReplay

    replay = DailyReplay(config)
    perp = build_instrument({"symbol": "BTC/USDT:USDT",
                             "asset_class": "crypto_perp"})
    stock = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
    assert replay._warmup_span(stock) > replay._warmup_span(perp) * 10


def test_starved_strategies_are_named_not_left_to_be_inferred(config, caplog):
    """A strategy that cannot be warmed declines silently, which looks
    exactly like having had no opportunity."""
    import logging

    from bot.markets.instrument import build_instrument
    from bot.utils.backtester import DailyReplay

    replay = DailyReplay(config)
    stock = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
    with caplog.at_level(logging.WARNING, logger="trading_bot"):
        replay._warmup_span(stock, available_bars=300, days=60)
    assert any("cannot be warmed" in r.message for r in caplog.records)


def test_each_instrument_is_only_warned_about_once(config, caplog):
    import logging

    from bot.markets.instrument import build_instrument
    from bot.utils.backtester import DailyReplay

    replay = DailyReplay(config)
    stock = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
    with caplog.at_level(logging.WARNING, logger="trading_bot"):
        for _ in range(4):
            replay._warmup_span(stock, available_bars=300, days=60)
    warnings = [r for r in caplog.records if "cannot be warmed" in r.message]
    assert len(warnings) == 1
