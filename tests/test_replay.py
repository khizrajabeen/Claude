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
        frames["BTC/USDT"]["1h"].index <= clock.now(), "close"
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
