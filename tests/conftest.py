"""Shared fixtures. Nothing here touches the network."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def config(tmp_path):
    """A complete config pointed at a temporary journal directory."""
    return {
        "exchange": {"name": "kraken", "market_type": "spot"},
        "data": {
            "symbols": ["BTC/USDT", "ETH/USDT", "SOL/USDT"],
            "timeframe": "1h",
            "higher_timeframe": "4h",
            "history_bars": 400,
        },
        "session": {
            "day_open": "00:00",
            "entry_window_minutes": 120,
            "flatten_at": "23:30",
            "manage_interval_seconds": 3600,
            "max_new_positions_per_day": 3,
            "carry_overnight": False,
            "max_holding_days": 3,
            "max_entry_drift_r": 0.5,
        },
        "risk": {
            "risk_per_trade_pct": 1.0,
            "min_risk_per_trade_pct": 0.1,
            "max_portfolio_heat_pct": 4.0,
            "max_open_positions": 5,
            "max_correlated_positions": 3,
            "max_net_beta": 1.5,
            "max_daily_loss_pct": 2.0,
            "daily_profit_lock_pct": 0,
            "max_total_drawdown_pct": 15.0,
            "derisk_after_losses": 3,
            "cooldown_after_losses": 2,
            "cooldown_minutes": 120,
        },
        "stops": {
            "atr_period": 14, "atr_stop_mult": 2.0, "target_r_multiple": 2.0,
            "breakeven_at_r": 1.0, "breakeven_pad_bps": 8,
            "trail_after_r": 1.0, "trail_atr_mult": 2.5,
            "time_stop_bars": 48, "min_stop_bps": 30, "max_stop_bps": 1200,
        },
        "sizing": {
            "max_position_pct": 20.0, "vol_target_annual_pct": 40.0,
            "min_position_usd": 25, "kelly_fraction": 0.25, "kelly_min_trades": 30,
        },
        "signals": {
            "min_edge_score": 0.18, "news_weight": 0.20, "book_weight": 0.08,
            "news_can_veto": True, "require_htf_agreement": True,
        },
        "filters": {
            "min_quote_volume_24h": 1_000_000, "max_spread_bps": 25,
            "min_atr_pct": 0.15, "max_atr_pct": 12.0, "min_bars": 120,
        },
        "news": {"enabled": False},
        "paper": {
            "initial_balance": 10_000.0, "fee_maker_bps": 2.0, "fee_taker_bps": 5.5,
            "slippage_bps": 3.0, "impact_coefficient": 0.5,
            "reference_daily_vol_bps": 300.0, "max_slippage_bps": 100.0,
            "funding_enabled": True, "default_funding_bps": 1.0,
        },
        "journal": {"dir": str(tmp_path / "state"), "rolling_window": 50},
        "logging": {"level": "CRITICAL", "log_to_file": False},
    }


def make_ohlcv(bars: int = 400, start_price: float = 100.0, drift: float = 0.0,
               vol: float = 0.01, seed: int = 0,
               start: datetime | None = None, freq_hours: int = 1) -> pd.DataFrame:
    """Deterministic synthetic OHLCV — no network, no flaky tests."""
    rng = np.random.default_rng(seed)
    start = start or datetime(2026, 1, 1, tzinfo=timezone.utc)
    index = pd.DatetimeIndex(
        [start + timedelta(hours=i * freq_hours) for i in range(bars)]
    )

    returns = rng.normal(drift, vol, bars)
    close = start_price * np.exp(np.cumsum(returns))
    open_ = np.concatenate([[start_price], close[:-1]])
    spread = np.abs(rng.normal(0, vol * 0.6, bars)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread
    volume = rng.uniform(80, 120, bars)

    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


@pytest.fixture
def ohlcv():
    return make_ohlcv()


class FakeExchange:
    """In-memory exchange serving synthetic bars, clipped to a clock."""

    def __init__(self, frames: dict[str, pd.DataFrame], clock=None,
                 htf: dict | None = None, volume: float = 50_000_000.0):
        self.frames = frames
        self.htf = htf or {}
        self.clock = clock
        self.volume = volume
        self.calls: list[str] = []

    def _clip(self, df):
        if self.clock is None:
            return df
        return df.loc[df.index <= self.clock.now()]

    def resolve_symbol(self, symbol):
        return symbol if symbol in self.frames else None

    def fetch_ohlcv(self, symbol, timeframe, limit=500, since=None):
        source = self.htf.get(symbol) if timeframe == "4h" and symbol in self.htf \
            else self.frames.get(symbol)
        if source is None:
            return pd.DataFrame()
        self.calls.append(f"ohlcv:{symbol}:{timeframe}")
        return self._clip(source).tail(limit)

    def get_current_price(self, symbol):
        df = self._clip(self.frames[symbol])
        if df.empty:
            raise LookupError(symbol)
        return float(df["close"].iloc[-1])

    def market_limits(self, symbol):
        return {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0}

    def quote_volume_24h(self, symbol):
        return self.volume

    def spread_bps(self, symbol):
        return 2.0

    def fetch_order_book(self, symbol, limit=20):
        price = self.get_current_price(symbol)
        return {
            "bids": [[price * (1 - i * 0.0001), 10.0] for i in range(1, limit + 1)],
            "asks": [[price * (1 + i * 0.0001), 10.0] for i in range(1, limit + 1)],
        }

    def fetch_funding_rate(self, symbol):
        return None
