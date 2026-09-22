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
            "history_bars": 900,
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
            "min_edge_cost_ratio": 3.0,
            "contrarian_risk_factor": 0.7,
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
            "min_edge_score": 0.15, "news_weight": 0.15, "book_weight": 0.06,
            "news_can_veto": True, "require_htf_agreement": True,
        },
        "filters": {
            "min_quote_volume_24h": 1_000_000, "max_spread_bps": 25,
            "min_atr_pct": 0.15, "max_atr_pct": 12.0, "min_bars": 120,
        },
        "strategies": {
            "enabled": ["clenow", "turtle", "holygrail",
                        "supertrend", "smc", "nwenvelope", "lorentzian"],
            "clenow": {"fast_ema": 50, "slow_ema": 100, "breakout_bars": 100},
            "turtle": {"entry_s1": 20, "entry_s2": 55, "n_period": 20},
            "holygrail": {"adx_floor": 30.0, "ema_period": 20},
            "supertrend": {"atr_period": 10, "performance_window": 200},
            "smc": {"swing_length": 10, "dealing_range_bars": 60},
            "nwenvelope": {"bandwidth": 8.0, "window": 200},
            "lorentzian": {"neighbours": 8, "horizon": 4, "history_bars": 500},
        },
        "portfolio": {
            "weighting": "inverse_vol", "correlation_haircut": True,
            "max_strategy_weight": 0.40, "min_strategy_weight": 0.05,
            "lookback_days": 60, "min_history_days": 10,
        },
        "vol_target": {
            "enabled": True, "target_annual_pct": 15.0, "min_scale": 0.25,
            "max_scale": 1.5, "window_days": 20, "min_days": 8,
            "cut_speed": 1.0, "restore_speed": 0.25,
            "drawdown_throttle": True, "throttle_start_pct": 4.0,
            "throttle_floor": 0.35,
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

    def fetch_ohlcv_paged(self, symbol, timeframe, bars):
        return self.fetch_ohlcv(symbol, timeframe, limit=bars)

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


class FakeRouter:
    """A DataRouter over synthetic frames, keyed by instrument symbol.

    The session and briefing only ever see the router interface, so this is
    the single seam a test needs to control every market — crypto or equity —
    without a network call.
    """

    def __init__(self, frames: dict[str, pd.DataFrame], clock=None,
                 htf: dict | None = None, volume: float = 50_000_000.0,
                 funding: dict | None = None):
        self.exchange = FakeExchange(frames, clock=clock, htf=htf, volume=volume)
        self.frames = frames
        self.funding = funding or {}
        self.calls = self.exchange.calls

    # ── Router interface ──────────────────────────────────────

    def provider_for(self, instrument):
        return self.exchange if instrument.symbol in self.frames else None

    def bars(self, instrument, timeframe=None, limit=500):
        tf = timeframe or instrument.timeframe
        return self.exchange.fetch_ohlcv(instrument.symbol, tf, limit=limit)

    def price(self, instrument):
        try:
            return self.exchange.get_current_price(instrument.symbol)
        except (LookupError, KeyError):
            return None

    def prices(self, instruments):
        out = {}
        for instrument in instruments:
            price = self.price(instrument)
            if price is not None:
                out[instrument.symbol] = price
        return out

    def quote_volume(self, instrument):
        return self.exchange.quote_volume_24h(instrument.symbol)

    def spread_bps(self, instrument):
        return self.exchange.spread_bps(instrument.symbol)

    def order_book(self, instrument, depth: int = 20):
        try:
            return self.exchange.fetch_order_book(instrument.symbol, limit=depth)
        except (LookupError, KeyError):
            return None

    def funding_rate(self, instrument):
        return self.funding.get(instrument.symbol)

    def market_limits(self, instrument):
        return self.exchange.market_limits(instrument.symbol)


def build_context(symbols=("BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"),
                  bars=600, drifts=None, vol=0.012, seed=0, funding=None,
                  htf_trends=None, day="2026-05-04", **read_overrides):
    """A MarketContext over synthetic markets, with reads derived from them.

    Reads are computed from the same bars the strategies see, so a test
    cannot accidentally describe a market the price series contradicts.
    """
    from bot.analysis import indicators as ind
    from bot.daily.briefing import SymbolRead
    from bot.strategies.base import MarketContext

    symbols = list(symbols)
    drifts = drifts if drifts is not None else [0.004] * len(symbols)
    htf_trends = htf_trends if htf_trends is not None else [0] * len(symbols)

    frames, reads = {}, {}
    for i, symbol in enumerate(symbols):
        df = make_ohlcv(bars=bars, start_price=100 * (i + 1),
                        drift=drifts[i % len(drifts)], vol=vol, seed=seed + i)
        frames[symbol] = df
        price = float(df["close"].iloc[-1])
        atr = ind.last_value(ind.atr(df, 14))
        adx_s, plus_di, minus_di = ind.adx(df, 14)
        defaults = dict(
            symbol=symbol, price=price, atr=atr,
            atr_pct=atr / price * 100 if price else 0.0,
            annualized_vol=ind.last_value(ind.realized_vol(df["close"], 24, "1h"), 0.6),
            adx=ind.last_value(adx_s, 20.0),
            plus_di=ind.last_value(plus_di), minus_di=ind.last_value(minus_di),
            rsi=ind.last_value(ind.rsi(df["close"]), 50.0),
            zscore=ind.last_value(ind.zscore(df["close"], 20)),
            donchian=ind.last_value(ind.donchian_position(df, 20), 0.5),
            macd_hist=ind.last_value(ind.macd_histogram(df["close"])),
            ema_fast=ind.last_value(ind.ema(df["close"], 21), price),
            ema_slow=ind.last_value(ind.ema(df["close"], 55), price),
            htf_trend=htf_trends[i % len(htf_trends)],
            regime="trending_up" if drifts[i % len(drifts)] > 0 else "trending_down",
            regime_confidence=0.8, quote_volume_24h=5e7, spread_bps=2.0,
            tradable=True, bars=len(df),
        )
        defaults.update(read_overrides)
        reads[symbol] = SymbolRead(**defaults)

    return MarketContext(day=day, reads=reads, frames=frames,
                         funding=funding or {}, equity=10_000.0, timeframe="1h")
