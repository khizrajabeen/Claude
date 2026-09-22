"""Indicator primitives used by the daily briefing and plan.

Deliberately dependency-light (pandas/numpy only) so the daily session runs
without the ML stack installed. Wilder's smoothing is used where Wilder
defined the indicator, because the EMA shortcut gives materially different
ADX and ATR values and those feed position size.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Bars per year, by timeframe — used to annualise realised volatility.
BARS_PER_YEAR = {
    "1m": 525_600, "3m": 175_200, "5m": 105_120, "15m": 35_040,
    "30m": 17_520, "1h": 8_760, "2h": 4_380, "4h": 2_190,
    "6h": 1_460, "12h": 730, "1d": 365,
}

TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "12h": 43200, "1d": 86400,
}


def wilder(series: pd.Series, period: int) -> pd.Series:
    """Wilder's smoothing — an EMA with alpha = 1/period."""
    return series.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs(),
    ], axis=1).max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    return wilder(true_range(df), period)


def adx(df: pd.DataFrame, period: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Wilder's ADX with +DI/-DI."""
    up = df["high"].diff()
    down = -df["low"].diff()

    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    tr_smooth = wilder(true_range(df), period)
    plus_di = 100 * wilder(pd.Series(plus_dm, index=df.index), period) / tr_smooth.replace(0, np.nan)
    minus_di = 100 * wilder(pd.Series(minus_dm, index=df.index), period) / tr_smooth.replace(0, np.nan)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, period), plus_di, minus_di


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = wilder(delta.clip(lower=0), period)
    loss = wilder((-delta).clip(lower=0), period)
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False, min_periods=span).mean()


def macd_histogram(close: pd.Series, fast: int = 12, slow: int = 26,
                   signal: int = 9) -> pd.Series:
    line = ema(close, fast) - ema(close, slow)
    return line - line.ewm(span=signal, adjust=False).mean()


def realized_vol(close: pd.Series, window: int = 20, timeframe: str = "1h") -> pd.Series:
    """Annualised realised volatility as a decimal (0.6 = 60%/yr)."""
    returns = np.log(close / close.shift(1))
    periods = BARS_PER_YEAR.get(timeframe, 8_760)
    return returns.rolling(window).std() * np.sqrt(periods)


def zscore(close: pd.Series, window: int = 20) -> pd.Series:
    mean = close.rolling(window).mean()
    std = close.rolling(window).std()
    return (close - mean) / std.replace(0, np.nan)


def donchian_position(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Where price sits in its recent range: 0 at the low, 1 at the high."""
    high = df["high"].rolling(window).max()
    low = df["low"].rolling(window).min()
    return (df["close"] - low) / (high - low).replace(0, np.nan)


def last_value(series: pd.Series, default: float = 0.0) -> float:
    """Last finite value of a series, or a default — indicator warm-up
    leaves NaNs that would otherwise propagate into sizing."""
    if series is None or len(series) == 0:
        return default
    clean = series.replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return default
    value = float(clean.iloc[-1])
    return value if np.isfinite(value) else default


def orderbook_imbalance(orderbook: dict, levels: int = 10) -> dict:
    """Resting-liquidity imbalance at the top of the book.

    Returns imbalance in [-1, 1] (positive = more bid size), the spread in
    basis points, and the depth on each side. It is a weak, fast-decaying
    signal — useful to confirm or fade an entry, not to generate one.
    """
    bids = (orderbook or {}).get("bids") or []
    asks = (orderbook or {}).get("asks") or []
    if not bids or not asks:
        return {"imbalance": 0.0, "spread_bps": 0.0, "bid_depth": 0.0, "ask_depth": 0.0}

    bid_depth = float(sum(level[1] * level[0] for level in bids[:levels]))
    ask_depth = float(sum(level[1] * level[0] for level in asks[:levels]))
    total = bid_depth + ask_depth

    best_bid, best_ask = float(bids[0][0]), float(asks[0][0])
    mid = (best_bid + best_ask) / 2

    return {
        "imbalance": round((bid_depth - ask_depth) / total, 4) if total > 0 else 0.0,
        "spread_bps": round((best_ask - best_bid) / mid * 10_000, 4) if mid > 0 else 0.0,
        "bid_depth": round(bid_depth, 2),
        "ask_depth": round(ask_depth, 2),
    }


# ── Structure primitives ─────────────────────────────────────

def swing_points(df: pd.DataFrame, length: int = 10) -> tuple[pd.Series, pd.Series]:
    """Confirmed swing highs and lows.

    A bar is a swing high when it is the highest of the `length` bars either
    side of it. Confirmation therefore arrives `length` bars *late*, and the
    series is shifted to reflect that: a swing point you could not have
    known about yet is look-ahead bias, and it is the single most common
    way structure-based indicators flatter themselves.
    """
    highs = df["high"]
    lows = df["low"]

    is_high = highs == highs.rolling(length * 2 + 1, center=True).max()
    is_low = lows == lows.rolling(length * 2 + 1, center=True).min()

    # Shift forward by `length`: the point is only knowable once the right
    # shoulder has printed.
    swing_high = highs.where(is_high).shift(length)
    swing_low = lows.where(is_low).shift(length)
    return swing_high.ffill(), swing_low.ffill()


def supertrend(df: pd.DataFrame, period: int = 10, factor: float = 3.0
               ) -> tuple[pd.Series, pd.Series]:
    """SuperTrend trailing stop. Returns (line, direction).

    Direction is +1 when price is above the stop (long) and -1 below. The
    band only ever ratchets in the direction of the trend, which is what
    makes it a trailing stop rather than a channel.
    """
    atr_series = atr(df, period)
    hl2 = (df["high"] + df["low"]) / 2
    upper = hl2 + factor * atr_series
    lower = hl2 - factor * atr_series

    close = df["close"].to_numpy(dtype=float)
    upper_arr = upper.to_numpy(dtype=float)
    lower_arr = lower.to_numpy(dtype=float)

    line = np.full(len(df), np.nan)
    direction = np.zeros(len(df), dtype=int)

    prev_line = np.nan
    prev_dir = 1
    for i in range(len(df)):
        up, low, price = upper_arr[i], lower_arr[i], close[i]
        if not np.isfinite(up) or not np.isfinite(low):
            continue

        if np.isnan(prev_line):
            prev_line, prev_dir = low, 1
        elif prev_dir == 1:
            # Rising stop: never let it fall back.
            low = max(low, prev_line)
            if price < low:
                prev_dir, prev_line = -1, up
            else:
                prev_line = low
        else:
            up = min(up, prev_line)
            if price > up:
                prev_dir, prev_line = 1, low
            else:
                prev_line = up

        line[i] = prev_line
        direction[i] = prev_dir

    return (pd.Series(line, index=df.index),
            pd.Series(direction, index=df.index))


def gaussian_kernel_regression(series: pd.Series, bandwidth: float = 8.0,
                               window: int = 200) -> pd.Series:
    """Endpoint Nadaraya-Watson estimate — the non-repainting form.

    The usual implementation smooths across the whole window and is
    recalculated every bar, so the fit at any past point keeps changing as
    new data arrives. That repaints: a backtest reading it is using
    information from the future.

    This evaluates the kernel only at the *last* bar of each window, using
    weights over bars already printed. The line is choppier than the
    repainting version. It is also the only one that could have been traded.
    """
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)

    offsets = np.arange(window)
    weights = np.exp(-(offsets ** 2) / (2 * bandwidth ** 2))
    weight_sum = weights.sum()
    if weight_sum <= 0:
        return pd.Series(out, index=series.index)

    for i in range(window - 1, n):
        # offsets[0] is the current bar, and weight falls with age.
        segment = values[i - window + 1 : i + 1][::-1]
        out[i] = float(np.dot(segment, weights) / weight_sum)

    return pd.Series(out, index=series.index)


def fair_value_gaps(df: pd.DataFrame, min_atr: float = 0.25
                    ) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Three-bar imbalances where price skipped a range.

    A bullish gap is where bar i-2's high sits below bar i's low: no trading
    happened in between. Returns (direction, top, bottom), aligned to the
    bar on which the gap becomes visible.
    """
    high, low = df["high"], df["low"]
    atr_series = atr(df, 14)

    prev_high = high.shift(2)
    prev_low = low.shift(2)

    bullish = low > prev_high
    bearish = high < prev_low

    size = pd.Series(np.nan, index=df.index)
    size[bullish] = (low - prev_high)[bullish]
    size[bearish] = (prev_low - high)[bearish]

    # Ignore gaps too small to matter relative to the asset's own range.
    significant = size >= (atr_series * min_atr)

    direction = pd.Series(0, index=df.index, dtype=int)
    direction[bullish & significant] = 1
    direction[bearish & significant] = -1

    top = pd.Series(np.nan, index=df.index)
    bottom = pd.Series(np.nan, index=df.index)
    top[bullish & significant] = low[bullish & significant]
    bottom[bullish & significant] = prev_high[bullish & significant]
    top[bearish & significant] = prev_low[bearish & significant]
    bottom[bearish & significant] = high[bearish & significant]

    return direction, top, bottom
