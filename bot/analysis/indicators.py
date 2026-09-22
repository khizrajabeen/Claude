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
    "12h": 43200, "1d": 86400, "1w": 604800,
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


def supertrend(df: pd.DataFrame, period: int = 10, factor: float = 3.0,
               atr_series: pd.Series | None = None
               ) -> tuple[pd.Series, pd.Series]:
    """SuperTrend trailing stop. Returns (line, direction).

    Direction is +1 when price is above the stop (long) and -1 below. The
    band only ever ratchets in the direction of the trend, which is what
    makes it a trailing stop rather than a channel.

    `atr_series` lets a caller that evaluates many factors at once compute
    the ATR once and pass it in. SuperTrend AI runs nine factors over the
    same bars and the ATR does not depend on the factor at all, so
    recomputing it nine times was eight ninths of that work wasted.
    """
    if atr_series is None:
        atr_series = atr(df, period)
    hl2 = (df["high"].to_numpy(dtype=float) + df["low"].to_numpy(dtype=float)) / 2.0
    atr_arr = atr_series.to_numpy(dtype=float)
    band = factor * atr_arr

    close = df["close"].to_numpy(dtype=float)
    line, direction = _supertrend_ratchet(close, hl2 + band, hl2 - band)
    return (pd.Series(line, index=df.index),
            pd.Series(direction, index=df.index))


def _supertrend_ratchet(close: np.ndarray, upper: np.ndarray, lower: np.ndarray
                        ) -> tuple[np.ndarray, np.ndarray]:
    """The sequential half of SuperTrend, kept as cheap as possible.

    The ratchet is a state machine — each bar's stop depends on the last —
    so it cannot be vectorised. What it can do is stop paying numpy's
    per-element overhead: the loop runs over Python floats from `tolist()`,
    which is several times faster than indexing arrays one element at a
    time, and skips straight to the first bar where the ATR has warmed up
    instead of testing every bar for NaN.
    """
    n = len(close)
    line = np.full(n, np.nan)
    direction = np.zeros(n, dtype=int)

    valid = np.isfinite(upper) & np.isfinite(lower)
    if not valid.any():
        return line, direction
    start = int(valid.argmax())

    closes = close.tolist()
    uppers = upper.tolist()
    lowers = lower.tolist()
    out_line = line.tolist()
    out_dir = direction.tolist()

    prev_line = lowers[start]
    prev_dir = 1
    out_line[start], out_dir[start] = prev_line, prev_dir

    for i in range(start + 1, n):
        up, low, price = uppers[i], lowers[i], closes[i]
        if up != up or low != low:      # NaN, without the numpy call
            continue                    # leaves nan/0, as a warm-up bar should

        if prev_dir == 1:
            # Rising stop: never let it fall back.
            if low < prev_line:
                low = prev_line
            if price < low:
                prev_dir, prev_line = -1, up
            else:
                prev_line = low
        else:
            if up > prev_line:
                up = prev_line
            if price > up:
                prev_dir, prev_line = 1, low
            else:
                prev_line = up

        out_line[i], out_dir[i] = prev_line, prev_dir

    return np.asarray(out_line, dtype=float), np.asarray(out_dir, dtype=int)


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
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    atr_arr = atr(df, 14).to_numpy(dtype=float)

    # Masked assignment into a Series goes through pandas' block manager
    # for every write; there are eight of them here, and this runs once
    # per symbol per pass. np.where does the same job in one pass over
    # the array with no index bookkeeping.
    prev_high = np.roll(high, 2)
    prev_low = np.roll(low, 2)
    prev_high[:2] = np.nan
    prev_low[:2] = np.nan

    bullish = low > prev_high
    bearish = high < prev_low

    size = np.where(bullish, low - prev_high,
                    np.where(bearish, prev_low - high, np.nan))
    # Ignore gaps too small to matter relative to the asset's own range.
    with np.errstate(invalid="ignore"):
        significant = size >= (atr_arr * min_atr)

    up = bullish & significant
    down = bearish & significant

    direction = np.where(up, 1, np.where(down, -1, 0)).astype(int)
    top = np.where(up, low, np.where(down, prev_low, np.nan))
    bottom = np.where(up, prev_high, np.where(down, high, np.nan))

    return (pd.Series(direction, index=df.index),
            pd.Series(top, index=df.index),
            pd.Series(bottom, index=df.index))


def cci(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Commodity Channel Index.

    Mean absolute deviation is approximated by the standard deviation times
    sqrt(2/pi) — the exact version needs a Python callback per window and is
    roughly fifty times slower for a difference that does not survive the
    next normalisation step.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    mean = typical.rolling(period).mean()
    deviation = typical.rolling(period).std() * np.sqrt(2 / np.pi)
    return (typical - mean) / (0.015 * deviation.replace(0, np.nan))


def wave_trend(df: pd.DataFrame, channel: int = 10, average: int = 11) -> pd.Series:
    """WaveTrend oscillator — the LazyBear formulation.

    An EMA of how far the typical price sits from its own EMA, scaled by
    the average absolute deviation. Used as a feature by several published
    classifiers, including Lorentzian Classification.
    """
    typical = (df["high"] + df["low"] + df["close"]) / 3
    esa = typical.ewm(span=channel, adjust=False).mean()
    deviation = (typical - esa).abs().ewm(span=channel, adjust=False).mean()
    # The 0.015 scaling is Lambert's, carried over from CCI.
    ci = (typical - esa) / (0.015 * deviation.replace(0, np.nan))
    return ci.ewm(span=average, adjust=False).mean()


def normalise(series: pd.Series, window: int = 200,
              lower: float = 0.0, upper: float = 1.0) -> pd.Series:
    """Rescale a series into [lower, upper] using a *trailing* window.

    Rescaling against the whole series would leak: the minimum three months
    from now would be setting today's value. This uses only bars already
    printed, which is the difference between a feature and a peek.
    """
    rolling_min = series.rolling(window, min_periods=window // 4).min()
    rolling_max = series.rolling(window, min_periods=window // 4).max()

    # The arithmetic runs on the arrays rather than the Series. Each
    # pandas operator here builds a fresh Series with its own index
    # alignment; the Lorentzian feature set calls this five times per
    # symbol per pass, and that overhead was the single largest line in
    # its profile. The result is identical — same index, same values.
    low_arr = rolling_min.to_numpy(dtype=float)
    span = rolling_max.to_numpy(dtype=float) - low_arr
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = (series.to_numpy(dtype=float) - low_arr) / np.where(span == 0, np.nan, span)
    scaled = np.clip(scaled, 0.0, 1.0) * (upper - lower) + lower
    return pd.Series(scaled, index=series.index)


def lorentzian_distance(current: np.ndarray, history: np.ndarray) -> np.ndarray:
    """Distance from one feature vector to many, under the Lorentzian metric.

        d = sum(log(1 + |x_i - y_i|))

    The logarithm is the whole point. Euclidean distance lets one wildly
    different feature dominate, which in markets means a single volatility
    spike decides who counts as a neighbour. Compressing each dimension
    keeps outliers from swamping the comparison — the same reason the
    metric is used where space is warped.
    """
    return np.log1p(np.abs(history - current)).sum(axis=1)
