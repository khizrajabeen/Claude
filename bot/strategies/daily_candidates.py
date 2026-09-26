"""Two predeclared long-only daily candidates, shared by research and paper.

Input timestamps are candle OPENS. Callers must remove unclosed candles.
Output is desired exposure after each completed close, not a same-bar fill.
These are experiments, not claims of profitability.
"""
import numpy as np
import pandas as pd

CANDIDATES = ("breakout_55_20", "momentum_90_200")
VERSION = "daily-v1"


def validate_bars(frame):
    required = {"open", "high", "low", "close", "volume"}
    if not required <= set(frame.columns) or frame.empty:
        raise ValueError("Missing OHLCV data")
    if frame.index.tz is None or not frame.index.is_unique or not frame.index.is_monotonic_increasing:
        raise ValueError("Bars must have unique, ascending timezone-aware timestamps")
    if not np.isfinite(frame[list(required)].to_numpy()).all():
        raise ValueError("Non-finite OHLCV")
    if (frame[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive prices")
    if ((frame.high < frame[["open", "close", "low"]].max(axis=1)) |
            (frame.low > frame[["open", "close", "high"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC")
    if len(frame) > 1 and not (frame.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all():
        raise ValueError("Daily crypto history contains missing bars")


def exposure(frame, candidate):
    validate_bars(frame)
    if candidate not in CANDIDATES:
        raise ValueError(f"Unknown candidate {candidate}")
    if candidate == "momentum_90_200":
        return ((frame.close > frame.close.shift(90)) &
                (frame.close > frame.close.rolling(200).mean())).astype(int)
    upper = frame.high.shift(1).rolling(55).max()
    lower = frame.low.shift(1).rolling(20).min()
    held = False
    values = []
    for close, hi, lo in zip(frame.close, upper, lower):
        if not held and close > hi:
            held = True
        elif held and close < lo:
            held = False
        values.append(int(held))
    return pd.Series(values, index=frame.index, name=candidate)


def completed_bars(frame, now):
    return frame.loc[frame.index + pd.Timedelta(days=1) <= pd.Timestamp(now)]
