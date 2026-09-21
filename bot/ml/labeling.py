"""Triple-barrier labelling and meta-labelling.

A model trained on "was the next bar up?" learns to predict a coin flip,
because that is what the next bar mostly is. It also ignores the fact that
a real position is closed by whichever of three things happens first: the
target, the stop, or the clock.

The triple-barrier method labels each bar by which barrier price touches
first, with the barriers set from the same ATR the live risk layer uses. A
label therefore means "a trade opened here, with this stop and this target,
would have won / lost / timed out" — the question the bot actually faces.

Meta-labelling then sits on top: a primary model picks the side, and a
secondary model predicts whether that call is worth taking. Trained on the
primary's own hit/miss record, it raises precision and gives position
sizing something calibrated to work with.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from bot.analysis.indicators import atr

logger = logging.getLogger("trading_bot")


def triple_barrier_labels(
    df: pd.DataFrame,
    atr_period: int = 14,
    upper_atr: float = 2.0,
    lower_atr: float = 2.0,
    max_holding_bars: int = 48,
    side: pd.Series | int = 1,
) -> pd.DataFrame:
    """Label every bar by which barrier is touched first.

    Args:
        df: OHLCV frame indexed by time.
        atr_period: ATR lookback for the barrier width.
        upper_atr: profit barrier, in ATR multiples.
        lower_atr: stop barrier, in ATR multiples.
        max_holding_bars: the vertical (time) barrier.
        side: +1 long, -1 short, or a Series of sides from a primary model.

    Returns a frame with:
        label       +1 profit barrier, -1 stop barrier, 0 timed out
        ret         realised return over the holding period
        bars_held   how long it took
        t1          index of the bar that closed the trade
        barrier_w   barrier half-width in price terms
        meta_label  1 when the trade made money, else 0 (for meta-labelling)
    """
    if len(df) < atr_period + 2:
        raise ValueError(f"need more than {atr_period + 2} bars to label")

    width = atr(df, atr_period)
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    widths = width.to_numpy(dtype=float)

    if isinstance(side, pd.Series):
        sides = side.reindex(df.index).fillna(0).to_numpy(dtype=float)
    else:
        sides = np.full(len(df), float(side))

    n = len(df)
    labels = np.zeros(n)
    rets = np.full(n, np.nan)
    bars = np.zeros(n, dtype=int)
    t1_idx = np.full(n, -1, dtype=int)

    for i in range(n):
        w = widths[i]
        direction = sides[i]
        if not np.isfinite(w) or w <= 0 or direction == 0:
            labels[i] = np.nan
            continue

        entry = close[i]
        if direction > 0:
            take, stop = entry + upper_atr * w, entry - lower_atr * w
        else:
            take, stop = entry - upper_atr * w, entry + lower_atr * w

        end = min(i + max_holding_bars, n - 1)
        outcome, exit_price, exit_at = 0, close[end], end

        for j in range(i + 1, end + 1):
            if direction > 0:
                # Stop first when a bar spans both: without tick data the
                # pessimistic read is the only defensible one, and it is the
                # same assumption the paper broker makes.
                if low[j] <= stop:
                    outcome, exit_price, exit_at = -1, stop, j
                    break
                if high[j] >= take:
                    outcome, exit_price, exit_at = 1, take, j
                    break
            else:
                if high[j] >= stop:
                    outcome, exit_price, exit_at = -1, stop, j
                    break
                if low[j] <= take:
                    outcome, exit_price, exit_at = 1, take, j
                    break

        labels[i] = outcome
        rets[i] = (exit_price - entry) / entry * direction
        bars[i] = exit_at - i
        t1_idx[i] = exit_at

    # The last max_holding_bars rows cannot resolve without future data.
    labels[max(0, n - max_holding_bars):] = np.nan

    out = pd.DataFrame(
        {
            "label": labels,
            "ret": rets,
            "bars_held": bars,
            "barrier_w": widths,
            "side": sides,
        },
        index=df.index,
    )
    out["t1"] = [df.index[k] if k >= 0 else pd.NaT for k in t1_idx]
    out["meta_label"] = (out["ret"] > 0).astype(float)
    out.loc[out["label"].isna(), "meta_label"] = np.nan

    resolved = out["label"].notna()
    if resolved.any():
        counts = out.loc[resolved, "label"].value_counts().to_dict()
        logger.info(
            "Triple-barrier labels: %d resolved | +1:%d  -1:%d  0:%d | "
            "median hold %.0f bars",
            int(resolved.sum()), int(counts.get(1.0, 0)), int(counts.get(-1.0, 0)),
            int(counts.get(0.0, 0)), float(out.loc[resolved, "bars_held"].median()),
        )
    return out


def sample_weights_by_uniqueness(labels: pd.DataFrame) -> pd.Series:
    """Down-weight observations whose holding periods overlap.

    Two labels that span the same bars are not two independent facts. Left
    unweighted, long-lived labels are effectively counted many times and the
    model over-fits whatever happened during them.
    """
    index = labels.index
    resolved = labels["t1"].notna()
    if not resolved.any():
        return pd.Series(1.0, index=index)

    position = pd.Series(np.arange(len(index)), index=index)
    concurrency = np.zeros(len(index))

    for start, end in labels.loc[resolved, "t1"].items():
        i, j = position.get(start), position.get(end)
        if i is None or j is None:
            continue
        concurrency[int(i):int(j) + 1] += 1

    concurrency[concurrency == 0] = 1.0

    weights = np.ones(len(index))
    for k, (start, end) in enumerate(labels.loc[resolved, "t1"].items()):
        i, j = position.get(start), position.get(end)
        if i is None or j is None:
            continue
        span = slice(int(i), int(j) + 1)
        weights[int(i)] = float(np.mean(1.0 / concurrency[span]))

    series = pd.Series(weights, index=index)
    series[~resolved] = 0.0
    total = series.sum()
    # Normalise to mean 1 so learning rates stay comparable.
    return series * (len(series) / total) if total > 0 else series


def primary_side_from_signal(edge: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Turn a continuous signal into the sides a meta-model will judge."""
    side = pd.Series(0.0, index=edge.index)
    side[edge > threshold] = 1.0
    side[edge < -threshold] = -1.0
    return side


def label_balance(labels: pd.DataFrame) -> dict:
    """Class balance, for spotting a degenerate labelling setup early."""
    resolved = labels["label"].dropna()
    if resolved.empty:
        return {"resolved": 0}
    counts = resolved.value_counts()
    total = len(resolved)
    return {
        "resolved": int(total),
        "profit_pct": round(float(counts.get(1.0, 0)) / total * 100, 2),
        "stop_pct": round(float(counts.get(-1.0, 0)) / total * 100, 2),
        "timeout_pct": round(float(counts.get(0.0, 0)) / total * 100, 2),
        "mean_ret": round(float(labels["ret"].dropna().mean()), 6),
        "median_bars": int(labels.loc[resolved.index, "bars_held"].median()),
    }
