"""No indicator may see the future.

A backtest is only evidence if a value computed at time t is the value
that would have existed at time t. The failure is quiet: an indicator
that peeks forward shifts every signal slightly earlier, which reads as
skill and cannot be reproduced live.

The test is mechanical. Compute an indicator over a full series, then
over the same series truncated at t, and compare the values at t. They
must be identical. Anything centred, back-filled, or fitted over the
whole sample fails here by construction.
"""

import numpy as np
import pandas as pd
import pytest

from bot.analysis import indicators as ind


def frame(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """A random walk with enough structure to move every indicator."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    high = close * (1 + abs(rng.normal(0, 0.004, n)))
    low = close * (1 - abs(rng.normal(0, 0.004, n)))
    open_ = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": rng.uniform(1e5, 5e5, n)},
        index=pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC"),
    )


def last(x):
    """The final value of whatever an indicator returned."""
    if isinstance(x, tuple):
        return tuple(float(pd.Series(s).iloc[-1]) for s in x)
    return float(pd.Series(x).iloc[-1])


SERIES_ON_CLOSE = {
    "rsi": lambda d: ind.rsi(d["close"], 14),
    "ema": lambda d: ind.ema(d["close"], 50),
    "macd_histogram": lambda d: ind.macd_histogram(d["close"]),
    "realized_vol": lambda d: ind.realized_vol(d["close"], 20),
    "zscore": lambda d: ind.zscore(d["close"], 20),
}
SERIES_ON_FRAME = {
    "atr": lambda d: ind.atr(d, 14),
    "adx": lambda d: ind.adx(d, 14),
    "donchian_position": lambda d: ind.donchian_position(d, 20),
    "supertrend": lambda d: ind.supertrend(d, 10, 3.0),
    "cci": lambda d: ind.cci(d, 20),
}


@pytest.mark.parametrize("name", sorted({**SERIES_ON_CLOSE, **SERIES_ON_FRAME}))
def test_an_indicator_at_t_does_not_change_when_the_future_is_removed(name):
    fn = {**SERIES_ON_CLOSE, **SERIES_ON_FRAME}[name]
    full = frame()
    for cut in (300, 350, 399):
        with_future = fn(full)
        without = fn(full.iloc[:cut + 1])

        a = last(without)
        b = (tuple(float(pd.Series(s).iloc[cut]) for s in with_future)
             if isinstance(with_future, tuple)
             else float(pd.Series(with_future).iloc[cut]))

        if isinstance(a, tuple):
            for i, (x, y) in enumerate(zip(a, b)):
                assert x == pytest.approx(y, rel=1e-6, abs=1e-9), \
                    f"{name}[{i}] at t={cut} moved when future bars were removed"
        else:
            assert a == pytest.approx(b, rel=1e-6, abs=1e-9), \
                f"{name} at t={cut} moved when future bars were removed"


def test_the_check_catches_a_deliberately_centred_indicator():
    """A guard that cannot fail proves nothing.

    A centred rolling mean is the canonical lookahead: at time t it
    averages bars on both sides of t.
    """
    full = frame()
    cut = 300
    centred = full["close"].rolling(21, center=True).mean()
    truncated = full.iloc[:cut + 1]["close"].rolling(21, center=True).mean()
    assert float(truncated.iloc[-1]) != pytest.approx(
        float(centred.iloc[cut]), rel=1e-6), \
        "the centred control should differ — if it does not, the check is inert"


def test_gaussian_kernel_regression_is_causal():
    """A smoother is where lookahead usually hides: fitted over the whole
    sample it is the smoothest and the most useless."""
    full = frame()
    cut = 350
    a = float(pd.Series(ind.gaussian_kernel_regression(
        full.iloc[:cut + 1]["close"])).iloc[-1])
    b = float(pd.Series(ind.gaussian_kernel_regression(full["close"])).iloc[cut])
    assert a == pytest.approx(b, rel=1e-6), \
        "the kernel regression sees bars after t"


def test_swing_points_do_not_confirm_before_they_could():
    """A swing high needs `length` bars after it before it is knowable.

    The implementation uses a CENTRED rolling max — lookahead by
    construction — and then shifts forward by `length` to pay it back.
    That is the correct construction and it is worth testing numerically
    rather than trusting, because an off-by-one in the shift would leak
    the future and still look reasonable.

    The series carries forward-filled PRICES, not flags. An earlier
    version of this test cast them with bool(), which only asked "is
    this non-zero" and passed on almost any input.
    """
    full = frame()
    highs_full, lows_full = ind.swing_points(full, length=10)

    compared = 0
    for cut in range(250, 399, 7):
        highs_cut, lows_cut = ind.swing_points(full.iloc[:cut + 1], length=10)
        for cut_series, full_series, label in (
            (highs_cut, highs_full, "swing high"),
            (lows_cut, lows_full, "swing low"),
        ):
            a, b = float(cut_series.iloc[-1]), float(full_series.iloc[cut])
            assert a == pytest.approx(b, rel=1e-9), \
                f"{label} at t={cut} changed once future bars were added"
            compared += 1
    assert compared >= 40, "the sweep must actually compare values"


def test_the_swing_shift_is_what_makes_it_causal():
    """Remove the compensating shift and the leak must show up.

    Without this control the test above would pass even if the shift
    were deleted, and would be asserting nothing.

    It sweeps rather than picking one bar: the series is forward-filled,
    so at most cut points the last confirmed swing is old enough that
    truncation changes nothing. The leak only shows where a centred
    maximum sits within `length` bars of the cut — which is exactly the
    window the shift exists to withhold.
    """
    full = frame()
    length = 10
    highs = full["high"]
    leaky_full = highs.where(
        highs == highs.rolling(length * 2 + 1, center=True).max()).ffill()

    leaked_at = []
    for cut in range(150, 399):
        t = full.iloc[:cut + 1]["high"]
        leaky_cut = t.where(
            t == t.rolling(length * 2 + 1, center=True).max()).ffill()
        a, b = float(leaky_cut.iloc[-1]), float(leaky_full.iloc[cut])
        if a == a and b == b and abs(a - b) > 1e-9:      # both non-NaN
            leaked_at.append(cut)

    assert leaked_at, ("the unshifted control never leaked, so the causality "
                       "test above is inert")
