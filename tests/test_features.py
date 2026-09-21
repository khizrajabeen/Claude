"""Feature engineering: correctness of the fast paths, and no look-ahead.

The statistical block was rewritten for speed (33s to 5s per symbol). The
tests that matter are that the vectorised versions produce the same numbers
as the obvious slow ones, and that no feature can see the future.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bot.ml.features import FeatureEngine, _rolling_autocorr, _strided_hurst
from tests.conftest import make_ohlcv

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


@pytest.fixture
def engine():
    return FeatureEngine({})


def test_vectorised_autocorr_matches_the_reference(engine):
    """The fast path must be the same estimator, not merely a similar one."""
    for seed in (0, 1, 2):
        series = pd.Series(np.random.default_rng(seed).normal(size=400))
        for window in (20, 60):
            fast = _rolling_autocorr(series, window)
            slow = series.rolling(window).apply(lambda x: x.autocorr(lag=1), raw=False)
            aligned = pd.concat([fast, slow], axis=1).dropna()
            assert len(aligned) > 300
            assert float((aligned[0] - aligned[1]).abs().max()) < 1e-10


def test_autocorr_detects_real_persistence():
    rng = np.random.default_rng(3)
    n = 500
    values = np.zeros(n)
    for i in range(1, n):
        values[i] = 0.7 * values[i - 1] + rng.normal(0, 0.3)

    result = _rolling_autocorr(pd.Series(values), 60).dropna()
    assert result.mean() > 0.4, "a strongly autocorrelated series must register"


def test_autocorr_is_near_zero_on_noise():
    series = pd.Series(np.random.default_rng(4).normal(size=800))
    assert abs(_rolling_autocorr(series, 60).dropna().mean()) < 0.1


def test_strided_hurst_stays_in_range_and_is_held_flat():
    returns = pd.Series(np.random.default_rng(5).normal(size=600))
    hurst = _strided_hurst(returns, window=60, stride=6)
    clean = hurst.dropna()
    assert len(clean) > 500
    assert clean.between(0, 1).all()
    # Held flat between recomputations, so consecutive values often repeat.
    assert (clean.diff() == 0).mean() > 0.5


def test_features_have_no_look_ahead(engine):
    """Truncating the series must not change the features that precede the
    cut — a feature that shifts is reading the future."""
    df = make_ohlcv(bars=800, seed=6)
    full = engine.build_features(df)
    partial = engine.build_features(df.iloc[:-50])

    columns = [c for c in engine.get_feature_columns(full)
               if not c.startswith(("vwap", "obv", "ad_line", "hurst"))]
    overlap = partial.index[-100:]

    for column in columns:
        a = full.loc[overlap, column]
        b = partial.loc[overlap, column]
        both = pd.concat([a, b], axis=1).dropna()
        if both.empty:
            continue
        largest = float(both.iloc[:, 0].abs().max()) or 1.0
        diff = float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max())
        assert diff <= largest * 1e-6 + 1e-9, f"{column} changed when the future was removed"


def test_no_target_columns_leak_into_features(engine):
    """Labels come from bot/ml/labeling.py; a next-bar direction column
    sitting in the feature frame is an invitation to leak."""
    features = engine.build_features(make_ohlcv(bars=400, seed=7))
    assert not [c for c in features.columns if c.startswith("target")]


def test_feature_columns_exclude_raw_prices(engine):
    features = engine.build_features(make_ohlcv(bars=400, seed=8))
    columns = engine.get_feature_columns(features)
    assert not ({"open", "high", "low", "close", "volume"} & set(columns))
    assert len(columns) > 80


def test_building_features_is_fast_enough_to_train_with(engine):
    """A minute per symbol makes the training pipeline unusable in practice."""
    import time

    df = make_ohlcv(bars=3000, seed=9)
    started = time.time()
    engine.build_features(df)
    elapsed = time.time() - started
    assert elapsed < 15, f"feature build took {elapsed:.1f}s for 3000 bars"


def test_building_features_emits_no_fragmentation_warning(engine):
    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        engine.build_features(make_ohlcv(bars=600, seed=10))
