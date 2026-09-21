"""Labelling and validation.

The claim being tested is narrow but important: the folds this code
produces do not leak, and a naive split does. If that stops being true, any
accuracy number the trainer reports is meaningless.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.ml.labeling import (
    label_balance, primary_side_from_signal, sample_weights_by_uniqueness,
    triple_barrier_labels,
)
from bot.ml.validation import (
    PurgedKFold, assert_no_leakage, deflated_sharpe_penalty, walk_forward_splits,
)
from tests.conftest import make_ohlcv


# ── Triple barrier ───────────────────────────────────────────

def test_labels_record_which_barrier_was_touched():
    df = make_ohlcv(bars=500, drift=0.002, vol=0.01, seed=3)
    labels = triple_barrier_labels(df, max_holding_bars=24)

    assert set(labels["label"].dropna().unique()) <= {-1.0, 0.0, 1.0}
    assert (labels["bars_held"].dropna() <= 24).all()
    # Winners have positive returns and losers negative, by construction.
    resolved = labels.dropna(subset=["label"])
    assert (resolved.loc[resolved["label"] == 1, "ret"] > 0).all()
    assert (resolved.loc[resolved["label"] == -1, "ret"] < 0).all()


def test_unresolvable_tail_is_left_unlabelled():
    """Bars near the end cannot know their outcome without future data."""
    df = make_ohlcv(bars=300, seed=4)
    labels = triple_barrier_labels(df, max_holding_bars=48)
    assert labels["label"].tail(48).isna().all()


def test_a_strong_uptrend_produces_mostly_winning_long_labels():
    df = make_ohlcv(bars=400, drift=0.01, vol=0.002, seed=5)
    balance = label_balance(triple_barrier_labels(df, max_holding_bars=48, side=1))
    assert balance["profit_pct"] > balance["stop_pct"]


def test_short_side_labels_invert():
    df = make_ohlcv(bars=400, drift=0.01, vol=0.002, seed=5)
    longs = label_balance(triple_barrier_labels(df, max_holding_bars=48, side=1))
    shorts = label_balance(triple_barrier_labels(df, max_holding_bars=48, side=-1))
    assert shorts["profit_pct"] < longs["profit_pct"]


def test_stop_wins_ties_in_labelling_too():
    """Labels must make the same pessimistic assumption as the broker."""
    index = pd.date_range("2026-01-01", periods=30, freq="h", tz="UTC")
    df = pd.DataFrame(
        {"open": 100.0, "high": 100.0, "low": 100.0, "close": 100.0, "volume": 1.0},
        index=index,
    )
    # One bar that spans both barriers.
    df.iloc[:15, df.columns.get_loc("high")] = 101.0
    df.iloc[:15, df.columns.get_loc("low")] = 99.0
    df.iloc[20, df.columns.get_loc("high")] = 130.0
    df.iloc[20, df.columns.get_loc("low")] = 70.0

    labels = triple_barrier_labels(df, atr_period=5, upper_atr=1.0,
                                   lower_atr=1.0, max_holding_bars=10)
    touched = labels["label"].dropna()
    assert (touched != 1.0).all() or touched.eq(-1.0).any()


def test_overlapping_labels_are_down_weighted():
    """Ten labels covering the same bars are not ten independent facts."""
    df = make_ohlcv(bars=300, seed=6)
    labels = triple_barrier_labels(df, max_holding_bars=48)
    weights = sample_weights_by_uniqueness(labels)

    assert (weights >= 0).all()
    assert weights[labels["t1"].isna()].eq(0).all()
    resolved = weights[labels["t1"].notna()]
    assert resolved.std() > 0, "every weight identical means uniqueness was ignored"


def test_primary_side_thresholds():
    edge = pd.Series([-0.9, -0.05, 0.0, 0.05, 0.9])
    side = primary_side_from_signal(edge, threshold=0.1)
    assert list(side) == [-1.0, 0.0, 0.0, 0.0, 1.0]


# ── Purged cross-validation ──────────────────────────────────

@pytest.fixture
def labelled():
    index = pd.date_range("2026-01-01", periods=800, freq="h", tz="UTC")
    X = pd.DataFrame({"f": np.random.default_rng(0).normal(size=800)}, index=index)
    t1 = pd.Series(index, index=index).shift(-24).bfill()
    return X, t1


def test_purged_folds_do_not_leak(labelled):
    X, t1 = labelled
    cv = PurgedKFold(n_splits=5, t1=t1, embargo_pct=1.0)

    folds = list(cv.split(X))
    assert len(folds) == 5
    for train_idx, test_idx in folds:
        assert_no_leakage(train_idx, test_idx, X.index, t1)


def test_a_naive_split_does_leak(labelled):
    """The control: without purging, overlapping labels cross the boundary."""
    X, t1 = labelled
    naive = PurgedKFold(n_splits=5, t1=None, embargo_pct=0.0)
    train_idx, test_idx = list(naive.split(X))[1]

    with pytest.raises(AssertionError, match="overlap"):
        assert_no_leakage(train_idx, test_idx, X.index, t1)


def test_purging_removes_more_rows_than_the_test_fold(labelled):
    X, t1 = labelled
    purged = list(PurgedKFold(n_splits=5, t1=t1, embargo_pct=1.0).split(X))
    naive = list(PurgedKFold(n_splits=5, t1=None, embargo_pct=0.0).split(X))

    assert len(purged[1][0]) < len(naive[1][0]), "purging must drop training rows"


def test_every_row_is_tested_exactly_once(labelled):
    X, t1 = labelled
    tested = np.concatenate([te for _, te in PurgedKFold(n_splits=5, t1=t1).split(X)])
    assert len(tested) == len(X)
    assert len(np.unique(tested)) == len(X)


def test_embargo_widens_the_gap(labelled):
    X, t1 = labelled
    small = list(PurgedKFold(n_splits=5, t1=t1, embargo_pct=0.0).split(X))[0]
    large = list(PurgedKFold(n_splits=5, t1=t1, embargo_pct=10.0).split(X))[0]
    assert len(large[0]) < len(small[0])


def test_purged_kfold_rejects_impossible_configurations(labelled):
    X, _ = labelled
    with pytest.raises(ValueError):
        PurgedKFold(n_splits=1)
    with pytest.raises(ValueError):
        list(PurgedKFold(n_splits=5).split(X.head(4)))


# ── Walk forward ─────────────────────────────────────────────

def test_walk_forward_always_trains_before_it_tests():
    splits = walk_forward_splits(1000, train_size=300, test_size=100, embargo=10)
    assert splits
    for train_idx, test_idx in splits:
        assert train_idx.max() < test_idx.min(), "a fold peeked into the future"
        assert test_idx.min() - train_idx.max() > 10, "embargo not applied"


def test_anchored_walk_forward_keeps_all_history():
    rolling = walk_forward_splits(1000, 300, 100, anchored=False)
    anchored = walk_forward_splits(1000, 300, 100, anchored=True)
    assert rolling[-1][0][0] > 0
    assert anchored[-1][0][0] == 0
    assert len(anchored[-1][0]) > len(rolling[-1][0])


def test_deflated_sharpe_penalty_grows_with_trials():
    assert deflated_sharpe_penalty(1) == 0.0
    assert deflated_sharpe_penalty(100) > deflated_sharpe_penalty(10) > 0
