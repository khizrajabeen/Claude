"""Leakage-free cross-validation for financial time series.

Ordinary k-fold assumes samples are independent. Financial labels are not:
a triple-barrier label at time t depends on prices up to t1, so a training
row can overlap a test row and leak the answer. Standard k-fold on this
data reports accuracy the live system will never see.

Two corrections, both from López de Prado:

  purging    drop training rows whose label window overlaps the test fold
  embargo    drop a further band of rows right after the test fold, because
             serial correlation in the features leaks across the boundary
             even when the label windows do not

Also here: a walk-forward splitter, which is the honest way to evaluate a
strategy that will be retrained periodically in production.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")


def _ordinals(index: pd.Index) -> np.ndarray:
    """Comparable integer positions for an index.

    Timestamps go to nanoseconds since the epoch so tz-aware and tz-naive
    values are never compared directly, which raises in pandas.

    Pooled datasets carry a (timestamp, symbol) MultiIndex — several
    symbols share each timestamp. Purging is a statement about *time*, not
    about rows, so the time level is what gets compared: a training row
    must be dropped when its label window overlaps the test window, whether
    or not it belongs to the same symbol. Purging per-symbol instead would
    leave BTC rows leaking into an ETH test fold.
    """
    if isinstance(index, pd.MultiIndex):
        for level in range(index.nlevels):
            values = index.get_level_values(level)
            if isinstance(values, pd.DatetimeIndex):
                return _ordinals(values)
        # No time level: fall back to position, which still blocks the
        # exact rows under test from appearing in training.
        return np.arange(len(index), dtype=np.int64)
    if isinstance(index, pd.DatetimeIndex):
        return index.tz_convert("UTC").asi8 if index.tz is not None else index.asi8
    try:
        return np.asarray(index, dtype=np.int64)
    except (TypeError, ValueError):
        return np.arange(len(index), dtype=np.int64)


def _ordinals_with_mask(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Integer ordinals plus a mask of the entries that are not missing."""
    resolved = series.notna().to_numpy()
    values = np.zeros(len(series), dtype=np.int64)
    if resolved.any():
        present = series[resolved]
        if pd.api.types.is_datetime64_any_dtype(present):
            idx = pd.DatetimeIndex(present)
            values[resolved] = idx.tz_convert("UTC").asi8 if idx.tz is not None else idx.asi8
        else:
            values[resolved] = present.to_numpy(dtype=np.int64)
    return values, resolved


class PurgedKFold:
    """K-fold over time with purging and an embargo.

    Args:
        n_splits: number of folds.
        t1: Series mapping each observation's start time to the time its
            label resolves. Without it, purging cannot know what overlaps.
        embargo_pct: embargo size as a percentage of the sample length.
    """

    def __init__(self, n_splits: int = 5, t1: pd.Series | None = None,
                 embargo_pct: float = 1.0):
        if n_splits < 2:
            raise ValueError("n_splits must be at least 2")
        self.n_splits = n_splits
        self.t1 = t1
        self.embargo_pct = float(embargo_pct)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        index = X.index if isinstance(X, (pd.DataFrame, pd.Series)) else pd.RangeIndex(len(X))
        n = len(index)
        if n < self.n_splits * 2:
            raise ValueError(f"{n} samples is too few for {self.n_splits} folds")

        embargo = int(n * self.embargo_pct / 100.0)
        positions = np.arange(n)
        fold_bounds = [
            (b[0], b[-1] + 1) for b in np.array_split(positions, self.n_splits)
        ]

        t1 = self.t1
        if t1 is not None:
            t1 = t1.reindex(index)

        for start, stop in fold_bounds:
            test_idx = positions[start:stop]
            train_mask = np.ones(n, dtype=bool)
            train_mask[start:stop] = False

            # Embargo the band immediately after the test fold.
            if embargo > 0:
                train_mask[stop:min(n, stop + embargo)] = False

            # Purge training rows whose label window overlaps the test fold.
            if t1 is not None:
                starts = _ordinals(index)
                label_end, resolved = _ordinals_with_mask(t1)
                test_start, test_end = starts[start], starts[stop - 1]

                overlaps = np.zeros(n, dtype=bool)
                overlaps[resolved] = (
                    (starts[resolved] <= test_end) & (label_end[resolved] >= test_start)
                )
                train_mask &= ~overlaps

            train_idx = positions[train_mask]
            if len(train_idx) == 0:
                logger.warning("Fold starting at %d has no training rows after purging", start)
                continue
            yield train_idx, test_idx


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step: int | None = None,
    embargo: int = 0,
    anchored: bool = False,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Rolling (or anchored) train/test windows moving forward in time.

    This is what production actually does: fit on history, trade the next
    stretch, refit. Evaluating that way is the only estimate that matches
    how the model will be used.
    """
    step = step or test_size
    splits = []
    start = 0
    while start + train_size + embargo + test_size <= n_samples:
        train_start = 0 if anchored else start
        train_end = start + train_size
        test_start = train_end + embargo
        test_end = test_start + test_size
        splits.append((
            np.arange(train_start, train_end),
            np.arange(test_start, test_end),
        ))
        start += step
    return splits


def deflated_sharpe_penalty(n_trials: int) -> float:
    """How much Sharpe to discount for having tried `n_trials` variants.

    Testing many configurations guarantees one looks good by luck. This
    returns the expected maximum Sharpe of `n_trials` strategies with no
    edge at all — anything below it is not evidence of anything.
    """
    if n_trials <= 1:
        return 0.0
    from scipy.stats import norm
    euler = 0.5772156649
    return float(
        norm.ppf(1 - 1 / n_trials) * (1 - euler)
        + norm.ppf(1 - 1 / (n_trials * np.e)) * euler
    )


def assert_no_leakage(train_idx: np.ndarray, test_idx: np.ndarray,
                      index: pd.Index, t1: pd.Series | None = None) -> None:
    """Raise if a split leaks. Cheap enough to run in tests and in training."""
    overlap = np.intersect1d(train_idx, test_idx)
    if len(overlap):
        raise AssertionError(f"{len(overlap)} rows appear in both train and test")

    if t1 is None:
        return

    all_starts = _ordinals(index)
    test_start, test_end = all_starts[test_idx.min()], all_starts[test_idx.max()]
    ends, resolved = _ordinals_with_mask(t1.reindex(index))
    starts = all_starts[train_idx]
    ends_train = ends[train_idx]
    resolved_train = resolved[train_idx]
    leaking = (
        (starts[resolved_train] <= test_end) & (ends_train[resolved_train] >= test_start)
    )
    if leaking.any():
        raise AssertionError(
            f"{int(leaking.sum())} training labels overlap the test window — "
            "purging did not run or t1 is wrong"
        )
