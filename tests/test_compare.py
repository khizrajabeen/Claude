"""The model bake-off.

What these guard is not which model wins — that is data-dependent and the
whole point of measuring — but that the comparison is *fair*: identical
folds, no leakage, preprocessing fit inside the fold, and a baseline that
every other model has to beat.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bot.ml.compare import ModelComparison, _predict_proba
from bot.ml.zoo import DEFAULT_LINEUP, REGISTRY, available_models


@pytest.fixture
def dataset():
    """Features with a real but weak signal, plus pure noise columns."""
    rng = np.random.default_rng(0)
    n = 500
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")

    signal = rng.normal(size=n)
    noise = rng.normal(size=(n, 6))
    # The label depends nonlinearly on two features, so trees and nets have
    # something linear models cannot fully capture.
    logit = 0.9 * signal + 0.8 * (signal * noise[:, 0])
    probability = 1 / (1 + np.exp(-logit))
    labels = pd.Series((rng.uniform(size=n) < probability).astype(int), index=index)

    features = pd.DataFrame(
        {"signal": signal, **{f"noise_{i}": noise[:, i] for i in range(6)}},
        index=index,
    )
    returns = pd.Series(np.where(labels == 1, 0.01, -0.01), index=index)
    t1 = pd.Series(index, index=index).shift(-12).bfill()
    weights = pd.Series(1.0, index=index)
    return features, labels, returns, weights, t1


def _fast(config):
    """Shrink the estimators for tests.

    These check the harness is fair, not how well sklearn's defaults fit.
    Full-size forests turn a 6-second suite into a 90-second one.
    """
    config["validation"] = {"n_splits": 3, "embargo_pct": 1.0}
    config["compare"] = {
        "model_params": {
            "random_forest": {"n_estimators": 30},
            "extra_trees": {"n_estimators": 30},
            "hist_gbm": {"max_iter": 40},
            "xgboost": {"n_estimators": 40},
            "lightgbm": {"n_estimators": 40},
            "mlp": {"max_iter": 80, "hidden_layer_sizes": (8,)},
        }
    }
    return config


def run(config, dataset, models):
    features, labels, returns, weights, t1 = dataset
    _fast(config)
    return ModelComparison(config).run(
        features=features, labels=labels, returns=returns,
        feature_columns=list(features.columns),
        sample_weight=weights, t1=t1, models=models,
    )


# ── Zoo ──────────────────────────────────────────────────────

def test_the_zoo_spans_several_model_families():
    families = {REGISTRY[name].family for name in DEFAULT_LINEUP}
    assert {"baseline", "linear", "trees", "boosting", "neural"} <= families


def test_missing_libraries_are_skipped_not_fatal():
    spec = REGISTRY["xgboost"]
    assert spec.requires == "xgboost"
    # Whatever is installed here, asking for an unknown name must not raise.
    assert available_models(["definitely_not_a_model"]) == []


def test_every_available_model_builds(config):
    for spec in available_models(list(REGISTRY)):
        assert spec.build() is not None


def test_default_lineup_covers_the_literature_claims():
    """Trees and nets are claimed to beat linear via nonlinear interactions;
    boosting is claimed to beat deep nets on tabular data. Both claims need
    representatives in the race."""
    families = {REGISTRY[name].family for name in DEFAULT_LINEUP}
    assert "linear" in families and "neural" in families
    assert "trees" in families and "boosting" in families


# ── Fairness ─────────────────────────────────────────────────

def test_all_models_see_identical_folds(config, dataset):
    results = run(config, dataset, ["majority", "logistic", "random_forest"])
    fold_counts = {r.folds for r in results if r.folds}
    assert len(fold_counts) == 1, "models were scored on different numbers of folds"

    sizes = [tuple(f["n_test"] for f in r.per_fold) for r in results if r.per_fold]
    assert len(set(sizes)) == 1, "models were scored on different test sets"


def test_the_baseline_is_included_and_beatable(config, dataset):
    results = run(config, dataset, ["majority", "hist_gbm"])
    baseline = next(r for r in results if r.name == "majority")
    assert baseline.folds > 0
    # The baseline has no ranking ability by construction.
    assert 0.45 <= baseline.auc <= 0.55


def test_a_real_signal_is_detected(config, dataset):
    """Sanity check on the harness: if it cannot find a planted signal, its
    verdict on real data means nothing."""
    results = run(config, dataset, ["majority", "logistic", "hist_gbm"])
    best = max((r for r in results if r.name != "majority"), key=lambda r: r.auc)
    assert best.auc > 0.6, "the harness failed to detect a signal that is there"


def test_pure_noise_produces_no_edge(config):
    """The more important direction: no signal must yield no edge."""
    rng = np.random.default_rng(1)
    n = 500
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    features = pd.DataFrame(rng.normal(size=(n, 6)),
                            columns=[f"f{i}" for i in range(6)], index=index)
    labels = pd.Series(rng.integers(0, 2, n), index=index)
    returns = pd.Series(np.where(labels == 1, 0.01, -0.01), index=index)
    t1 = pd.Series(index, index=index).shift(-12).bfill()

    _fast(config)
    results = ModelComparison(config).run(
        features=features, labels=labels, returns=returns,
        feature_columns=list(features.columns), t1=t1,
        models=["logistic", "random_forest", "hist_gbm"],
    )
    for result in results:
        assert result.auc < 0.60, f"{result.name} found signal in noise (AUC {result.auc})"


def test_results_are_ranked_by_auc(config, dataset):
    results = run(config, dataset, ["majority", "logistic", "random_forest", "hist_gbm"])
    aucs = [r.auc for r in results if r.folds]
    assert aucs == sorted(aucs, reverse=True)


def test_every_result_carries_its_uncertainty(config, dataset):
    """A mean with no spread cannot be judged."""
    for result in run(config, dataset, ["logistic", "hist_gbm"]):
        assert result.accuracy_std >= 0
        assert result.folds == len(result.per_fold)
        assert result.majority > 0


def test_calibration_is_reported_not_just_accuracy(config, dataset):
    """Sizing consumes probabilities, so overconfidence is a real cost."""
    for result in run(config, dataset, ["logistic", "hist_gbm"]):
        assert result.log_loss > 0
        assert 0 <= result.brier <= 1


def test_per_trade_return_is_reported_not_a_meaningless_sum(config, dataset):
    """Summing thousands of overlapping per-bar outcomes yields a headline
    like -4000% that only measures costs times trade count."""
    for result in run(config, dataset, ["logistic", "hist_gbm"]):
        assert -500 < result.avg_trade_bps < 500
        assert 0 <= result.trade_rate <= 1.0
        assert result.trades_taken >= 0


def test_paper_pnl_charges_costs(config, dataset):
    """A model that is right 51% of the time still loses after fees."""
    comparison = ModelComparison(config)
    comparison.cost_bps = 0.0
    comparison.confidence_floor = 0.5
    free = comparison._paper_pnl(np.array([0.9, 0.9]), np.array([0.01, 0.01]))

    comparison.cost_bps = 50.0
    costly = comparison._paper_pnl(np.array([0.9, 0.9]), np.array([0.01, 0.01]))
    assert sum(costly) < sum(free)


def test_low_confidence_predictions_are_not_traded(config, dataset):
    comparison = ModelComparison(config)
    comparison.confidence_floor = 0.9
    assert comparison._paper_pnl(np.array([0.55, 0.45]), np.array([0.01, -0.01])) == []


def test_a_failing_model_does_not_abort_the_comparison(config, dataset):
    features, labels, returns, weights, t1 = dataset

    class Exploding:
        def fit(self, *a, **k):
            raise RuntimeError("boom")

    spec = REGISTRY["logistic"]
    original = spec.builder
    try:
        spec.builder = lambda **k: Exploding()
        _fast(config)
        results = ModelComparison(config).run(
            features=features, labels=labels, returns=returns,
            feature_columns=list(features.columns), sample_weight=weights, t1=t1,
            models=["logistic", "hist_gbm"],
        )
    finally:
        spec.builder = original

    failed = next(r for r in results if r.name == "logistic")
    survived = next(r for r in results if r.name == "hist_gbm")
    assert failed.error and failed.folds == 0
    assert survived.folds > 0


def test_too_little_data_is_refused(config, dataset):
    features, labels, returns, weights, t1 = dataset
    config["validation"] = {"n_splits": 5, "embargo_pct": 1.0}
    with pytest.raises(ValueError, match="too few"):
        ModelComparison(config).run(
            features=features.head(40), labels=labels.head(40),
            returns=returns.head(40), feature_columns=list(features.columns),
            t1=t1.head(40), models=["logistic"],
        )


def test_probability_extraction_handles_any_estimator():
    class OnlyDecision:
        def decision_function(self, X):
            return np.array([2.0, -2.0])

    proba = _predict_proba(OnlyDecision(), np.zeros((2, 3)))
    assert 0 < proba[1] < 0.5 < proba[0] < 1
