"""The model zoo — candidates for the bake-off, on equal terms.

The literature disagrees about what wins on financial data, so the only
useful answer is a measurement on this data, with every model given the
same features, the same triple-barrier labels and the same purged folds.

What the published work claims, and what this is set up to check:

  * Gu, Kelly and Xiu find trees and neural networks beat linear models on
    US equities, attributing the gain to nonlinear interactions, with the
    neural-net long-short decile spread reaching a Sharpe around 1.35.
  * More recent work on tabular financial features tends to find gradient
    boosting matching or beating deep learning at far lower cost — an
    LSTM has to learn from sequence what a tree is handed as a feature.

Both can be true: NNs win where the panel is huge (thousands of names,
decades of months), trees win where it is not. A few thousand crypto bars
is emphatically the second case, which is the prior this zoo is built to
test rather than assume.

Every model here is a scikit-learn-compatible classifier. Models whose
libraries are missing are reported as unavailable rather than crashing the
comparison.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger("trading_bot")


@dataclass
class ModelSpec:
    """One candidate: how to build it, and why it is in the race."""

    name: str
    family: str
    rationale: str
    builder: object = None
    requires: str = ""
    scale_features: bool = False
    params: dict = field(default_factory=dict)

    def build(self, **overrides):
        params = {**self.params, **overrides}
        return self.builder(**params)

    def available(self) -> bool:
        if not self.requires:
            return True
        import importlib
        try:
            importlib.import_module(self.requires)
            return True
        except ImportError:
            return False


# ── Builders ─────────────────────────────────────────────────

def _majority(**_):
    """Always predict the majority class. Every other model has to beat
    this, and on financial data most do not."""
    from sklearn.dummy import DummyClassifier
    return DummyClassifier(strategy="prior")


def _logistic(**params):
    from sklearn.linear_model import LogisticRegression
    defaults = dict(C=0.1, max_iter=2000, solver="lbfgs")
    return LogisticRegression(**{**defaults, **params})


def _elastic_net(**params):
    from sklearn.linear_model import LogisticRegression
    # L1/L2 mix: heavy regularisation is the point on noisy, collinear
    # financial features.
    defaults = dict(penalty="elasticnet", l1_ratio=0.5, C=0.05,
                    solver="saga", max_iter=3000)
    return LogisticRegression(**{**defaults, **params})


def _random_forest(**params):
    from sklearn.ensemble import RandomForestClassifier
    defaults = dict(n_estimators=400, max_depth=6, min_samples_leaf=25,
                    max_features="sqrt", n_jobs=-1, random_state=7)
    return RandomForestClassifier(**{**defaults, **params})


def _extra_trees(**params):
    from sklearn.ensemble import ExtraTreesClassifier
    defaults = dict(n_estimators=400, max_depth=8, min_samples_leaf=25,
                    max_features="sqrt", n_jobs=-1, random_state=7)
    return ExtraTreesClassifier(**{**defaults, **params})


def _hist_gbm(**params):
    from sklearn.ensemble import HistGradientBoostingClassifier
    defaults = dict(max_iter=300, max_depth=4, learning_rate=0.05,
                    l2_regularization=1.0, min_samples_leaf=25, random_state=7)
    return HistGradientBoostingClassifier(**{**defaults, **params})


def _xgboost(**params):
    import xgboost as xgb
    defaults = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.5,
                    reg_lambda=2.0, min_child_weight=5, eval_metric="logloss",
                    verbosity=0, n_jobs=-1, random_state=7)
    return xgb.XGBClassifier(**{**defaults, **params})


def _lightgbm(**params):
    import lightgbm as lgb
    defaults = dict(n_estimators=300, max_depth=4, num_leaves=15,
                    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=2.0, min_child_samples=25,
                    n_jobs=-1, verbose=-1, random_state=7)
    return lgb.LGBMClassifier(**{**defaults, **params})


def _mlp(**params):
    from sklearn.neural_network import MLPClassifier
    # Deliberately small. Gu/Kelly/Xiu found shallow networks beat deep
    # ones on asset pricing panels, and this sample is far smaller.
    defaults = dict(hidden_layer_sizes=(32, 16), alpha=1e-2, max_iter=600,
                    early_stopping=True, n_iter_no_change=20, random_state=7)
    return MLPClassifier(**{**defaults, **params})


REGISTRY: dict[str, ModelSpec] = {
    spec.name: spec for spec in [
        ModelSpec("majority", "baseline", builder=_majority,
                  rationale="predicts the majority class; the bar to clear"),
        ModelSpec("logistic", "linear", builder=_logistic, scale_features=True,
                  rationale="regularised linear baseline"),
        ModelSpec("elasticnet", "linear", builder=_elastic_net, scale_features=True,
                  rationale="L1/L2 mix; drops collinear features outright"),
        ModelSpec("random_forest", "trees", builder=_random_forest,
                  rationale="bagged trees; low variance, nonlinear"),
        ModelSpec("extra_trees", "trees", builder=_extra_trees,
                  rationale="extra randomisation; more bias, less overfit"),
        ModelSpec("hist_gbm", "boosting", builder=_hist_gbm,
                  rationale="gradient boosting with no extra dependency"),
        ModelSpec("xgboost", "boosting", builder=_xgboost, requires="xgboost",
                  rationale="the standard tabular boosting benchmark"),
        ModelSpec("lightgbm", "boosting", builder=_lightgbm, requires="lightgbm",
                  rationale="leaf-wise boosting; the tabular finance favourite"),
        ModelSpec("mlp", "neural", builder=_mlp, scale_features=True,
                  rationale="shallow net; the nonlinear-interaction claim"),
    ]
}

DEFAULT_LINEUP = ["majority", "logistic", "random_forest", "hist_gbm",
                  "xgboost", "lightgbm", "mlp"]


def available_models(names: list[str] | None = None) -> list[ModelSpec]:
    """The requested models that can actually be built here."""
    requested = names or DEFAULT_LINEUP
    out = []
    for name in requested:
        spec = REGISTRY.get(name)
        if spec is None:
            logger.warning("Unknown model '%s' — known: %s", name, ", ".join(REGISTRY))
            continue
        if not spec.available():
            logger.info("Skipping %s: %s is not installed", name, spec.requires)
            continue
        out.append(spec)
    return out
