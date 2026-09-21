"""Gradient-boosted ensemble with leakage-free fitting.

What changed from the previous version, and why it mattered:

  * Feature selection and the scaler were fit on the whole dataset, then
    evaluated on a slice of it. Both steps see the validation rows, so the
    reported accuracy was partly a measure of having already looked at the
    answers. Everything is now fit inside each training fold.
  * Validation used a plain chronological split with no purging, so labels
    spanning the boundary leaked. Folds now come from PurgedKFold.
  * The networks emitted three classes against two-class labels, leaving a
    dead output. Sequence models are now optional and off by default: on a
    few thousand crypto bars a bidirectional LSTM plus a transformer have
    far more capacity than the data supports, and the trees carry the
    signal. Enable them only with a lot more data than this fetches.

Sample weights come from label uniqueness, so overlapping triple-barrier
labels are not counted as independent observations.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from bot.ml.validation import PurgedKFold

logger = logging.getLogger("trading_bot")

MODEL_DIR = Path("models")


class EnsembleModel:
    """XGBoost + LightGBM, soft-voted, fit without leakage."""

    def __init__(self, config: dict):
        self.config = config
        self.model_config = config.get("model", {})
        self.weights = self.model_config.get("ensemble_weights", {
            "xgboost": 0.5, "lightgbm": 0.5,
        })
        self.top_k = int(self.model_config.get("features_top_k", 60))
        self.min_confidence = float(self.model_config.get("min_confidence", 0.55))

        self.xgb_model = None
        self.lgb_model = None
        self.scaler: RobustScaler | None = None
        self.feature_columns: list[str] = []
        self.cv_report: dict = {}
        self.is_trained = False

    # ── Fitting ───────────────────────────────────────────────

    def fit(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        feature_columns: list[str],
        sample_weight: pd.Series | None = None,
        t1: pd.Series | None = None,
        n_splits: int = 5,
        embargo_pct: float = 1.0,
    ) -> dict:
        """Cross-validate with purging, then refit on everything.

        `labels` must be binary (1 = the trade worked). `t1` is when each
        label resolves; without it purging cannot run and the scores will
        be optimistic, so its absence is logged loudly.
        """
        X_all, y_all, w_all, t1_all = _align(features, labels, feature_columns,
                                             sample_weight, t1)
        if len(X_all) < n_splits * 20:
            raise ValueError(
                f"{len(X_all)} usable rows is too few to validate over {n_splits} folds"
            )
        if t1_all is None:
            logger.warning(
                "No label end times supplied — cross-validation cannot purge "
                "overlapping labels and scores will be optimistic"
            )

        logger.info("Fitting on %d rows x %d features", len(X_all), len(feature_columns))

        cv = PurgedKFold(n_splits=n_splits, t1=t1_all, embargo_pct=embargo_pct)
        fold_scores = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(X_all), start=1):
            score = self._fit_fold(X_all, y_all, w_all, train_idx, test_idx, fold)
            if score:
                fold_scores.append(score)

        self.cv_report = _summarize_folds(fold_scores)
        logger.info(
            "Purged CV: accuracy %.4f ± %.4f | AUC %.4f | base rate %.4f",
            self.cv_report.get("accuracy_mean", 0.0),
            self.cv_report.get("accuracy_std", 0.0),
            self.cv_report.get("auc_mean", 0.0),
            self.cv_report.get("base_rate", 0.0),
        )
        edge = self.cv_report.get("accuracy_mean", 0.0) - self.cv_report.get("base_rate", 0.0)
        if edge <= 0.005:
            logger.warning(
                "Cross-validated accuracy is within noise of always predicting the "
                "majority class (edge %+.4f). This model has not found anything; "
                "do not size positions on it.", edge,
            )

        # Refit on the full sample for live use. The CV report above, not
        # anything measured here, is the estimate of out-of-sample skill.
        self.feature_columns = self._select_features(
            X_all.to_numpy(), y_all.to_numpy(), feature_columns,
            w_all.to_numpy() if w_all is not None else None,
        )
        X_final = np.nan_to_num(X_all[self.feature_columns].to_numpy(), nan=0.0)
        self.scaler = RobustScaler().fit(X_final)
        scaled = self.scaler.transform(X_final)

        weights = w_all.to_numpy() if w_all is not None else None
        self.xgb_model = _fit_xgb(scaled, y_all.to_numpy(), weights)
        self.lgb_model = _fit_lgb(scaled, y_all.to_numpy(), weights)
        self.is_trained = True
        self.save()
        return self.cv_report

    def _fit_fold(self, X_all, y_all, w_all, train_idx, test_idx, fold) -> dict | None:
        """Fit one fold with every preprocessing step inside the fold."""
        try:
            from sklearn.metrics import roc_auc_score
        except ImportError:
            roc_auc_score = None

        X_tr_raw = X_all.iloc[train_idx]
        y_tr = y_all.iloc[train_idx].to_numpy()
        X_te_raw = X_all.iloc[test_idx]
        y_te = y_all.iloc[test_idx].to_numpy()
        w_tr = w_all.iloc[train_idx].to_numpy() if w_all is not None else None

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
            logger.debug("Fold %d has a single class — skipped", fold)
            return None

        # Selection and scaling fit on training rows only.
        selected = self._select_features(
            X_tr_raw.to_numpy(), y_tr, list(X_all.columns), w_tr
        )
        scaler = RobustScaler().fit(
            np.nan_to_num(X_tr_raw[selected].to_numpy(), nan=0.0)
        )
        X_tr = scaler.transform(np.nan_to_num(X_tr_raw[selected].to_numpy(), nan=0.0))
        X_te = scaler.transform(np.nan_to_num(X_te_raw[selected].to_numpy(), nan=0.0))

        xgb_model = _fit_xgb(X_tr, y_tr, w_tr)
        lgb_model = _fit_lgb(X_tr, y_tr, w_tr)

        proba = (
            self.weights.get("xgboost", 0.5) * xgb_model.predict_proba(X_te)[:, 1]
            + self.weights.get("lightgbm", 0.5) * lgb_model.predict_proba(X_te)[:, 1]
        )
        total = self.weights.get("xgboost", 0.5) + self.weights.get("lightgbm", 0.5)
        proba = proba / total if total else proba

        predicted = (proba > 0.5).astype(int)
        accuracy = float((predicted == y_te).mean())
        # The majority-class rate is the bar any model has to clear.
        base_rate = float(max(y_te.mean(), 1 - y_te.mean()))
        auc = float(roc_auc_score(y_te, proba)) if roc_auc_score and len(np.unique(y_te)) > 1 else 0.5

        logger.info(
            "  fold %d: n_train=%d n_test=%d accuracy=%.4f base=%.4f auc=%.4f",
            fold, len(train_idx), len(test_idx), accuracy, base_rate, auc,
        )
        return {"accuracy": accuracy, "base_rate": base_rate, "auc": auc,
                "n_train": len(train_idx), "n_test": len(test_idx)}

    def _select_features(self, X: np.ndarray, y: np.ndarray,
                         columns: list[str], weights: np.ndarray | None) -> list[str]:
        """Pick the top-k features by gain. Fit on whatever rows are given,
        which callers must keep to a training fold."""
        import xgboost as xgb

        k = min(self.top_k, len(columns))
        selector = xgb.XGBClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="logloss", verbosity=0, n_jobs=-1,
        )
        selector.fit(np.nan_to_num(X, nan=0.0), y, sample_weight=weights)
        order = np.argsort(selector.feature_importances_)[-k:]
        return [columns[i] for i in sorted(order)]

    # ── Prediction ────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> dict:
        """Probability that the next trade works, for the most recent row."""
        if not self.is_trained:
            raise RuntimeError("model is not trained")

        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise KeyError(f"missing {len(missing)} feature(s), e.g. {missing[:3]}")

        X = np.nan_to_num(df[self.feature_columns].to_numpy()[-1:], nan=0.0)
        scaled = self.scaler.transform(X)

        sub = {
            "xgboost": float(self.xgb_model.predict_proba(scaled)[0][1]),
            "lightgbm": float(self.lgb_model.predict_proba(scaled)[0][1]),
        }
        w = self.weights
        total = w.get("xgboost", 0.5) + w.get("lightgbm", 0.5)
        probability = (
            w.get("xgboost", 0.5) * sub["xgboost"] + w.get("lightgbm", 0.5) * sub["lightgbm"]
        ) / (total or 1.0)

        return {
            "probability": probability,
            # Distance from a coin flip, on a 0-1 scale.
            "confidence": abs(probability - 0.5) * 2,
            "direction": 1 if probability > 0.5 else -1,
            "sub_predictions": sub,
        }

    def edge_contribution(self, df: pd.DataFrame) -> float:
        """Signed tilt in [-1, 1] for the day planner to blend in.

        Returns 0 below the confidence floor rather than a weak opinion,
        because a barely-better-than-random probability is noise.
        """
        if not self.is_trained:
            return 0.0
        try:
            prediction = self.predict(df)
        except (KeyError, RuntimeError) as e:
            logger.debug("Model prediction unavailable: %s", e)
            return 0.0
        if prediction["confidence"] < self.min_confidence:
            return 0.0
        return round(prediction["direction"] * prediction["confidence"], 4)

    # ── Persistence ───────────────────────────────────────────

    def save(self, directory: Path | None = None) -> None:
        directory = Path(directory or MODEL_DIR)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "xgb": self.xgb_model,
                "lgb": self.lgb_model,
                "scaler": self.scaler,
                "feature_columns": self.feature_columns,
                "weights": self.weights,
                "cv_report": self.cv_report,
            },
            directory / "ensemble.pkl",
        )
        logger.info("Model saved to %s", directory / "ensemble.pkl")

    def load(self, directory: Path | None = None) -> bool:
        directory = Path(directory or MODEL_DIR)
        path = directory / "ensemble.pkl"
        if not path.exists():
            logger.info("No trained model at %s", path)
            self.is_trained = False
            return False
        try:
            bundle = joblib.load(path)
        except Exception as e:
            logger.error("Could not load model: %s", e)
            self.is_trained = False
            return False

        self.xgb_model = bundle.get("xgb")
        self.lgb_model = bundle.get("lgb")
        self.scaler = bundle.get("scaler")
        self.feature_columns = bundle.get("feature_columns", [])
        self.weights = bundle.get("weights", self.weights)
        self.cv_report = bundle.get("cv_report", {})
        self.is_trained = bool(self.xgb_model and self.lgb_model and self.scaler)
        if self.is_trained:
            logger.info("Model loaded (%d features, CV accuracy %.4f)",
                        len(self.feature_columns),
                        self.cv_report.get("accuracy_mean", 0.0))
        return self.is_trained


# ── helpers ──────────────────────────────────────────────────

def _fit_xgb(X, y, weights):
    import xgboost as xgb
    model = xgb.XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0, min_child_weight=5,
        eval_metric="logloss", verbosity=0, n_jobs=-1,
    )
    model.fit(X, y, sample_weight=weights)
    return model


def _fit_lgb(X, y, weights):
    import lightgbm as lgb
    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=4, num_leaves=15, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.5, reg_lambda=2.0, min_child_samples=20,
        n_jobs=-1, verbose=-1,
    )
    model.fit(X, y, sample_weight=weights)
    return model


def _align(features, labels, feature_columns, sample_weight, t1):
    """Drop rows without a resolved label and line everything up."""
    columns = [c for c in feature_columns if c in features.columns]
    frame = features[columns].replace([np.inf, -np.inf], np.nan)
    mask = labels.reindex(frame.index).notna()
    frame = frame.loc[mask]
    y = labels.reindex(frame.index).astype(int)
    w = sample_weight.reindex(frame.index) if sample_weight is not None else None
    if w is not None:
        # Zero-weight rows contribute nothing but still skew the folds.
        keep = w > 0
        frame, y, w = frame.loc[keep], y.loc[keep], w.loc[keep]
    t = t1.reindex(frame.index) if t1 is not None else None
    return frame, y, w, t


def _summarize_folds(scores: list[dict]) -> dict:
    if not scores:
        return {}
    acc = np.array([s["accuracy"] for s in scores])
    auc = np.array([s["auc"] for s in scores])
    base = np.array([s["base_rate"] for s in scores])
    return {
        "folds": len(scores),
        "accuracy_mean": round(float(acc.mean()), 4),
        "accuracy_std": round(float(acc.std()), 4),
        "auc_mean": round(float(auc.mean()), 4),
        "auc_std": round(float(auc.std()), 4),
        "base_rate": round(float(base.mean()), 4),
        "edge_over_base": round(float(acc.mean() - base.mean()), 4),
        "per_fold": scores,
    }
