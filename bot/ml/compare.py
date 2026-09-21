"""Model bake-off under identical, leakage-free conditions.

Every candidate sees the same features, the same triple-barrier labels, the
same sample weights and the same purged folds. Preprocessing is fit inside
each training fold, so no model gets to peek. Without that discipline a
comparison measures which model best exploited the leak.

What is reported, and why these and not accuracy alone:

  accuracy vs majority   Financial labels are close to balanced but never
                         exactly; beating the majority class is the bar.
  AUC                    Ranking quality, which is what position sizing
                         actually consumes.
  log loss               Calibration. A model that is right but wildly
                         overconfident is worse for sizing than one that
                         is right less often and honest about it.
  fold spread            A mean that swings 15 points across folds is not
                         an edge, it is a coin landing well.
  strategy return        The only number that pays: trading the model's
                         calls on the labelled outcomes, after costs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from bot.ml.validation import PurgedKFold
from bot.ml.zoo import ModelSpec, available_models

logger = logging.getLogger("trading_bot")


@dataclass
class ModelResult:
    """One model's measured performance."""

    name: str
    family: str
    folds: int = 0
    accuracy: float = 0.0
    accuracy_std: float = 0.0
    majority: float = 0.0
    edge: float = 0.0
    auc: float = 0.0
    auc_std: float = 0.0
    log_loss: float = 0.0
    brier: float = 0.0
    strategy_return_pct: float = 0.0
    strategy_sharpe: float = 0.0
    trades_taken: int = 0
    fit_seconds: float = 0.0
    error: str = ""
    per_fold: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class ModelComparison:
    """Runs the bake-off and reports a table."""

    def __init__(self, config: dict):
        self.config = config
        validation = config.get("validation", {})
        self.n_splits = int(validation.get("n_splits", 5))
        self.embargo_pct = float(validation.get("embargo_pct", 1.0))
        compare = config.get("compare", {})
        self.confidence_floor = float(compare.get("confidence_floor", 0.55))
        self.cost_bps = float(compare.get("round_trip_cost_bps", 12.0))

    def run(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        returns: pd.Series,
        feature_columns: list[str],
        sample_weight: pd.Series | None = None,
        t1: pd.Series | None = None,
        models: list[str] | None = None,
    ) -> list[ModelResult]:
        """Evaluate each model on identical purged folds."""
        specs = available_models(models)
        if not specs:
            raise RuntimeError("No models available to compare")

        X, y, w, t, r = _align(features, labels, returns, feature_columns,
                               sample_weight, t1)
        if len(X) < self.n_splits * 30:
            raise ValueError(
                f"{len(X)} usable rows is too few for a {self.n_splits}-fold comparison"
            )

        if t is None:
            logger.warning(
                "No label end times — folds cannot be purged and every score "
                "below will be optimistic"
            )

        cv = PurgedKFold(n_splits=self.n_splits, t1=t, embargo_pct=self.embargo_pct)
        folds = list(cv.split(X))
        logger.info(
            "Comparing %d models on %d rows x %d features over %d purged folds",
            len(specs), len(X), len(feature_columns), len(folds),
        )

        results = [self._evaluate(spec, X, y, w, r, folds) for spec in specs]
        results.sort(key=lambda res: (-res.auc, -res.edge))
        self._log_table(results)
        return results

    # ── Per model ─────────────────────────────────────────────

    def _evaluate(self, spec: ModelSpec, X, y, w, r, folds) -> ModelResult:
        from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
        from sklearn.preprocessing import RobustScaler

        result = ModelResult(name=spec.name, family=spec.family)
        started = time.time()

        accuracies, aucs, losses, briers = [], [], [], []
        pnl_by_fold = []

        for index, (train_idx, test_idx) in enumerate(folds, start=1):
            y_train = y.iloc[train_idx].to_numpy()
            y_test = y.iloc[test_idx].to_numpy()
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            X_train = np.nan_to_num(X.iloc[train_idx].to_numpy(), nan=0.0)
            X_test = np.nan_to_num(X.iloc[test_idx].to_numpy(), nan=0.0)

            # Scaling is fit on the training fold only. Linear models and
            # nets need it; trees are indifferent.
            if spec.scale_features:
                scaler = RobustScaler().fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

            try:
                model = spec.build()
                weights = w.iloc[train_idx].to_numpy() if w is not None else None
                _fit(model, X_train, y_train, weights)
                proba = _predict_proba(model, X_test)
            except Exception as e:
                result.error = f"{type(e).__name__}: {e}"[:160]
                logger.warning("  %s failed on fold %d: %s", spec.name, index, e)
                continue

            predicted = (proba > 0.5).astype(int)
            accuracies.append(float((predicted == y_test).mean()))
            if len(np.unique(y_test)) > 1:
                aucs.append(float(roc_auc_score(y_test, proba)))
            clipped = np.clip(proba, 1e-6, 1 - 1e-6)
            losses.append(float(log_loss(y_test, clipped, labels=[0, 1])))
            briers.append(float(brier_score_loss(y_test, clipped)))

            pnl_by_fold.append(self._paper_pnl(proba, r.iloc[test_idx].to_numpy()))
            result.per_fold.append({
                "fold": index, "n_test": len(test_idx),
                "accuracy": round(accuracies[-1], 4),
                "auc": round(aucs[-1], 4) if aucs else None,
            })

        result.fit_seconds = round(time.time() - started, 2)
        if not accuracies:
            result.error = result.error or "no fold produced a score"
            return result

        result.folds = len(accuracies)
        result.accuracy = round(float(np.mean(accuracies)), 4)
        result.accuracy_std = round(float(np.std(accuracies)), 4)
        result.majority = round(float(max(y.mean(), 1 - y.mean())), 4)
        result.edge = round(result.accuracy - result.majority, 4)
        result.auc = round(float(np.mean(aucs)), 4) if aucs else 0.5
        result.auc_std = round(float(np.std(aucs)), 4) if aucs else 0.0
        result.log_loss = round(float(np.mean(losses)), 4)
        result.brier = round(float(np.mean(briers)), 4)

        trades = [t for fold in pnl_by_fold for t in fold]
        if trades:
            series = pd.Series(trades)
            result.trades_taken = len(series)
            result.strategy_return_pct = round(float(series.sum()) * 100, 3)
            result.strategy_sharpe = (
                round(float(series.mean() / series.std() * np.sqrt(252)), 3)
                if series.std() > 0 else 0.0
            )
        return result

    def _paper_pnl(self, proba: np.ndarray, realised: np.ndarray) -> list[float]:
        """Trade the model's confident calls against the labelled outcomes.

        Not a backtest — there is no sizing, stop or queue here — but it
        answers the question accuracy cannot: would acting on this have
        paid after costs?
        """
        cost = self.cost_bps / 10_000
        trades = []
        for probability, outcome in zip(proba, realised):
            if not np.isfinite(outcome):
                continue
            if probability >= self.confidence_floor:
                trades.append(float(outcome) - cost)
            elif probability <= 1 - self.confidence_floor:
                trades.append(-float(outcome) - cost)
        return trades

    def _log_table(self, results: list[ModelResult]) -> None:
        logger.info("═" * 92)
        logger.info("  MODEL COMPARISON — identical features, labels and purged folds")
        logger.info("═" * 92)
        logger.info(
            "  %-14s %-9s %8s %8s %7s %9s %8s %9s %7s",
            "model", "family", "acc", "vs base", "AUC", "logloss", "ret %", "sharpe", "secs",
        )
        logger.info("  " + "-" * 88)
        for res in results:
            if res.error and not res.folds:
                logger.info("  %-14s %-9s  failed: %s", res.name, res.family, res.error)
                continue
            logger.info(
                "  %-14s %-9s %8.4f %+8.4f %7.4f %9.4f %8.2f %9.2f %7.1f",
                res.name, res.family, res.accuracy, res.edge, res.auc,
                res.log_loss, res.strategy_return_pct, res.strategy_sharpe,
                res.fit_seconds,
            )
        logger.info("═" * 92)

        best = next((r for r in results if r.folds and r.name != "majority"), None)
        baseline = next((r for r in results if r.name == "majority"), None)
        if best is None:
            return

        if best.auc <= 0.52 or best.edge <= 0.005:
            logger.info(
                "  Verdict: nothing here beats the baseline by enough to trade. "
                "AUC %.4f is within noise of 0.5.", best.auc,
            )
        else:
            logger.info(
                "  Best by AUC: %s (%s) — AUC %.4f, accuracy %+.4f over majority, "
                "fold spread ±%.4f", best.name, best.family, best.auc,
                best.edge, best.accuracy_std,
            )
            if best.accuracy_std > abs(best.edge):
                logger.info(
                    "  Caution: fold-to-fold spread exceeds the edge — this is not "
                    "yet evidence of skill."
                )
        if baseline and baseline.folds:
            logger.info("  Baseline (majority class): accuracy %.4f", baseline.accuracy)


# ── helpers ──────────────────────────────────────────────────

def _fit(model, X, y, weights):
    """Fit with sample weights where the estimator supports them."""
    if weights is None:
        model.fit(X, y)
        return
    try:
        model.fit(X, y, sample_weight=weights)
    except TypeError:
        # Not every estimator takes weights; unweighted is better than none.
        model.fit(X, y)


def _predict_proba(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1.0 / (1.0 + np.exp(-scores))
    return model.predict(X).astype(float)


def _align(features, labels, returns, feature_columns, sample_weight, t1):
    """Drop unresolved rows and line everything up on one index."""
    columns = [c for c in feature_columns if c in features.columns]
    frame = features[columns].replace([np.inf, -np.inf], np.nan)

    mask = labels.reindex(frame.index).notna()
    frame = frame.loc[mask]
    y = labels.reindex(frame.index).astype(int)
    r = returns.reindex(frame.index)
    w = sample_weight.reindex(frame.index) if sample_weight is not None else None

    if w is not None:
        keep = w > 0
        frame, y, r, w = frame.loc[keep], y.loc[keep], r.loc[keep], w.loc[keep]

    t = t1.reindex(frame.index) if t1 is not None else None
    return frame, y, w, t, r
