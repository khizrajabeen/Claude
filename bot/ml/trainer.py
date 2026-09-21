"""Training pipeline: data → features → triple-barrier labels → purged CV.

The pipeline is deliberately blunt about what it finds. A model that cannot
beat the majority-class rate out of sample is reported as such and refuses
to claim an edge, because the expensive mistake is not a weak model — it is
a weak model that looks strong because the validation leaked.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from bot.exchange import ExchangeClient
from bot.ml.features import FeatureEngine
from bot.ml.labeling import label_balance, sample_weights_by_uniqueness, triple_barrier_labels
from bot.ml.model import MODEL_DIR, EnsembleModel

logger = logging.getLogger("trading_bot")


class TrainingPipeline:
    """Fetch, label, validate and fit."""

    def __init__(self, config: dict):
        self.config = config
        self.feature_engine = FeatureEngine(config)
        self.model = EnsembleModel(config)
        self.labeling = config.get("labeling", {})
        self.validation = config.get("validation", {})
        self.timeframe = config["data"].get("timeframe", "1h")
        self.symbols = list(config["data"].get("symbols", []))

    def compare(self, days: int | None = None, models: list[str] | None = None) -> dict:
        """Run the model bake-off on exactly the data training would use."""
        from bot.ml.compare import ModelComparison

        features, labels, weights, t1, returns = self._assemble(days)
        feature_columns = self.feature_engine.get_feature_columns(features)

        results = ModelComparison(self.config).run(
            features=features, labels=labels, returns=returns,
            feature_columns=feature_columns, sample_weight=weights, t1=t1,
            models=models,
        )

        report = {
            "rows": len(features),
            "features": len(feature_columns),
            "timeframe": self.timeframe,
            "results": [r.to_dict() for r in results],
        }
        self._save_report(report, name="model_comparison.json")
        return report

    def _assemble(self, days: int | None = None):
        """Download, label and pool every symbol into one aligned dataset."""
        days = days or int(self.config["data"].get("train_days", 180))
        frames = self._load(days)
        if not frames:
            raise RuntimeError("No data downloaded — cannot train")

        blocks = []
        for symbol, df in frames.items():
            prepared = self._prepare(symbol, df)
            if prepared is not None:
                blocks.append(prepared)

        if not blocks:
            raise RuntimeError("No symbol produced enough resolved labels to train on")

        features = pd.concat([b[0] for b in blocks]).sort_index()
        labels = pd.concat([b[1] for b in blocks]).sort_index()
        weights = pd.concat([b[2] for b in blocks]).sort_index()
        t1 = pd.concat([b[3] for b in blocks]).sort_index()
        returns = pd.concat([b[4] for b in blocks]).sort_index()
        return features, labels, weights, t1, returns

    def run(self, days: int | None = None) -> dict:
        """Train on every configured symbol pooled together.

        Pooling is deliberate: crypto majors share most of their structure,
        and one symbol rarely supplies enough resolved labels to validate on.
        """
        features, labels, weights, t1, _ = self._assemble(days)
        feature_columns = self.feature_engine.get_feature_columns(features)
        logger.info("Training set: %d rows, %d features",
                    len(features), len(feature_columns))

        report = self.model.fit(
            features=features,
            labels=labels,
            feature_columns=feature_columns,
            sample_weight=weights,
            t1=t1,
            n_splits=int(self.validation.get("n_splits", 5)),
            embargo_pct=float(self.validation.get("embargo_pct", 1.0)),
        )

        report["rows"] = len(features)
        report["timeframe"] = self.timeframe
        self._save_report(report)
        self._log_verdict(report)
        return report

    # ── Steps ─────────────────────────────────────────────────

    def _load(self, days: int) -> dict[str, pd.DataFrame]:
        from bot.analysis.indicators import TIMEFRAME_SECONDS
        from bot.utils.backtester import _paged_ohlcv

        exchange = ExchangeClient(self.config)
        bar_seconds = TIMEFRAME_SECONDS.get(self.timeframe, 3600)
        needed = int(days * 86_400 / bar_seconds)

        frames = {}
        for symbol in self.symbols:
            resolved = exchange.resolve_symbol(symbol)
            if resolved is None:
                logger.warning("%s not listed — skipped", symbol)
                continue
            try:
                df = _paged_ohlcv(exchange, resolved, self.timeframe, needed)
                if len(df) < 300:
                    logger.warning("%s: only %d bars — skipped", symbol, len(df))
                    continue
                frames[symbol] = df
                logger.info("  %-10s %d bars (%s → %s)", symbol, len(df),
                            df.index[0].date(), df.index[-1].date())
            except Exception as e:
                logger.warning("Download failed for %s: %s", symbol, e)
        return frames

    def _prepare(self, symbol: str, df: pd.DataFrame):
        """Features, labels, weights and label end times for one symbol."""
        features = self.feature_engine.build_features(df)

        labels = triple_barrier_labels(
            df,
            atr_period=int(self.labeling.get("atr_period", 14)),
            upper_atr=float(self.labeling.get("upper_barrier_atr", 2.0)),
            lower_atr=float(self.labeling.get("lower_barrier_atr", 2.0)),
            max_holding_bars=int(self.labeling.get("max_holding_bars", 48)),
            side=1,
        )
        balance = label_balance(labels)
        logger.info("  %-10s labels: %s", symbol, balance)
        if balance.get("resolved", 0) < 200:
            logger.warning("  %s: only %d resolved labels — skipped",
                           symbol, balance.get("resolved", 0))
            return None

        weights = sample_weights_by_uniqueness(labels)

        # A "win" is the trade making money, which is the meta-label. The
        # side is fixed at long here; a directional primary model would
        # supply `side` instead and this becomes true meta-labelling.
        target = labels["meta_label"]

        # Make the index unique across symbols before pooling.
        keyed = pd.MultiIndex.from_arrays(
            [features.index, [symbol] * len(features)], names=["timestamp", "symbol"]
        )
        features = features.set_axis(keyed)
        target = target.set_axis(keyed)
        weights = weights.set_axis(keyed)
        t1 = labels["t1"].set_axis(keyed)
        returns = labels["ret"].set_axis(keyed)
        return features, target, weights, t1, returns

    def _save_report(self, report: dict, name: str = "training_report.json") -> None:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = MODEL_DIR / name
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("Training report written to %s", path)

    def _log_verdict(self, report: dict) -> None:
        edge = report.get("edge_over_base", 0.0)
        logger.info("═" * 62)
        logger.info("  TRAINING RESULT")
        logger.info("═" * 62)
        logger.info("  Rows         : %d", report.get("rows", 0))
        logger.info("  Purged folds : %d", report.get("folds", 0))
        logger.info("  Accuracy     : %.4f ± %.4f",
                    report.get("accuracy_mean", 0), report.get("accuracy_std", 0))
        logger.info("  Majority rate: %.4f", report.get("base_rate", 0))
        logger.info("  Edge         : %+.4f", edge)
        logger.info("  AUC          : %.4f", report.get("auc_mean", 0))
        logger.info("─" * 62)
        if edge <= 0.005:
            logger.info("  Verdict: no usable edge. Leave model.enabled: false.")
        elif edge < 0.02:
            logger.info("  Verdict: marginal. Worth paper trading, not sizing up.")
        else:
            logger.info("  Verdict: an edge worth blending in. Set model.enabled: true,")
            logger.info("           then confirm it forward in paper before trusting it.")
        logger.info("═" * 62)
