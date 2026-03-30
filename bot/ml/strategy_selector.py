"""Multi-strategy system with automatic selection.

Trains separate ML models for different market conditions and
automatically selects the best strategy for the current regime.

Strategies:
  1. Momentum — works in trending markets (Hurst > 0.55)
  2. Mean Reversion — works in ranging markets (Hurst < 0.45)
  3. Breakout — works in quiet markets transitioning to volatile
  4. Scalp — works on short timeframes with high volume
  5. News/Sentiment — triggered by strong news signals

The StrategySelector evaluates all strategies and picks the one
with the highest expected edge for current conditions.
"""

import logging
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from bot.ml.model import EnsembleModel
from bot.analysis.regime import MarketRegime

logger = logging.getLogger("trading_bot")


class StrategyType(Enum):
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    BREAKOUT = "breakout"
    SCALP = "scalp"
    NEWS = "news"


@dataclass
class StrategyResult:
    """Result from a single strategy evaluation."""
    strategy: StrategyType
    direction: str  # "long", "short", "none"
    confidence: float
    predicted_return: float
    edge_score: float  # How well this strategy fits current conditions


class StrategyConfig:
    """Strategy-specific feature engineering and model tuning."""

    # Which features matter most for each strategy type
    FEATURE_PRIORITIES = {
        StrategyType.MOMENTUM: [
            "roc_", "dist_ema_", "ema_cross_", "macd_",
            "htf_trend", "adx", "hurst",
        ],
        StrategyType.MEAN_REVERSION: [
            "zscore_", "range_position_", "stoch_", "williams_r_",
            "cci_", "mfi_", "ob_imbalance",
        ],
        StrategyType.BREAKOUT: [
            "vol_ratio_", "natr_", "dist_resistance_", "dist_support_",
            "vol_spike", "parkinson_vol_", "garman_klass_",
        ],
        StrategyType.SCALP: [
            "return_1", "return_2", "ob_imbalance", "ob_spread_bps",
            "tf_buy_ratio", "tf_cvd", "vol_sma_5",
        ],
        StrategyType.NEWS: [
            "news_sentiment_", "news_signal_strength", "news_volume_",
            "news_bullish_ratio", "news_bearish_ratio",
        ],
    }

    # Optimal model params per strategy
    MODEL_PARAMS = {
        StrategyType.MOMENTUM: {
            "xgb": {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.03},
            "lgb": {"max_depth": 8, "n_estimators": 600, "learning_rate": 0.03},
            "sequence_length": 90,
        },
        StrategyType.MEAN_REVERSION: {
            "xgb": {"max_depth": 6, "n_estimators": 400, "learning_rate": 0.05},
            "lgb": {"max_depth": 6, "n_estimators": 400, "learning_rate": 0.05},
            "sequence_length": 40,
        },
        StrategyType.BREAKOUT: {
            "xgb": {"max_depth": 7, "n_estimators": 500, "learning_rate": 0.04},
            "lgb": {"max_depth": 7, "n_estimators": 500, "learning_rate": 0.04},
            "sequence_length": 60,
        },
        StrategyType.SCALP: {
            "xgb": {"max_depth": 5, "n_estimators": 300, "learning_rate": 0.08},
            "lgb": {"max_depth": 5, "n_estimators": 300, "learning_rate": 0.08},
            "sequence_length": 20,
        },
        StrategyType.NEWS: {
            "xgb": {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.1},
            "lgb": {"max_depth": 5, "n_estimators": 200, "learning_rate": 0.1},
            "sequence_length": 30,
        },
    }


class StrategySelector:
    """Trains and evaluates multiple strategies, selects the best one."""

    def __init__(self, config: dict):
        self.config = config
        self.models: dict[StrategyType, EnsembleModel] = {}
        self.performance_history: dict[StrategyType, list[float]] = {
            st: [] for st in StrategyType
        }

    def train_all_strategies(
        self,
        feat_df: pd.DataFrame,
        feature_columns: list[str],
    ):
        """Train a separate model for each strategy type."""
        logger.info("Training multi-strategy models...")

        for strategy_type in StrategyType:
            logger.info(f"  Training {strategy_type.value} strategy...")

            # Filter features relevant to this strategy
            priority_prefixes = StrategyConfig.FEATURE_PRIORITIES.get(
                strategy_type, []
            )
            relevant_cols = self._filter_features(
                feature_columns, priority_prefixes
            )

            if len(relevant_cols) < 10:
                # If too few strategy-specific features, use all
                relevant_cols = feature_columns

            # Create model with strategy-specific config
            model_config = dict(self.config)
            params = StrategyConfig.MODEL_PARAMS.get(strategy_type, {})
            seq_len = params.get("sequence_length", 60)
            model_config.setdefault("model", {})["sequence_length"] = seq_len

            model = EnsembleModel(model_config)

            try:
                model.train(feat_df, relevant_cols)
                self.models[strategy_type] = model
                logger.info(f"  {strategy_type.value}: trained on {len(relevant_cols)} features")
            except Exception as e:
                logger.warning(f"  {strategy_type.value}: training failed ({e})")

    def select_best_strategy(
        self,
        feat_df: pd.DataFrame,
        regime: MarketRegime,
        regime_confidence: float,
        news_signal: dict | None = None,
    ) -> StrategyResult | None:
        """Evaluate all strategies and return the best one.

        Selection criteria:
          1. Regime fitness (is this strategy suited to current regime?)
          2. Model confidence (how sure is the model?)
          3. Historical performance (has this strategy been working lately?)
          4. Predicted return magnitude
        """
        if not self.models:
            logger.warning("No trained strategies available")
            return None

        results: list[StrategyResult] = []

        for strategy_type, model in self.models.items():
            if not model.is_trained:
                continue

            try:
                pred = model.predict(feat_df)
            except Exception:
                continue

            # Base confidence from model
            confidence = pred["confidence"]
            predicted_return = pred["predicted_return"]

            # Regime fitness score
            fitness = self._regime_fitness(strategy_type, regime)

            # Historical performance bonus
            perf_bonus = self._performance_bonus(strategy_type)

            # News boost for news strategy
            news_boost = 0.0
            if strategy_type == StrategyType.NEWS and news_signal:
                news_boost = news_signal.get("confidence", 0) * 0.3

            # Combined edge score
            edge_score = (
                confidence * 0.35
                + fitness * 0.25
                + abs(predicted_return) * 100 * 0.15
                + perf_bonus * 0.15
                + news_boost * 0.10
            )

            direction = "long" if pred["direction"] == 1 else "short"

            # News strategy overrides direction based on sentiment
            if strategy_type == StrategyType.NEWS and news_signal:
                direction = news_signal.get("direction", direction)
                confidence = max(confidence, news_signal.get("confidence", 0))

            results.append(StrategyResult(
                strategy=strategy_type,
                direction=direction,
                confidence=confidence,
                predicted_return=predicted_return,
                edge_score=edge_score,
            ))

        if not results:
            return None

        # Select best by edge score
        best = max(results, key=lambda r: r.edge_score)

        logger.info(
            f"Strategy selection: {best.strategy.value} "
            f"(edge={best.edge_score:.3f}, conf={best.confidence:.3f}) | "
            + " | ".join(
                f"{r.strategy.value}={r.edge_score:.3f}" for r in results
            )
        )

        return best

    def record_outcome(self, strategy: StrategyType, pnl: float):
        """Record trade outcome for strategy performance tracking."""
        self.performance_history[strategy].append(pnl)
        # Keep last 50 trades per strategy
        if len(self.performance_history[strategy]) > 50:
            self.performance_history[strategy] = self.performance_history[strategy][-50:]

    def _regime_fitness(self, strategy: StrategyType, regime: MarketRegime) -> float:
        """Score how well a strategy fits the current regime (0 to 1)."""
        fitness_map = {
            StrategyType.MOMENTUM: {
                MarketRegime.TRENDING_UP: 1.0,
                MarketRegime.TRENDING_DOWN: 0.8,
                MarketRegime.RANGING: 0.2,
                MarketRegime.VOLATILE: 0.4,
                MarketRegime.QUIET: 0.3,
            },
            StrategyType.MEAN_REVERSION: {
                MarketRegime.TRENDING_UP: 0.2,
                MarketRegime.TRENDING_DOWN: 0.2,
                MarketRegime.RANGING: 1.0,
                MarketRegime.VOLATILE: 0.5,
                MarketRegime.QUIET: 0.7,
            },
            StrategyType.BREAKOUT: {
                MarketRegime.TRENDING_UP: 0.5,
                MarketRegime.TRENDING_DOWN: 0.5,
                MarketRegime.RANGING: 0.4,
                MarketRegime.VOLATILE: 0.7,
                MarketRegime.QUIET: 1.0,
            },
            StrategyType.SCALP: {
                MarketRegime.TRENDING_UP: 0.6,
                MarketRegime.TRENDING_DOWN: 0.6,
                MarketRegime.RANGING: 0.8,
                MarketRegime.VOLATILE: 1.0,
                MarketRegime.QUIET: 0.3,
            },
            StrategyType.NEWS: {
                MarketRegime.TRENDING_UP: 0.7,
                MarketRegime.TRENDING_DOWN: 0.7,
                MarketRegime.RANGING: 0.5,
                MarketRegime.VOLATILE: 0.9,
                MarketRegime.QUIET: 0.6,
            },
        }
        return fitness_map.get(strategy, {}).get(regime, 0.5)

    def _performance_bonus(self, strategy: StrategyType) -> float:
        """Recent performance bonus (0 to 1)."""
        history = self.performance_history.get(strategy, [])
        if len(history) < 5:
            return 0.5  # Neutral if not enough data

        recent = history[-10:]
        win_rate = sum(1 for p in recent if p > 0) / len(recent)
        return win_rate

    def _filter_features(
        self, all_features: list[str], priority_prefixes: list[str]
    ) -> list[str]:
        """Filter features by priority prefixes."""
        selected = []
        for feat in all_features:
            for prefix in priority_prefixes:
                if feat.startswith(prefix) or prefix in feat:
                    selected.append(feat)
                    break
        return selected
