"""Model training pipeline — multi-coin, multi-strategy.

Orchestrates:
  1. Auto data download for all coins
  2. Feature engineering per coin
  3. Combined multi-coin training (more data = better model)
  4. Multi-strategy training (momentum, mean-reversion, breakout, scalp, news)
  5. RL position sizing agent
  6. Model serialization

Designed for HPC: uses all available GPU and CPU cores.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bot.exchange import ExchangeClient
from bot.ml.data_pipeline import DataPipeline
from bot.ml.auto_data import AutoDataManager
from bot.ml.features import FeatureEngine
from bot.ml.model import EnsembleModel
from bot.ml.strategy_selector import StrategySelector
from bot.ml.rl_agent import RLPositionSizer
from bot.analysis.news_sentiment import NewsSentimentAnalyzer

logger = logging.getLogger("trading_bot")


class TrainingPipeline:
    """End-to-end training pipeline with multi-coin and multi-strategy support."""

    def __init__(self, config: dict):
        self.config = config
        self.exchange = ExchangeClient(config)
        self.data_pipeline = DataPipeline(self.exchange, config)
        self.auto_data = AutoDataManager(config, self.data_pipeline)
        self.feature_engine = FeatureEngine(config)
        self.model = EnsembleModel(config)
        self.strategy_selector = StrategySelector(config)
        self.rl_sizer = RLPositionSizer(config)
        self.news_analyzer = NewsSentimentAnalyzer(config)

    def run_full_training(
        self,
        symbol: str | None = None,
        multi_coin: bool = True,
        multi_strategy: bool = True,
        auto_discover: bool = False,
    ):
        """Run the complete training pipeline.

        Args:
            symbol: Primary symbol (or None for config default)
            multi_coin: Train on multiple coins for better generalization
            multi_strategy: Train separate models per strategy type
            auto_discover: Auto-discover top coins by volume
        """
        symbol = symbol or self.config["trading"]["primary_symbol"]
        timeframes = self.config["data"]["timeframes"]
        primary_tf = timeframes[0] if timeframes else "1h"
        higher_tf = timeframes[-1] if len(timeframes) > 1 else None

        logger.info("=" * 60)
        logger.info("  TRAINING PIPELINE START")
        logger.info("=" * 60)
        logger.info(f"  Primary  : {symbol}")
        logger.info(f"  Multi-coin: {multi_coin}")
        logger.info(f"  Multi-strat: {multi_strategy}")
        logger.info(f"  Device   : {self.model.device}")
        if torch.cuda.is_available():
            logger.info(f"  GPU      : {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU Mem  : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        logger.info("=" * 60)

        start = time.time()

        # ── Step 1: Download Data ─────────────────────────────
        logger.info("[1/6] Downloading data...")

        symbols = self.config["data"].get("symbols", [symbol])
        if auto_discover:
            discovered = self.auto_data.discover_top_coins(top_n=15)
            symbols = list(set(symbols + discovered))
            logger.info(f"Total symbols: {len(symbols)}")

        all_data = self.auto_data.download_all(
            symbols=symbols,
            timeframes=timeframes,
            days=self.config["data"].get("lookback_days", 90),
        )

        # ── Step 2: Feature Engineering ───────────────────────
        logger.info("[2/6] Engineering features...")

        if multi_coin and len(symbols) > 1:
            feat_df, feature_cols = self._build_multicoin_features(
                all_data, symbols, primary_tf, higher_tf
            )
        else:
            feat_df, feature_cols = self._build_single_features(
                all_data, symbol, primary_tf, higher_tf
            )

        # Add news sentiment features
        logger.info("  Adding news sentiment features...")
        try:
            for sym in symbols[:5]:  # Top 5 coins
                news_feats = self.news_analyzer.get_sentiment_features(sym)
                for k, v in news_feats.items():
                    col_name = f"{sym.split('/')[0]}_{k}"
                    feat_df[col_name] = v
                    if col_name not in feature_cols:
                        feature_cols.append(col_name)
        except Exception as e:
            logger.warning(f"News features unavailable: {e}")

        logger.info(f"Total features: {len(feature_cols)} | Samples: {len(feat_df)}")

        # ── Step 3: Train Primary Ensemble ────────────────────
        logger.info("[3/6] Training primary ensemble model...")
        self.model.train(feat_df, feature_cols)

        # ── Step 4: Train Multi-Strategy Models ───────────────
        if multi_strategy:
            logger.info("[4/6] Training multi-strategy models...")
            self.strategy_selector.train_all_strategies(feat_df, feature_cols)
        else:
            logger.info("[4/6] Skipping multi-strategy (disabled)")

        # ── Step 5: Train RL Agent ────────────────────────────
        logger.info("[5/6] Training RL position sizer...")
        # Use primary symbol's price data
        primary_data = all_data.get(symbol, {}).get(primary_tf)
        if primary_data is not None and len(primary_data) > 200:
            price_data = primary_data["close"].values
            feat_clean = feat_df[feature_cols].dropna()
            rl_features = np.zeros((len(price_data), 3))
            n = min(len(feat_clean), len(price_data))
            if n > 0 and feat_clean.shape[1] >= 3:
                rl_features[-n:, :3] = feat_clean.values[-n:, :3]
            self.rl_sizer.train(price_data, rl_features)
        else:
            logger.warning("Insufficient data for RL training")

        # ── Step 6: Validate ──────────────────────────────────
        logger.info("[6/6] Validation check...")
        self._validation_check(feat_df, feature_cols, symbol)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"  TRAINING COMPLETE ({elapsed:.0f}s)")
        logger.info(f"  Models trained on {len(symbols)} coins")
        logger.info(f"  {len(feature_cols)} features")
        if multi_strategy:
            logger.info(f"  {len(self.strategy_selector.models)} strategies")
        logger.info("=" * 60)

        return self.model, self.strategy_selector, self.rl_sizer

    def _build_multicoin_features(
        self, all_data, symbols, primary_tf, higher_tf
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build combined feature DataFrame from multiple coins."""
        all_feat_dfs = []

        for sym in symbols:
            tf_data = all_data.get(sym, {})
            primary_df = tf_data.get(primary_tf)
            if primary_df is None or len(primary_df) < 100:
                continue

            primary_df = self.data_pipeline.clean(primary_df)
            higher_df = tf_data.get(higher_tf) if higher_tf else None
            if higher_df is not None:
                higher_df = self.data_pipeline.clean(higher_df)

            try:
                orderbook = self.data_pipeline.fetch_orderbook(sym)
            except Exception:
                orderbook = None
            try:
                trades_df = self.data_pipeline.fetch_recent_trades(sym)
            except Exception:
                trades_df = None

            feat_df = self.feature_engine.build_features(
                primary_df,
                orderbook=orderbook,
                trades_df=trades_df,
                higher_tf_df=higher_df,
            )
            feat_df["_symbol"] = sym
            all_feat_dfs.append(feat_df)

            logger.info(f"  {sym}: {len(feat_df)} samples, {len(feat_df.columns)} cols")

        if not all_feat_dfs:
            raise ValueError("No coin data produced features")

        # Combine — use intersection of columns
        common_cols = set(all_feat_dfs[0].columns)
        for df in all_feat_dfs[1:]:
            common_cols &= set(df.columns)

        combined = pd.concat(
            [df[list(common_cols)] for df in all_feat_dfs], axis=0
        ).sort_index()

        feature_cols = self.feature_engine.get_feature_columns(combined)
        # Remove internal columns
        feature_cols = [c for c in feature_cols if not c.startswith("_")]

        logger.info(f"  Combined: {len(combined)} samples from {len(all_feat_dfs)} coins")
        return combined, feature_cols

    def _build_single_features(
        self, all_data, symbol, primary_tf, higher_tf
    ) -> tuple[pd.DataFrame, list[str]]:
        """Build features from a single coin."""
        tf_data = all_data.get(symbol, {})
        primary_df = tf_data.get(primary_tf)
        if primary_df is None or len(primary_df) < 200:
            raise ValueError(f"Insufficient data for {symbol}")

        primary_df = self.data_pipeline.clean(primary_df)
        higher_df = tf_data.get(higher_tf) if higher_tf else None
        if higher_df is not None:
            higher_df = self.data_pipeline.clean(higher_df)

        try:
            orderbook = self.data_pipeline.fetch_orderbook(symbol)
        except Exception:
            orderbook = None
        try:
            trades_df = self.data_pipeline.fetch_recent_trades(symbol)
        except Exception:
            trades_df = None

        feat_df = self.feature_engine.build_features(
            primary_df,
            orderbook=orderbook,
            trades_df=trades_df,
            higher_tf_df=higher_df,
        )
        feature_cols = self.feature_engine.get_feature_columns(feat_df)
        return feat_df, feature_cols

    def _validation_check(self, feat_df, feature_cols, symbol):
        """Quick validation that the model produces reasonable predictions."""
        try:
            pred = self.model.predict(feat_df)
            logger.info(
                f"  Validation prediction: "
                f"dir={'BUY' if pred['direction'] == 1 else 'SELL'} "
                f"conf={pred['confidence']:.3f} "
                f"return={pred['predicted_return']:+.4f}"
            )
        except Exception as e:
            logger.warning(f"  Validation failed: {e}")

    def retrain_if_needed(self, last_train_time: float) -> bool:
        """Check if it's time to retrain and do so."""
        interval_hours = self.config.get("model", {}).get("retrain_interval_hours", 24)
        elapsed_hours = (time.time() - last_train_time) / 3600

        if elapsed_hours >= interval_hours:
            logger.info(f"Retraining (last train was {elapsed_hours:.1f}h ago)...")
            self.run_full_training()
            return True
        return False
