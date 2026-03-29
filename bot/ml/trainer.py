"""Model training pipeline — orchestrates data collection, feature engineering,
model training, and RL agent training.

Designed to run on HPC: uses all available GPU and CPU cores.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bot.exchange import ExchangeClient
from bot.ml.data_pipeline import DataPipeline
from bot.ml.features import FeatureEngine
from bot.ml.model import EnsembleModel
from bot.ml.rl_agent import RLPositionSizer

logger = logging.getLogger("trading_bot")


class TrainingPipeline:
    """End-to-end training pipeline."""

    def __init__(self, config: dict):
        self.config = config
        self.exchange = ExchangeClient(config)
        self.data_pipeline = DataPipeline(self.exchange, config)
        self.feature_engine = FeatureEngine(config)
        self.model = EnsembleModel(config)
        self.rl_sizer = RLPositionSizer(config)

    def run_full_training(self, symbol: str | None = None):
        """Run the complete training pipeline.

        Steps:
            1. Collect historical data (multi-timeframe)
            2. Engineer features
            3. Train ensemble model (XGBoost + LightGBM + LSTM + Transformer)
            4. Train RL position sizing agent
        """
        symbol = symbol or self.config["trading"]["primary_symbol"]
        timeframes = self.config["data"]["timeframes"]
        primary_tf = timeframes[0] if timeframes else "1h"
        higher_tf = timeframes[-1] if len(timeframes) > 1 else None

        logger.info("=" * 60)
        logger.info("  TRAINING PIPELINE START")
        logger.info("=" * 60)
        logger.info(f"  Symbol   : {symbol}")
        logger.info(f"  Device   : {self.model.device}")
        if torch.cuda.is_available():
            logger.info(f"  GPU      : {torch.cuda.get_device_name(0)}")
            logger.info(f"  GPU Mem  : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
        logger.info("=" * 60)

        start = time.time()

        # ── Step 1: Collect Data ──────────────────────────────
        logger.info("[1/4] Collecting historical data...")
        all_data = self.data_pipeline.collect_all_timeframes(symbol)
        primary_df = all_data.get(primary_tf)
        higher_df = all_data.get(higher_tf) if higher_tf else None

        if primary_df is None or len(primary_df) < 200:
            raise ValueError(f"Insufficient data: got {len(primary_df) if primary_df is not None else 0} candles")

        # Clean
        primary_df = self.data_pipeline.clean(primary_df)
        if higher_df is not None:
            higher_df = self.data_pipeline.clean(higher_df)

        logger.info(f"Data: {len(primary_df)} candles ({primary_tf})")

        # ── Step 2: Feature Engineering ───────────────────────
        logger.info("[2/4] Engineering features...")

        # Get orderbook + trades for the latest snapshot
        try:
            orderbook = self.data_pipeline.fetch_orderbook(symbol)
            trades_df = self.data_pipeline.fetch_recent_trades(symbol)
        except Exception as e:
            logger.warning(f"Could not fetch orderbook/trades: {e}")
            orderbook = None
            trades_df = None

        feat_df = self.feature_engine.build_features(
            primary_df,
            orderbook=orderbook,
            trades_df=trades_df,
            higher_tf_df=higher_df,
        )

        feature_cols = self.feature_engine.get_feature_columns(feat_df)
        logger.info(f"Generated {len(feature_cols)} features")

        # ── Step 3: Train Ensemble ────────────────────────────
        logger.info("[3/4] Training ensemble model...")
        self.model.train(feat_df, feature_cols)

        # ── Step 4: Train RL Agent ────────────────────────────
        logger.info("[4/4] Training RL position sizer...")
        price_data = primary_df["close"].values
        # Use top 3 features as RL observations
        feat_clean = feat_df[feature_cols].dropna()
        if len(feat_clean) > len(price_data):
            feat_clean = feat_clean.iloc[: len(price_data)]

        # Pad or trim to match price data length
        rl_features = np.zeros((len(price_data), 3))
        feat_vals = feat_clean.values
        n = min(len(feat_vals), len(price_data))
        rl_features[-n:, :min(3, feat_vals.shape[1])] = feat_vals[-n:, :3]

        self.rl_sizer.train(price_data, rl_features)

        elapsed = time.time() - start
        logger.info("=" * 60)
        logger.info(f"  TRAINING COMPLETE ({elapsed:.0f}s)")
        logger.info("=" * 60)

        return self.model, self.rl_sizer

    def retrain_if_needed(self, last_train_time: float) -> bool:
        """Check if it's time to retrain and do so."""
        interval_hours = self.config.get("model", {}).get("retrain_interval_hours", 24)
        elapsed_hours = (time.time() - last_train_time) / 3600

        if elapsed_hours >= interval_hours:
            logger.info(f"Retraining (last train was {elapsed_hours:.1f}h ago)...")
            self.run_full_training()
            return True
        return False
