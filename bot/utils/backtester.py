"""Walk-forward backtester — realistic historical simulation.

Unlike naive backtests, this uses walk-forward analysis:
  1. Train on window [0, T]
  2. Test on window [T, T+step]
  3. Slide forward and repeat

This prevents look-ahead bias and gives realistic performance estimates.
"""

import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd

from bot.ml.features import FeatureEngine
from bot.ml.model import EnsembleModel
from bot.trading.paper_trader import PaperTrader
from bot.trading.position_sizer import DynamicPositionSizer
from bot.analysis.regime import RegimeDetector
from bot.ml.rl_agent import KellyCriterionSizer

logger = logging.getLogger("trading_bot")


class WalkForwardBacktester:
    """Walk-forward backtesting engine."""

    def __init__(self, config: dict):
        self.config = config
        self.feature_engine = FeatureEngine(config)
        self.regime_detector = RegimeDetector(config)
        self.position_sizer = DynamicPositionSizer(config)
        self.kelly = KellyCriterionSizer(kelly_fraction=0.25)

    def run(
        self,
        df: pd.DataFrame,
        train_window: int = 5000,
        test_window: int = 500,
        step: int = 500,
    ) -> dict:
        """Run walk-forward backtest.

        Args:
            df: Full OHLCV DataFrame
            train_window: Number of candles for training
            test_window: Number of candles for testing
            step: Step size for sliding window

        Returns:
            dict with performance metrics and trade list
        """
        logger.info("=" * 60)
        logger.info("  WALK-FORWARD BACKTEST")
        logger.info(f"  Data: {len(df)} candles")
        logger.info(f"  Train window: {train_window} | Test: {test_window} | Step: {step}")
        logger.info("=" * 60)

        paper = PaperTrader(self.config)
        all_signals = []
        start_time = time.time()

        n_folds = 0
        i = train_window

        while i + test_window <= len(df):
            n_folds += 1
            train_df = df.iloc[i - train_window : i]
            test_df = df.iloc[i : i + test_window]

            logger.info(
                f"Fold {n_folds}: train [{i-train_window}:{i}] → test [{i}:{i+test_window}]"
            )

            # Build features on training data
            train_feat = self.feature_engine.build_features(train_df)
            feature_cols = self.feature_engine.get_feature_columns(train_feat)

            # Train a fresh model for this fold
            model = EnsembleModel(self.config)
            try:
                model.train(train_feat, feature_cols)
            except Exception as e:
                logger.warning(f"Training failed on fold {n_folds}: {e}")
                i += step
                continue

            # Test on out-of-sample data
            for j in range(len(test_df)):
                idx = i + j
                if idx < train_window:
                    continue

                # Build features up to current point (no future data)
                lookback = df.iloc[max(0, idx - 200) : idx + 1]
                try:
                    feat = self.feature_engine.build_features(lookback)
                except Exception:
                    continue

                if len(feat) < 2:
                    continue

                current_price = float(lookback["close"].iloc[-1])
                symbol = self.config["trading"]["primary_symbol"]

                # Check exits on existing positions
                paper.check_exits(symbol, current_price)

                # Get ML prediction
                try:
                    pred = model.predict(feat)
                except Exception:
                    continue

                confidence = pred["confidence"]
                min_conf = self.config.get("model", {}).get("min_confidence", 0.65)

                if confidence < min_conf:
                    continue

                # Get regime
                regime_result = self.regime_detector.detect(lookback)
                regime = regime_result["regime"]

                # Position sizing
                sizing = self.position_sizer.compute(
                    portfolio_value=paper.balance,
                    model_confidence=confidence,
                    predicted_return=pred["predicted_return"],
                    regime=regime,
                    regime_confidence=regime_result["confidence"],
                    kelly_fraction=self.kelly.get_kelly_size(),
                )

                if sizing["position_pct"] == 0:
                    continue

                # Check risk-reward
                min_rr = self.risk_config_get("min_risk_reward_ratio", 2.0)
                predicted_return_abs = abs(pred["predicted_return"])
                risk_pct = self.config.get("risk", {}).get("trailing_stop_pct", 1.5)
                if predicted_return_abs > 0 and (predicted_return_abs * 100 / risk_pct) < min_rr:
                    continue

                direction = sizing["direction"]
                existing = paper.get_open_positions(symbol)

                # Open position if none exists in same direction
                if not any(p.side == direction for p in existing):
                    amount = (paper.balance * sizing["position_pct"] / 100) / current_price
                    if amount > 0:
                        paper.open_position(
                            symbol=symbol,
                            side=direction,
                            amount=amount,
                            price=current_price,
                            leverage=sizing["leverage"],
                            stop_loss_pct=risk_pct,
                            take_profit_pct=risk_pct * min_rr,
                            trailing_stop_pct=self.config.get("risk", {}).get("trailing_stop_pct", 1.5),
                        )
                        all_signals.append({
                            "idx": idx,
                            "price": current_price,
                            "direction": direction,
                            "confidence": confidence,
                            "position_pct": sizing["position_pct"],
                            "leverage": sizing["leverage"],
                        })

                        # Update Kelly
                        if len(paper.trade_journal) > 0:
                            last_trade = paper.trade_journal[-1]
                            self.kelly.update(last_trade.pnl)

                # Close opposite positions
                for pos in existing:
                    if pos.side != direction and confidence > 0.7:
                        paper.close_position(pos, current_price, "signal_reversal")

            i += step

        elapsed = time.time() - start_time
        logger.info(f"Backtest completed in {elapsed:.0f}s ({n_folds} folds)")

        paper.print_stats()
        stats = paper.get_stats()
        stats["n_folds"] = n_folds
        stats["signals"] = all_signals
        stats["elapsed_seconds"] = elapsed

        return stats

    def risk_config_get(self, key, default):
        return self.config.get("risk", {}).get(key, default)
