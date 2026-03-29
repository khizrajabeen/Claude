"""Signal engine — orchestrates ML prediction, order flow, regime, and position sizing
into actionable trading decisions.

This is the brain of the bot. Each cycle:
  1. Fetch fresh data
  2. Build features (including orderbook + trade flow)
  3. Get ML ensemble prediction
  4. Detect market regime
  5. Analyze order flow
  6. Compute position size (Kelly + RL + regime + confidence)
  7. Validate risk-reward ratio
  8. Execute via paper or live trader
"""

import logging
import time

import numpy as np

from bot.exchange import ExchangeClient
from bot.ml.data_pipeline import DataPipeline
from bot.ml.features import FeatureEngine
from bot.ml.model import EnsembleModel
from bot.ml.rl_agent import RLPositionSizer, KellyCriterionSizer
from bot.analysis.orderflow import OrderFlowAnalyzer
from bot.analysis.regime import RegimeDetector
from bot.trading.position_sizer import DynamicPositionSizer

logger = logging.getLogger("trading_bot")

# Timeframe → seconds
TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "12h": 43200, "1d": 86400,
}


class SignalEngine:
    """Core signal generation + execution loop."""

    def __init__(self, config: dict, trader, exchange: ExchangeClient):
        """
        Args:
            config: Full bot config
            trader: PaperTrader or LiveTrader instance
            exchange: ExchangeClient instance
        """
        self.config = config
        self.trader = trader
        self.exchange = exchange

        self.data_pipeline = DataPipeline(exchange, config)
        self.feature_engine = FeatureEngine(config)
        self.model = EnsembleModel(config)
        self.rl_sizer = RLPositionSizer(config)
        self.orderflow = OrderFlowAnalyzer(config)
        self.regime_detector = RegimeDetector(config)
        self.position_sizer = DynamicPositionSizer(config)
        self.kelly = KellyCriterionSizer(kelly_fraction=0.25)

        self.symbol = config["trading"]["primary_symbol"]
        self.timeframe = config["data"]["timeframes"][0]
        self.running = False
        self.cycle_count = 0
        self.last_train_time = 0.0

    def initialize(self):
        """Load trained models. Must be called before run()."""
        logger.info("Loading ML models...")
        self.model.load()
        if not self.model.is_trained:
            logger.warning("No trained model found — run 'train' mode first!")
            return False

        self.rl_sizer.load()
        self.last_train_time = time.time()
        logger.info("Signal engine initialized")
        return True

    def run_cycle(self):
        """Execute one complete trading cycle."""
        self.cycle_count += 1
        logger.info(f"━━━ Cycle {self.cycle_count} ━━━")

        try:
            # 1. Fetch data
            df = self.data_pipeline.fetch_historical(self.symbol, self.timeframe, days=7)
            if len(df) < 100:
                logger.warning("Insufficient data, skipping cycle")
                return

            current_price = float(df["close"].iloc[-1])
            logger.info(f"{self.symbol}: ${current_price:,.2f}")

            # 2. Check exits on existing positions
            closed = self.trader.check_exits(self.symbol, current_price)
            for trade in closed:
                self.kelly.update(trade.pnl)

            # Check if trading is halted
            if hasattr(self.trader, "is_halted") and self.trader.is_halted:
                logger.warning("Trading halted due to drawdown limit")
                return
            if hasattr(self.trader, "kill_switch") and self.trader.kill_switch:
                logger.error("Kill switch active")
                return

            # 3. Fetch orderbook + trades for order flow
            orderbook = None
            trades_df = None
            try:
                orderbook = self.data_pipeline.fetch_orderbook(self.symbol)
                trades_df = self.data_pipeline.fetch_recent_trades(self.symbol)
            except Exception as e:
                logger.debug(f"Orderbook/trades unavailable: {e}")

            # 4. Build features
            higher_tf = self.config["data"]["timeframes"][-1] if len(self.config["data"]["timeframes"]) > 1 else None
            higher_df = None
            if higher_tf and higher_tf != self.timeframe:
                try:
                    higher_df = self.data_pipeline.fetch_historical(self.symbol, higher_tf, days=30)
                except Exception:
                    pass

            feat_df = self.feature_engine.build_features(
                df, orderbook=orderbook, trades_df=trades_df, higher_tf_df=higher_df,
            )

            # 5. ML prediction
            prediction = self.model.predict(feat_df)
            confidence = prediction["confidence"]
            predicted_return = prediction["predicted_return"]
            buy_prob = prediction["buy_probability"]

            logger.info(
                f"ML Signal: {'BUY' if prediction['direction'] == 1 else 'SELL'} "
                f"| conf={confidence:.3f} | buy_prob={buy_prob:.3f} "
                f"| pred_return={predicted_return:+.4f}"
            )
            logger.info(f"  Sub-models: {prediction['sub_predictions']}")

            # 6. Market regime
            regime_result = self.regime_detector.detect(df)
            regime = regime_result["regime"]
            logger.info(f"Regime: {regime.value} (conf={regime_result['confidence']:.2f})")

            # 7. Order flow analysis
            if orderbook:
                of_book = self.orderflow.analyze_orderbook(orderbook)
                logger.info(f"Orderbook: imbalance={of_book.get('imbalance', 0):+.3f}")

            if trades_df is not None and not trades_df.empty:
                of_trades = self.orderflow.analyze_trades(trades_df)
                logger.info(
                    f"Order flow: CVD={of_trades['cvd_normalized']:+.3f} "
                    f"| whales={of_trades['whale_trades']} (bias={of_trades['whale_bias']:+.2f})"
                )

            # 8. Position sizing
            min_conf = self.config.get("model", {}).get("min_confidence", 0.65)
            if confidence < min_conf:
                logger.info(f"Confidence {confidence:.3f} < threshold {min_conf} — no trade")
                return

            portfolio_value = self._get_portfolio_value()

            # RL agent observation
            rl_suggestion = None
            if self.rl_sizer.is_trained:
                obs = np.array([
                    confidence,
                    predicted_return,
                    portfolio_value / 10000,  # Normalized
                    0,  # Current position (simplified)
                    0,  # Unrealized PnL
                    float(feat_df.get("realized_vol_20", pd.Series([0])).iloc[-1]) if "realized_vol_20" in feat_df.columns else 0,
                    self.trader.total_drawdown_pct / 100 if hasattr(self.trader, "total_drawdown_pct") else 0,
                    0.5,  # Win rate placeholder
                    1 if regime.value.startswith("trending") else 0,
                    0,
                ], dtype=np.float32)
                rl_suggestion = self.rl_sizer.get_position_size(obs)
                logger.info(f"RL suggestion: {rl_suggestion:+.3f}")

            sizing = self.position_sizer.compute(
                portfolio_value=portfolio_value,
                model_confidence=confidence,
                predicted_return=predicted_return,
                regime=regime,
                regime_confidence=regime_result["confidence"],
                kelly_fraction=self.kelly.get_kelly_size(),
                rl_suggestion=rl_suggestion,
                current_drawdown_pct=self.trader.total_drawdown_pct if hasattr(self.trader, "total_drawdown_pct") else 0,
            )

            if sizing["position_pct"] == 0:
                return

            # 9. Risk-reward check
            risk_pct = self.config.get("risk", {}).get("trailing_stop_pct", 1.5)
            min_rr = self.config.get("risk", {}).get("min_risk_reward_ratio", 2.0)
            reward_pct = abs(predicted_return) * 100
            if risk_pct > 0 and reward_pct / risk_pct < min_rr:
                logger.info(
                    f"Risk-reward {reward_pct/risk_pct:.2f} < min {min_rr} — skipping"
                )
                return

            # 10. Execute trade
            direction = sizing["direction"]
            existing = self.trader.get_open_positions(self.symbol)

            # Close opposite positions first
            for pos in existing:
                if pos.side != direction and confidence > 0.7:
                    if hasattr(self.trader, "close_position"):
                        if hasattr(pos, "entry_price"):
                            # PaperTrader needs price
                            self.trader.close_position(pos, current_price, "signal_reversal")
                        else:
                            self.trader.close_position(pos, "signal_reversal")

            # Open new position if none in same direction
            if not any(p.side == direction for p in existing):
                amount = (portfolio_value * sizing["position_pct"] / 100) / current_price
                if amount > 0:
                    self.trader.open_position(
                        symbol=self.symbol,
                        side=direction,
                        amount=amount,
                        price=current_price,
                        leverage=sizing["leverage"],
                        stop_loss_pct=risk_pct,
                        take_profit_pct=risk_pct * min_rr,
                        trailing_stop_pct=self.config.get("risk", {}).get("trailing_stop_pct", 1.5),
                    )

        except Exception as e:
            logger.error(f"Cycle error: {e}", exc_info=True)

    def run(self):
        """Continuous trading loop."""
        if not self.model.is_trained:
            logger.error("Model not trained! Run 'python main.py train' first.")
            return

        interval = TF_SECONDS.get(self.timeframe, 60)
        self.running = True

        logger.info(f"Starting signal engine | {self.symbol} | {self.timeframe} | interval={interval}s")

        while self.running:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)

            if self.running:
                logger.info(f"Next cycle in {interval}s...")
                time.sleep(interval)

    def stop(self):
        self.running = False

    def _get_portfolio_value(self) -> float:
        """Get current portfolio value."""
        if hasattr(self.trader, "balance"):
            return self.trader.balance
        try:
            balance = self.exchange.fetch_balance()
            quote = self.symbol.split("/")[1]
            return float(balance.get("total", {}).get(quote, 0))
        except Exception:
            return self.config.get("paper", {}).get("initial_balance", 10000)
