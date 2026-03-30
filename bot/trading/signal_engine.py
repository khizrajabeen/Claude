"""Signal engine — the brain of the bot.

Orchestrates ML prediction, multi-strategy selection, news sentiment,
order flow, regime detection, and position sizing into trade decisions.

Each cycle:
  1. Fetch fresh data + orderbook + trades
  2. Build features (200+ including news sentiment)
  3. Detect market regime
  4. Run multi-strategy selector (picks best strategy for conditions)
  5. Analyze order flow for confirmation
  6. Check news for sentiment-driven trades
  7. Compute position size (Kelly + RL + regime + confidence)
  8. Validate risk-reward ratio
  9. Execute via paper or live trader
"""

import logging
import time

import numpy as np
import pandas as pd

from bot.exchange import ExchangeClient
from bot.ml.data_pipeline import DataPipeline
from bot.ml.features import FeatureEngine
from bot.ml.model import EnsembleModel
from bot.ml.strategy_selector import StrategySelector
from bot.ml.rl_agent import RLPositionSizer, KellyCriterionSizer
from bot.analysis.orderflow import OrderFlowAnalyzer
from bot.analysis.regime import RegimeDetector
from bot.analysis.news_sentiment import NewsSentimentAnalyzer
from bot.trading.position_sizer import DynamicPositionSizer

logger = logging.getLogger("trading_bot")

TF_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "12h": 43200, "1d": 86400,
}


class SignalEngine:
    """Core signal generation + execution loop."""

    def __init__(self, config: dict, trader, exchange: ExchangeClient):
        self.config = config
        self.trader = trader
        self.exchange = exchange

        self.data_pipeline = DataPipeline(exchange, config)
        self.feature_engine = FeatureEngine(config)
        self.model = EnsembleModel(config)
        self.strategy_selector = StrategySelector(config)
        self.rl_sizer = RLPositionSizer(config)
        self.orderflow = OrderFlowAnalyzer(config)
        self.regime_detector = RegimeDetector(config)
        self.news_analyzer = NewsSentimentAnalyzer(config)
        self.position_sizer = DynamicPositionSizer(config)
        self.kelly = KellyCriterionSizer(kelly_fraction=0.25)

        self.symbols = config["data"].get("symbols", [config["trading"]["primary_symbol"]])
        self.primary_symbol = config["trading"]["primary_symbol"]
        self.timeframe = config["data"]["timeframes"][0]
        self.use_multi_strategy = config.get("model", {}).get("multi_strategy", True)
        self.use_news = config.get("news", {}).get("enabled", True)
        self.running = False
        self.cycle_count = 0
        self.last_train_time = 0.0

    def initialize(self) -> bool:
        """Load trained models."""
        logger.info("Loading ML models...")
        self.model.load()
        if not self.model.is_trained:
            logger.warning("No trained model found — run 'python main.py train' first!")
            return False

        self.rl_sizer.load()
        self.last_train_time = time.time()
        logger.info("Signal engine initialized")
        return True

    def run_cycle(self):
        """Execute one complete trading cycle across all symbols."""
        self.cycle_count += 1
        logger.info(f"━━━ Cycle {self.cycle_count} ━━━")

        for symbol in self.symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}", exc_info=True)

    def _process_symbol(self, symbol: str):
        """Run full signal pipeline for one symbol."""
        # 1. Fetch data
        df = self.data_pipeline.fetch_historical(symbol, self.timeframe, days=7)
        if len(df) < 100:
            logger.warning(f"{symbol}: insufficient data ({len(df)} candles)")
            return

        current_price = float(df["close"].iloc[-1])
        logger.info(f"{symbol}: ${current_price:,.2f}")

        # 2. Check exits on existing positions
        closed = self.trader.check_exits(symbol, current_price)
        for trade in closed:
            self.kelly.update(trade.pnl)
            # Record outcome for strategy selector
            if hasattr(trade, "exit_reason"):
                logger.info(f"  Closed: {trade.exit_reason} PnL=${trade.pnl:+.2f}")

        # Check halt conditions
        if hasattr(self.trader, "is_halted") and self.trader.is_halted:
            logger.warning("Trading halted (drawdown limit)")
            return
        if hasattr(self.trader, "kill_switch") and self.trader.kill_switch:
            logger.error("Kill switch active")
            return

        # 3. Fetch orderbook + trade flow
        orderbook, trades_df = None, None
        try:
            orderbook = self.data_pipeline.fetch_orderbook(symbol)
            trades_df = self.data_pipeline.fetch_recent_trades(symbol)
        except Exception:
            pass

        # 4. Build features
        higher_tf = self.config["data"]["timeframes"][-1] if len(self.config["data"]["timeframes"]) > 1 else None
        higher_df = None
        if higher_tf and higher_tf != self.timeframe:
            try:
                higher_df = self.data_pipeline.fetch_historical(symbol, higher_tf, days=30)
            except Exception:
                pass

        feat_df = self.feature_engine.build_features(
            df, orderbook=orderbook, trades_df=trades_df, higher_tf_df=higher_df,
        )

        # Add news sentiment features
        if self.use_news:
            try:
                news_feats = self.news_analyzer.get_sentiment_features(symbol)
                for k, v in news_feats.items():
                    feat_df[k] = v
            except Exception:
                pass

        # 5. Detect regime
        regime_result = self.regime_detector.detect(df)
        regime = regime_result["regime"]
        logger.info(f"  Regime: {regime.value} (conf={regime_result['confidence']:.2f})")

        # 6. Get trade signal
        direction = None
        confidence = 0.0
        predicted_return = 0.0

        # Check news-driven trades first
        news_signal = None
        if self.use_news:
            try:
                news_signal = self.news_analyzer.should_trade_on_news(symbol)
                if news_signal:
                    logger.info(
                        f"  NEWS SIGNAL: {news_signal['direction']} "
                        f"(conf={news_signal['confidence']:.2f}) — {news_signal['reason']}"
                    )
            except Exception:
                pass

        # Multi-strategy selection
        if self.use_multi_strategy and self.strategy_selector.models:
            best = self.strategy_selector.select_best_strategy(
                feat_df, regime, regime_result["confidence"], news_signal
            )
            if best:
                direction = best.direction
                confidence = best.confidence
                predicted_return = best.predicted_return
                logger.info(
                    f"  Strategy: {best.strategy.value} | {direction} "
                    f"conf={confidence:.3f} edge={best.edge_score:.3f}"
                )
        else:
            # Fallback: use primary ensemble model
            try:
                prediction = self.model.predict(feat_df)
                direction = "long" if prediction["direction"] == 1 else "short"
                confidence = prediction["confidence"]
                predicted_return = prediction["predicted_return"]
                logger.info(
                    f"  ML Signal: {direction} conf={confidence:.3f} "
                    f"pred_return={predicted_return:+.4f}"
                )
            except Exception as e:
                logger.error(f"  Prediction failed: {e}")
                return

        if direction is None or direction == "none":
            return

        # 7. Order flow confirmation
        if orderbook:
            of_book = self.orderflow.analyze_orderbook(orderbook)
            imbalance = of_book.get("imbalance", 0)
            # Order flow should confirm direction
            if direction == "long" and imbalance < -0.3:
                logger.info(f"  Order flow contradicts BUY (imbalance={imbalance:+.2f})")
                confidence *= 0.7  # Reduce confidence
            elif direction == "short" and imbalance > 0.3:
                logger.info(f"  Order flow contradicts SELL (imbalance={imbalance:+.2f})")
                confidence *= 0.7

        if trades_df is not None and not trades_df.empty:
            of_trades = self.orderflow.analyze_trades(trades_df)
            logger.info(
                f"  Flow: CVD={of_trades['cvd_normalized']:+.3f} "
                f"whales={of_trades['whale_trades']} bias={of_trades['whale_bias']:+.2f}"
            )

        # 8. Confidence check
        min_conf = self.config.get("model", {}).get("min_confidence", 0.65)
        if confidence < min_conf:
            logger.info(f"  Confidence {confidence:.3f} < {min_conf} — no trade")
            return

        # 9. Position sizing
        portfolio_value = self._get_portfolio_value()

        rl_suggestion = None
        if self.rl_sizer.is_trained:
            obs = np.array([
                confidence,
                predicted_return,
                portfolio_value / 10000,
                0,
                0,
                float(feat_df["realized_vol_20"].iloc[-1]) if "realized_vol_20" in feat_df.columns else 0,
                self.trader.total_drawdown_pct / 100 if hasattr(self.trader, "total_drawdown_pct") else 0,
                0.5,
                1 if regime.value.startswith("trending") else 0,
                0,
            ], dtype=np.float32)
            rl_suggestion = self.rl_sizer.get_position_size(obs)

        sizing = self.position_sizer.compute(
            portfolio_value=portfolio_value,
            model_confidence=confidence,
            predicted_return=predicted_return,
            regime=regime,
            regime_confidence=regime_result["confidence"],
            kelly_fraction=self.kelly.get_kelly_size(),
            rl_suggestion=rl_suggestion,
            current_drawdown_pct=getattr(self.trader, "total_drawdown_pct", 0),
        )

        if sizing["position_pct"] == 0:
            return

        # 10. Risk-reward check
        risk_pct = self.config.get("risk", {}).get("trailing_stop_pct", 1.5)
        min_rr = self.config.get("risk", {}).get("min_risk_reward_ratio", 2.0)
        reward_pct = abs(predicted_return) * 100
        if risk_pct > 0 and reward_pct > 0 and reward_pct / risk_pct < min_rr:
            logger.info(f"  R/R {reward_pct/risk_pct:.2f} < {min_rr} — skipping")
            return

        # 11. Execute
        existing = self.trader.get_open_positions(symbol)

        # Close opposite positions
        for pos in existing:
            if pos.side != direction and confidence > 0.7:
                self.trader.close_position(pos, current_price, "signal_reversal")

        # Open new position
        if not any(p.side == direction for p in existing):
            amount = (portfolio_value * sizing["position_pct"] / 100) / current_price
            if amount > 0:
                self.trader.open_position(
                    symbol=symbol,
                    side=direction,
                    amount=amount,
                    price=current_price,
                    leverage=sizing["leverage"],
                    stop_loss_pct=risk_pct,
                    take_profit_pct=risk_pct * min_rr,
                    trailing_stop_pct=self.config.get("risk", {}).get("trailing_stop_pct", 1.5),
                )

    def run(self):
        """Continuous trading loop."""
        if not self.model.is_trained:
            logger.error("Model not trained! Run 'python main.py train' first.")
            return

        interval = TF_SECONDS.get(self.timeframe, 60)
        self.running = True

        logger.info(
            f"Signal engine started | {len(self.symbols)} symbols "
            f"| {self.timeframe} | interval={interval}s"
        )

        while self.running:
            try:
                self.run_cycle()
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)

            if self.running:
                logger.info(f"Next cycle in {interval}s...")
                time.sleep(interval)

    def stop(self):
        self.running = False

    def _get_portfolio_value(self) -> float:
        if hasattr(self.trader, "balance"):
            return self.trader.balance
        try:
            balance = self.exchange.fetch_balance()
            quote = self.primary_symbol.split("/")[1]
            return float(balance.get("total", {}).get(quote, 0))
        except Exception:
            return self.config.get("paper", {}).get("initial_balance", 10000)
