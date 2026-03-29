#!/usr/bin/env python3
"""AI Crypto Trading Bot — Entry Point.

Modes:
    python main.py train                    # Train ML models on historical data
    python main.py paper                    # Paper trade with trained models
    python main.py live                     # Live trade (real money!)
    python main.py backtest                 # Walk-forward backtest
    python main.py analyze                  # Analyze current market (one-shot)

Options:
    --config FILE       Config file (default: config.yaml)
    --symbol PAIR       Override trading pair (e.g. ETH/USDT)
    --sandbox           Force sandbox/testnet mode
"""

import argparse
import signal as sig_module
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Crypto Trading Bot — ML Ensemble + RL Position Sizing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                     Train models (run this first!)
  python main.py train --symbol ETH/USDT   Train on ETH
  python main.py paper                     Paper trade with AI signals
  python main.py live --sandbox            Live trade on testnet
  python main.py backtest                  Walk-forward backtest
  python main.py analyze                   One-shot market analysis
        """,
    )

    parser.add_argument(
        "mode",
        choices=["train", "paper", "live", "backtest", "analyze"],
        help="Operating mode",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--symbol", help="Override trading pair")
    parser.add_argument("--sandbox", action="store_true", help="Force sandbox mode")

    # Training options
    parser.add_argument("--days", type=int, help="Override lookback days for training")

    # Backtest options
    parser.add_argument("--train-window", type=int, default=5000, help="Backtest train window (candles)")
    parser.add_argument("--test-window", type=int, default=500, help="Backtest test window (candles)")

    return parser.parse_args()


def print_banner(logger, config, mode):
    logger.info("=" * 60)
    logger.info("  AI CRYPTO TRADING BOT v2.0")
    logger.info("  ML Ensemble + RL Position Sizing")
    logger.info("=" * 60)
    logger.info(f"  Mode      : {mode.upper()}")
    logger.info(f"  Exchange  : {config['exchange']['name']}")
    logger.info(f"  Symbol    : {config['trading']['primary_symbol']}")
    logger.info(f"  Sandbox   : {config['exchange'].get('sandbox', True)}")
    logger.info(f"  Device    : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Mem   : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    logger.info(f"  Timeframes: {config['data']['timeframes']}")
    logger.info("=" * 60)


def mode_train(config, logger, args):
    """Train ML ensemble + RL agent on historical data."""
    from bot.ml.trainer import TrainingPipeline

    symbol = args.symbol or config["trading"]["primary_symbol"]
    if args.days:
        config["data"]["lookback_days"] = args.days

    pipeline = TrainingPipeline(config)
    pipeline.run_full_training(symbol)

    logger.info("Training complete! You can now run: python main.py paper")


def mode_paper(config, logger, args):
    """Paper trade using trained ML models."""
    from bot.exchange import ExchangeClient
    from bot.trading.paper_trader import PaperTrader
    from bot.trading.signal_engine import SignalEngine

    exchange = ExchangeClient(config)
    paper_trader = PaperTrader(config)
    engine = SignalEngine(config, paper_trader, exchange)

    if not engine.initialize():
        logger.error("Failed to load models. Run 'python main.py train' first!")
        sys.exit(1)

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down paper trader...")
        engine.stop()
        paper_trader.print_stats()
        paper_trader.save_journal()

    sig_module.signal(sig_module.SIGINT, shutdown)
    sig_module.signal(sig_module.SIGTERM, shutdown)

    logger.info(f"Paper trading started | Balance: ${paper_trader.balance:,.2f}")
    engine.run()

    # Print final stats
    paper_trader.print_stats()
    paper_trader.save_journal()


def mode_live(config, logger, args):
    """Live trading with real money. Requires explicit confirmation."""
    from bot.exchange import ExchangeClient
    from bot.trading.live_trader import LiveTrader
    from bot.trading.signal_engine import SignalEngine

    if not config["exchange"].get("api_key"):
        logger.error("API key not configured! Set EXCHANGE_API_KEY in .env")
        sys.exit(1)

    if not config["exchange"].get("sandbox", True) and not args.sandbox:
        logger.warning("=" * 60)
        logger.warning("  WARNING: LIVE TRADING WITH REAL MONEY!")
        logger.warning("  Press Ctrl+C within 10 seconds to cancel...")
        logger.warning("=" * 60)
        try:
            time.sleep(10)
        except KeyboardInterrupt:
            logger.info("Cancelled.")
            sys.exit(0)

    if args.sandbox:
        config["exchange"]["sandbox"] = True

    exchange = ExchangeClient(config)
    live_trader = LiveTrader(config, exchange)
    engine = SignalEngine(config, live_trader, exchange)

    if not engine.initialize():
        logger.error("Failed to load models. Run 'python main.py train' first!")
        sys.exit(1)

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down live trader...")
        engine.stop()

    sig_module.signal(sig_module.SIGINT, shutdown)
    sig_module.signal(sig_module.SIGTERM, shutdown)

    logger.info(f"LIVE trading started | Balance: ${live_trader.balance:,.2f}")
    engine.run()


def mode_backtest(config, logger, args):
    """Run walk-forward backtest."""
    from bot.exchange import ExchangeClient
    from bot.ml.data_pipeline import DataPipeline
    from bot.utils.backtester import WalkForwardBacktester

    symbol = args.symbol or config["trading"]["primary_symbol"]
    primary_tf = config["data"]["timeframes"][0]

    exchange = ExchangeClient(config)
    pipeline = DataPipeline(exchange, config)

    logger.info(f"Fetching data for backtest: {symbol} {primary_tf}...")
    df = pipeline.fetch_historical(symbol, primary_tf, days=config["data"].get("lookback_days", 90))

    backtester = WalkForwardBacktester(config)
    results = backtester.run(
        df,
        train_window=args.train_window,
        test_window=args.test_window,
    )

    logger.info("Backtest results saved")
    return results


def mode_analyze(config, logger, args):
    """One-shot market analysis — show current signals without trading."""
    from bot.exchange import ExchangeClient
    from bot.ml.data_pipeline import DataPipeline
    from bot.ml.features import FeatureEngine
    from bot.ml.model import EnsembleModel
    from bot.analysis.orderflow import OrderFlowAnalyzer
    from bot.analysis.regime import RegimeDetector

    symbol = args.symbol or config["trading"]["primary_symbol"]
    primary_tf = config["data"]["timeframes"][0]

    exchange = ExchangeClient(config)
    pipeline = DataPipeline(exchange, config)
    feature_engine = FeatureEngine(config)
    regime_detector = RegimeDetector(config)
    orderflow = OrderFlowAnalyzer(config)

    # Fetch data
    df = pipeline.fetch_historical(symbol, primary_tf, days=7)
    current_price = float(df["close"].iloc[-1])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  MARKET ANALYSIS: {symbol}")
    logger.info(f"  Price: ${current_price:,.2f}")
    logger.info(f"{'=' * 60}")

    # Regime
    regime_result = regime_detector.detect(df)
    logger.info(f"\n  Market Regime: {regime_result['regime'].value}")
    logger.info(f"  Regime Confidence: {regime_result['confidence']:.2f}")
    for k, v in regime_result["details"].items():
        logger.info(f"    {k}: {v}")

    # Order flow
    try:
        orderbook = pipeline.fetch_orderbook(symbol)
        trades_df = pipeline.fetch_recent_trades(symbol)

        of_book = orderflow.analyze_orderbook(orderbook)
        logger.info(f"\n  Orderbook Analysis:")
        for k, v in of_book.items():
            logger.info(f"    {k}: {v}")

        of_trades = orderflow.analyze_trades(trades_df)
        logger.info(f"\n  Order Flow Analysis:")
        for k, v in of_trades.items():
            logger.info(f"    {k}: {v}")
    except Exception as e:
        logger.info(f"\n  Order flow unavailable: {e}")

    # ML prediction (if model exists)
    model = EnsembleModel(config)
    model.load()
    if model.is_trained:
        feat_df = feature_engine.build_features(df)
        prediction = model.predict(feat_df)
        logger.info(f"\n  ML Prediction:")
        logger.info(f"    Direction: {'BUY' if prediction['direction'] == 1 else 'SELL'}")
        logger.info(f"    Confidence: {prediction['confidence']:.3f}")
        logger.info(f"    Buy Probability: {prediction['buy_probability']:.3f}")
        logger.info(f"    Predicted Return: {prediction['predicted_return']:+.4f}")
        logger.info(f"    Sub-models:")
        for name, vals in prediction["sub_predictions"].items():
            logger.info(f"      {name}: {vals}")
    else:
        logger.info("\n  No trained model — run 'python main.py train' first")

    logger.info(f"\n{'=' * 60}")


def main():
    args = parse_args()

    from bot.config_loader import load_config
    from bot.logger import setup_logger

    config = load_config(args.config)

    if args.symbol:
        config["trading"]["primary_symbol"] = args.symbol
    if args.sandbox:
        config["exchange"]["sandbox"] = True

    logger = setup_logger(config)
    print_banner(logger, config, args.mode)

    mode_map = {
        "train": mode_train,
        "paper": mode_paper,
        "live": mode_live,
        "backtest": mode_backtest,
        "analyze": mode_analyze,
    }

    mode_map[args.mode](config, logger, args)


if __name__ == "__main__":
    main()
