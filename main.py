#!/usr/bin/env python3
"""AI Crypto Trading Bot v2.1 — Entry Point.

Modes:
    python main.py train                    # Train on all coins + all strategies
    python main.py train --single-coin      # Train on primary coin only
    python main.py download                 # Download/refresh market data
    python main.py paper                    # Paper trade with trained models
    python main.py live                     # Live trade (real money!)
    python main.py backtest                 # Walk-forward backtest
    python main.py analyze                  # Analyze current market (one-shot)
    python main.py news                     # Check current news sentiment

Options:
    --config FILE       Config file (default: config.yaml)
    --symbol PAIR       Override trading pair (e.g. ETH/USDT)
    --sandbox           Force sandbox/testnet mode
    --auto-discover     Auto-detect top coins by volume
"""

import argparse
import signal as sig_module
import sys
import time

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Crypto Trading Bot — Multi-Coin, Multi-Strategy, News-Aware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py train                     Train all coins + all strategies
  python main.py train --auto-discover     Auto-find top coins and train
  python main.py train --single-coin       Train primary coin only
  python main.py download                  Download data for all coins
  python main.py download --auto-discover  Download top coins by volume
  python main.py paper                     Paper trade with AI signals
  python main.py live --sandbox            Live trade on testnet
  python main.py backtest                  Walk-forward backtest
  python main.py analyze                   One-shot market analysis
  python main.py analyze --symbol SOL/USDT Analyze specific coin
  python main.py news                      Check crypto news sentiment
  python main.py news --symbol ETH/USDT    News for specific coin
        """,
    )

    parser.add_argument(
        "mode",
        choices=["train", "download", "paper", "live", "backtest", "analyze", "news"],
        help="Operating mode",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--symbol", help="Override trading pair")
    parser.add_argument("--sandbox", action="store_true", help="Force sandbox mode")

    # Training options
    parser.add_argument("--days", type=int, help="Override lookback days")
    parser.add_argument("--single-coin", action="store_true", help="Train on primary coin only")
    parser.add_argument("--no-multi-strategy", action="store_true", help="Skip multi-strategy training")
    parser.add_argument("--auto-discover", action="store_true", help="Auto-discover top coins by volume")

    # Backtest options
    parser.add_argument("--train-window", type=int, default=5000, help="Backtest train window")
    parser.add_argument("--test-window", type=int, default=500, help="Backtest test window")

    return parser.parse_args()


def print_banner(logger, config, mode):
    logger.info("=" * 60)
    logger.info("  AI CRYPTO TRADING BOT v2.1")
    logger.info("  Multi-Coin | Multi-Strategy | News-Aware")
    logger.info("=" * 60)
    logger.info(f"  Mode       : {mode.upper()}")
    logger.info(f"  Exchange   : {config['exchange']['name']}")
    logger.info(f"  Coins      : {config['data']['symbols']}")
    logger.info(f"  Primary    : {config['trading']['primary_symbol']}")
    logger.info(f"  Sandbox    : {config['exchange'].get('sandbox', True)}")
    logger.info(f"  Device     : {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        logger.info(f"  GPU Mem    : {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    logger.info(f"  Timeframes : {config['data']['timeframes']}")
    logger.info(f"  Strategies : {config.get('strategies', {}).get('enabled', ['all'])}")
    logger.info(f"  News       : {'ON' if config.get('news', {}).get('enabled') else 'OFF'}")
    logger.info("=" * 60)


def mode_download(config, logger, args):
    """Download/refresh market data for all configured coins."""
    from bot.exchange import ExchangeClient
    from bot.ml.data_pipeline import DataPipeline
    from bot.ml.auto_data import AutoDataManager

    if args.days:
        config["data"]["lookback_days"] = args.days

    exchange = ExchangeClient(config)
    pipeline = DataPipeline(exchange, config)
    auto_data = AutoDataManager(config, pipeline)

    symbols = config["data"].get("symbols", [])
    if args.auto_discover:
        discovered = auto_data.discover_top_coins(
            top_n=config["data"].get("auto_discover_count", 15)
        )
        symbols = list(set(symbols + discovered))
        logger.info(f"Total symbols to download: {len(symbols)}")

    all_data = auto_data.download_all(
        symbols=symbols,
        days=config["data"].get("lookback_days", 90),
    )

    total = sum(len(tfs) for tfs in all_data.values())
    logger.info(f"Download complete: {total} datasets for {len(all_data)} coins")


def mode_train(config, logger, args):
    """Train ML models — multi-coin, multi-strategy."""
    from bot.ml.trainer import TrainingPipeline

    symbol = args.symbol or config["trading"]["primary_symbol"]
    if args.days:
        config["data"]["lookback_days"] = args.days

    pipeline = TrainingPipeline(config)
    pipeline.run_full_training(
        symbol=symbol,
        multi_coin=not args.single_coin,
        multi_strategy=not args.no_multi_strategy,
        auto_discover=args.auto_discover,
    )

    logger.info("Training complete! Next steps:")
    logger.info("  python main.py paper      # Paper trade")
    logger.info("  python main.py backtest   # Backtest")
    logger.info("  python main.py analyze    # Market analysis")


def mode_paper(config, logger, args):
    """Paper trade using trained ML models."""
    from bot.exchange import ExchangeClient
    from bot.trading.paper_trader import PaperTrader
    from bot.trading.signal_engine import SignalEngine

    exchange = ExchangeClient(config)
    paper_trader = PaperTrader(config)
    engine = SignalEngine(config, paper_trader, exchange)

    if not engine.initialize():
        logger.error("No trained models. Run 'python main.py train' first!")
        sys.exit(1)

    def shutdown(signum, frame):
        logger.info("Shutting down paper trader...")
        engine.stop()
        paper_trader.print_stats()
        paper_trader.save_journal()

    sig_module.signal(sig_module.SIGINT, shutdown)
    sig_module.signal(sig_module.SIGTERM, shutdown)

    logger.info(f"Paper trading started | Balance: ${paper_trader.balance:,.2f}")
    logger.info(f"Trading {len(engine.symbols)} symbols: {engine.symbols}")
    engine.run()

    paper_trader.print_stats()
    paper_trader.save_journal()


def mode_live(config, logger, args):
    """Live trading with real money."""
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
        logger.error("No trained models. Run 'python main.py train' first!")
        sys.exit(1)

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

    logger.info(f"Fetching data: {symbol} {primary_tf}...")
    df = pipeline.fetch_historical(
        symbol, primary_tf, days=config["data"].get("lookback_days", 90)
    )

    backtester = WalkForwardBacktester(config)
    results = backtester.run(
        df,
        train_window=args.train_window,
        test_window=args.test_window,
    )

    logger.info("Backtest complete")
    return results


def mode_analyze(config, logger, args):
    """One-shot market analysis."""
    from bot.exchange import ExchangeClient
    from bot.ml.data_pipeline import DataPipeline
    from bot.ml.features import FeatureEngine
    from bot.ml.model import EnsembleModel
    from bot.analysis.orderflow import OrderFlowAnalyzer
    from bot.analysis.regime import RegimeDetector
    from bot.analysis.news_sentiment import NewsSentimentAnalyzer

    symbol = args.symbol or config["trading"]["primary_symbol"]
    primary_tf = config["data"]["timeframes"][0]

    exchange = ExchangeClient(config)
    pipeline = DataPipeline(exchange, config)
    feature_engine = FeatureEngine(config)
    regime_detector = RegimeDetector(config)
    orderflow = OrderFlowAnalyzer(config)
    news = NewsSentimentAnalyzer(config)

    df = pipeline.fetch_historical(symbol, primary_tf, days=7)
    current_price = float(df["close"].iloc[-1])

    logger.info(f"\n{'=' * 60}")
    logger.info(f"  MARKET ANALYSIS: {symbol}")
    logger.info(f"  Price: ${current_price:,.2f}")
    logger.info(f"{'=' * 60}")

    # Regime
    regime_result = regime_detector.detect(df)
    logger.info(f"\n  Market Regime: {regime_result['regime'].value}")
    logger.info(f"  Confidence: {regime_result['confidence']:.2f}")
    for k, v in regime_result["details"].items():
        logger.info(f"    {k}: {v}")

    # Order flow
    try:
        orderbook = pipeline.fetch_orderbook(symbol)
        trades_df = pipeline.fetch_recent_trades(symbol)

        of_book = orderflow.analyze_orderbook(orderbook)
        logger.info(f"\n  Orderbook:")
        for k, v in of_book.items():
            logger.info(f"    {k}: {v}")

        of_trades = orderflow.analyze_trades(trades_df)
        logger.info(f"\n  Order Flow:")
        for k, v in of_trades.items():
            logger.info(f"    {k}: {v}")
    except Exception as e:
        logger.info(f"\n  Order flow unavailable: {e}")

    # News sentiment
    try:
        sent = news.get_sentiment_for_coin(symbol)
        logger.info(f"\n  News Sentiment:")
        logger.info(f"    Score: {sent['score']:+.3f}")
        logger.info(f"    Articles: {sent['article_count']} ({sent['bullish_count']}B / {sent['bearish_count']}R)")
        logger.info(f"    Signal Strength: {sent['signal_strength']:.3f}")
        for h in sent["top_headlines"][:3]:
            logger.info(f"    • [{h['score']:+.2f}] {h['title'][:80]}")

        news_signal = news.should_trade_on_news(symbol)
        if news_signal:
            logger.info(f"    NEWS SIGNAL: {news_signal['direction']} (conf={news_signal['confidence']:.2f})")
    except Exception as e:
        logger.info(f"\n  News unavailable: {e}")

    # ML prediction
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
        for name, vals in prediction["sub_predictions"].items():
            logger.info(f"    {name}: {vals}")
    else:
        logger.info("\n  No trained model — run 'python main.py train' first")

    logger.info(f"\n{'=' * 60}")


def mode_news(config, logger, args):
    """Check current news sentiment for all coins."""
    from bot.analysis.news_sentiment import NewsSentimentAnalyzer

    news = NewsSentimentAnalyzer(config)

    symbols = [args.symbol] if args.symbol else config["data"].get("symbols", ["BTC/USDT"])

    logger.info(f"\n{'=' * 60}")
    logger.info("  CRYPTO NEWS SENTIMENT")
    logger.info(f"{'=' * 60}")

    for symbol in symbols:
        coin = symbol.split("/")[0]
        sent = news.get_sentiment_for_coin(symbol, hours=24)

        # Sentiment bar visualization
        score = sent["score"]
        bar_len = 20
        filled = int((score + 1) / 2 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)

        logger.info(f"\n  {coin}:")
        logger.info(f"    Sentiment: [{bar}] {score:+.3f}")
        logger.info(f"    Articles:  {sent['article_count']} ({sent['bullish_count']} bullish / {sent['bearish_count']} bearish)")
        logger.info(f"    Signal:    {sent['signal_strength']:.3f}")

        for h in sent["top_headlines"][:3]:
            emoji = "+" if h["score"] > 0.2 else "-" if h["score"] < -0.2 else "~"
            logger.info(f"      [{emoji}{abs(h['score']):.1f}] {h['title'][:70]}")

        trade_signal = news.should_trade_on_news(symbol)
        if trade_signal:
            logger.info(f"    >> TRADE SIGNAL: {trade_signal['direction'].upper()} (conf={trade_signal['confidence']:.2f})")
            logger.info(f"       Reason: {trade_signal['reason']}")

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
        "download": mode_download,
        "paper": mode_paper,
        "live": mode_live,
        "backtest": mode_backtest,
        "analyze": mode_analyze,
        "news": mode_news,
    }

    mode_map[args.mode](config, logger, args)


if __name__ == "__main__":
    main()
