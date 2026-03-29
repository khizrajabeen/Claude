#!/usr/bin/env python3
"""Crypto Auto Trading Bot — Entry Point.

Usage:
    python main.py                          # Run with default config.yaml
    python main.py --config config.yaml     # Run with custom config
    python main.py --dry-run                # Analyze without placing orders
    python main.py --strategy rsi           # Override strategy
    python main.py --symbol ETH/USDT        # Override trading pair
"""

import argparse
import signal
import sys

from bot.config_loader import load_config
from bot.logger import setup_logger
from bot.trader import Trader


def parse_args():
    parser = argparse.ArgumentParser(description="Crypto Auto Trading Bot")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--strategy",
        choices=["sma_crossover", "rsi", "bollinger_bands"],
        help="Override trading strategy",
    )
    parser.add_argument("--symbol", help="Override trading pair (e.g. ETH/USDT)")
    parser.add_argument("--timeframe", help="Override candle timeframe (e.g. 5m, 1h)")
    parser.add_argument(
        "--dry-run", action="store_true", help="Run one cycle without placing real orders"
    )
    parser.add_argument(
        "--sandbox", action="store_true", default=None,
        help="Force sandbox/testnet mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)

    # CLI overrides
    if args.strategy:
        config["trading"]["strategy"] = args.strategy
    if args.symbol:
        config["trading"]["symbol"] = args.symbol
    if args.timeframe:
        config["trading"]["timeframe"] = args.timeframe
    if args.sandbox is not None:
        config["exchange"]["sandbox"] = args.sandbox

    logger = setup_logger(config)
    logger.info("=" * 50)
    logger.info("  Crypto Auto Trading Bot v1.0.0")
    logger.info("=" * 50)
    logger.info(f"  Exchange  : {config['exchange']['name']}")
    logger.info(f"  Symbol    : {config['trading']['symbol']}")
    logger.info(f"  Timeframe : {config['trading']['timeframe']}")
    logger.info(f"  Strategy  : {config['trading']['strategy']}")
    logger.info(f"  Sandbox   : {config['exchange'].get('sandbox', True)}")
    logger.info(f"  Stop-Loss : {config['risk_management']['stop_loss_pct']}%")
    logger.info(f"  Take-Profit: {config['risk_management']['take_profit_pct']}%")
    logger.info("=" * 50)

    trader = Trader(config)

    # Graceful shutdown on Ctrl+C
    def shutdown(signum, frame):
        logger.info("Received shutdown signal")
        trader.stop()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if args.dry_run:
        logger.info("DRY RUN — executing one analysis cycle (no orders)")
        try:
            df = trader.exchange.fetch_ohlcv(
                config["trading"]["symbol"], config["trading"]["timeframe"], limit=100
            )
            sig = trader.strategy.analyze(df)
            indicators = trader.strategy.get_indicator_values(df)
            current_price = float(df["close"].iloc[-1])
            logger.info(f"Price: {current_price}")
            logger.info(f"Signal: {sig.value}")
            logger.info(f"Indicators: {indicators}")
        except Exception as e:
            logger.error(f"Dry run failed: {e}", exc_info=True)
            sys.exit(1)
    else:
        trader.run()


if __name__ == "__main__":
    main()
