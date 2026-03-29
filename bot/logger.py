"""Logging setup for the trading bot."""

import logging
import sys


def setup_logger(config: dict) -> logging.Logger:
    """Configure and return the bot logger."""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO").upper(), logging.INFO)

    logger = logging.getLogger("trading_bot")
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    if log_config.get("log_to_file"):
        file_handler = logging.FileHandler(
            log_config.get("log_file", "trading_bot.log")
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
