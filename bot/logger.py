"""Logging setup for the trading bot."""

from __future__ import annotations

import logging
import sys


def setup_logger(config: dict) -> logging.Logger:
    """Configure and return the bot logger.

    Idempotent: calling it twice does not duplicate every log line, which
    matters because the session, the replay and the tests all set up
    logging in the same process.
    """
    log_config = config.get("logging", {})
    level = getattr(logging, str(log_config.get("level", "INFO")).upper(), logging.INFO)

    logger = logging.getLogger("trading_bot")
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        for handler in logger.handlers:
            handler.setLevel(level)
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_config.get("log_to_file"):
        file_handler = logging.FileHandler(log_config.get("log_file", "trading_bot.log"))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
