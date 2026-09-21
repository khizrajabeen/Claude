"""Load and validate bot configuration from YAML plus environment overrides."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

REQUIRED_SECTIONS = ("exchange", "data", "session", "risk", "stops", "sizing")


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration, preferring config.local.yaml when present."""
    load_dotenv()

    local_path = Path(config_path).with_suffix(".local.yaml")
    if local_path.exists():
        config_path = str(local_path)

    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        config = yaml.safe_load(f) or {}

    # Secrets never live in the YAML.
    for env_var, (section, key) in {
        "EXCHANGE_API_KEY": ("exchange", "api_key"),
        "EXCHANGE_API_SECRET": ("exchange", "api_secret"),
        "EXCHANGE_API_PASSWORD": ("exchange", "api_password"),
    }.items():
        value = os.getenv(env_var)
        if value:
            config.setdefault(section, {})[key] = value

    validate_config(config)
    return config


def validate_config(config: dict) -> None:
    """Fail loudly at startup rather than mid-session."""
    for section in REQUIRED_SECTIONS:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    if not config["exchange"].get("name"):
        raise ValueError("exchange.name is required")

    symbols = config["data"].get("symbols")
    if not symbols:
        raise ValueError("data.symbols must list at least one pair")
    for symbol in symbols:
        if "/" not in symbol:
            raise ValueError(f"data.symbols entry '{symbol}' must look like BASE/QUOTE")

    if not config["data"].get("timeframe"):
        raise ValueError("data.timeframe is required")

    risk = config["risk"]
    for key in ("risk_per_trade_pct", "max_portfolio_heat_pct",
                "max_daily_loss_pct", "max_total_drawdown_pct"):
        value = risk.get(key)
        if value is None or float(value) <= 0:
            raise ValueError(f"risk.{key} must be a positive number")

    heat = float(risk["max_portfolio_heat_pct"])
    per_trade = float(risk["risk_per_trade_pct"])
    if per_trade > heat:
        raise ValueError(
            f"risk.risk_per_trade_pct ({per_trade}) exceeds "
            f"risk.max_portfolio_heat_pct ({heat}) — no trade could ever open"
        )
    if float(risk["max_daily_loss_pct"]) >= float(risk["max_total_drawdown_pct"]):
        raise ValueError(
            "risk.max_daily_loss_pct must be below risk.max_total_drawdown_pct, "
            "or the daily breaker never fires before the hard halt"
        )

    stops = config["stops"]
    if float(stops.get("atr_stop_mult", 0)) <= 0:
        raise ValueError("stops.atr_stop_mult must be positive")
    if float(stops.get("target_r_multiple", 0)) <= 0:
        raise ValueError("stops.target_r_multiple must be positive")

    session = config["session"]
    for key in ("day_open", "flatten_at"):
        value = str(session.get(key, ""))
        if ":" not in value:
            raise ValueError(f"session.{key} must be 'HH:MM'")

    mode = config["exchange"].get("market_type", "spot")
    if mode not in ("spot", "swap", "future"):
        raise ValueError("exchange.market_type must be spot, swap or future")

    _validate_history_depth(config)


def _validate_history_depth(config: dict) -> None:
    """Fetch enough bars for every enabled strategy's lookback.

    A strategy given less history than its longest lookback returns no
    signals at all, which looks identical to "no opportunity today". That
    is the worst kind of bug, so it is caught at startup instead.
    """
    from bot.strategies import build_strategies

    history = int(config.get("data", {}).get("history_bars", 500))
    for strategy in build_strategies(config):
        needed = strategy.required_bars()
        if needed > history:
            raise ValueError(
                f"data.history_bars ({history}) is below what strategy "
                f"'{strategy.name}' needs ({needed} bars). Raise history_bars, "
                f"shorten that strategy's lookback, or disable it — as configured "
                f"it would silently never produce a signal."
            )
