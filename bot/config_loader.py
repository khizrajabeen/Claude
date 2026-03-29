"""Load and validate bot configuration from YAML and environment variables."""

import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


def load_config(config_path: str = "config.yaml") -> dict:
    """Load configuration from YAML file, with .env overrides for secrets."""
    load_dotenv()

    # Prefer config.local.yaml if it exists
    local_path = Path(config_path).with_suffix(".local.yaml")
    if local_path.exists():
        config_path = str(local_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    # Override API credentials from environment variables
    env_key = os.getenv("EXCHANGE_API_KEY")
    env_secret = os.getenv("EXCHANGE_API_SECRET")
    if env_key:
        config["exchange"]["api_key"] = env_key
    if env_secret:
        config["exchange"]["api_secret"] = env_secret

    _validate_config(config)
    return config


def _validate_config(config: dict) -> None:
    """Validate required configuration fields."""
    required_sections = ["exchange", "trading", "risk_management"]
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: '{section}'")

    if not config["exchange"].get("name"):
        raise ValueError("Exchange name is required")
    if not config["trading"].get("symbol"):
        raise ValueError("Trading symbol is required")
    if config["risk_management"]["stop_loss_pct"] <= 0:
        raise ValueError("Stop-loss must be positive")
    if config["risk_management"]["take_profit_pct"] <= 0:
        raise ValueError("Take-profit must be positive")
