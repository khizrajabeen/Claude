# Crypto Auto Trading Bot

A Python-based automated cryptocurrency trading bot that supports 100+ exchanges via [ccxt](https://github.com/ccxt/ccxt), multiple trading strategies, and built-in risk management.

## Features

- **Multi-exchange support** — Binance, Coinbase, Kraken, Bybit, and 100+ more
- **3 built-in strategies** — SMA Crossover, RSI, Bollinger Bands
- **Risk management** — Stop-loss, take-profit, daily loss limits, position sizing
- **Sandbox mode** — Test safely on exchange testnets before going live
- **Configurable** — YAML config with CLI overrides
- **Extensible** — Add custom strategies by subclassing `BaseStrategy`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure

Copy the example env file and add your exchange API keys:

```bash
cp .env.example .env
# Edit .env with your API key and secret
```

Or edit `config.yaml` directly (not recommended — use `.env` for secrets).

### 3. Run in sandbox mode (recommended first)

```bash
# Dry run — analyze the market without placing orders
python main.py --dry-run

# Live sandbox trading
python main.py --sandbox
```

### 4. Run for real

```bash
# Edit config.yaml and set sandbox: false, then:
python main.py
```

## CLI Options

```
python main.py [OPTIONS]

  --config FILE        Config file path (default: config.yaml)
  --strategy NAME      Override strategy: sma_crossover, rsi, bollinger_bands
  --symbol PAIR        Override trading pair (e.g. ETH/USDT)
  --timeframe TF       Override timeframe (e.g. 5m, 1h, 4h)
  --dry-run            Run one analysis cycle without placing orders
  --sandbox            Force sandbox/testnet mode
```

## Strategies

| Strategy | Description | Best For |
|---|---|---|
| `sma_crossover` | Buys when fast SMA crosses above slow SMA | Trending markets |
| `rsi` | Buys when RSI is oversold, sells when overbought | Range-bound markets |
| `bollinger_bands` | Buys at lower band, sells at upper band | Mean-reversion |

### Adding a Custom Strategy

Create a new file in `bot/strategies/` and subclass `BaseStrategy`:

```python
from bot.strategies.base import BaseStrategy, Signal

class MyStrategy(BaseStrategy):
    name = "My Strategy"

    def analyze(self, df):
        # Your logic here
        return Signal.BUY  # or Signal.SELL or Signal.HOLD

    def get_indicator_values(self, df):
        return {"my_indicator": 42}
```

Then register it in `bot/strategies/__init__.py`.

## Configuration

All settings are in `config.yaml`. Key sections:

- **exchange** — Exchange name, API keys, sandbox mode
- **trading** — Symbol, timeframe, strategy, order size
- **risk_management** — Stop-loss, take-profit, daily loss limit, max position size
- **strategies** — Per-strategy parameters (SMA periods, RSI thresholds, etc.)

## Risk Management

The bot includes these safety mechanisms:

- **Stop-loss** — Auto-closes positions exceeding the configured loss %
- **Take-profit** — Auto-closes positions when target gain % is reached
- **Daily loss limit** — Halts trading if cumulative daily losses exceed threshold
- **Position size limit** — Rejects trades exceeding max % of portfolio
- **Max open trades** — Limits the number of concurrent positions

## Project Structure

```
├── main.py                  # Entry point / CLI
├── config.yaml              # Bot configuration
├── requirements.txt         # Python dependencies
├── .env.example             # API key template
└── bot/
    ├── __init__.py
    ├── config_loader.py     # YAML + env config loading
    ├── exchange.py          # ccxt exchange wrapper
    ├── logger.py            # Logging setup
    ├── risk_manager.py      # Risk management engine
    ├── trader.py            # Core trading loop
    └── strategies/
        ├── __init__.py      # Strategy registry
        ├── base.py          # BaseStrategy ABC
        ├── sma_crossover.py # SMA Crossover strategy
        ├── rsi_strategy.py  # RSI strategy
        └── bollinger_bands.py # Bollinger Bands strategy
```

## Disclaimer

This bot is for educational purposes. Cryptocurrency trading involves substantial risk of loss. Use at your own risk. Always test thoroughly in sandbox mode before using real funds.
