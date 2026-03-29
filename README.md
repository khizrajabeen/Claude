# AI Crypto Trading Bot v2.0

Production-grade AI-powered cryptocurrency trading bot with ML ensemble predictions, reinforcement learning position sizing, order flow analysis, and dynamic portfolio scaling.

**No RSI. No Bollinger Bands. Pure ML.**

## What Makes This Different

| Amateur Bots | This Bot |
|---|---|
| RSI/BB/MACD signals | XGBoost + LightGBM + LSTM + Transformer ensemble |
| Fixed position sizes | RL agent (PPO) + Kelly Criterion dynamic sizing |
| One timeframe | Multi-timeframe feature fusion |
| No order flow | Whale detection, CVD, liquidity analysis |
| Fixed leverage | Portfolio-tier scaling (small → conservative, big → aggressive) |
| No regime awareness | Market regime detection (trending/ranging/volatile) |
| Hope-based stops | Trailing stops + regime-adjusted risk management |

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Signal Engine                     │
├──────────┬──────────┬───────────┬───────────────────┤
│  Data    │ Feature  │ ML        │ Market            │
│ Pipeline │ Engine   │ Ensemble  │ Analysis          │
│          │ (200+    │           │                   │
│ Multi-TF │ features)│ XGBoost   │ Order Flow        │
│ OHLCV    │          │ LightGBM  │ Regime Detect     │
│ Orderbook│ Returns  │ LSTM+Attn │ Whale Detection   │
│ Trades   │ Volatility│Transformer│ Liquidity Maps   │
│          │ Volume   │           │                   │
│          │ Momentum │ Confidence│ Regime-Adjusted   │
│          │ Statistic│ Score     │ Parameters        │
├──────────┴──────────┴───────────┴───────────────────┤
│              Dynamic Position Sizer                  │
│  Kelly Criterion × RL Agent × Regime × Confidence   │
│  Portfolio Tier Scaling (auto-scales with growth)    │
├─────────────────────┬───────────────────────────────┤
│   Paper Trader      │     Live Trader               │
│   (Simulation)      │     (Real Exchange)           │
│   Slippage + Fees   │     Smart Execution           │
│   Full Journal      │     Exchange SL/TP            │
│   Risk Limits       │     Kill Switch               │
└─────────────────────┴───────────────────────────────┘
```

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

Requires Python 3.11+ and optionally CUDA for GPU acceleration.

### 2. Configure

```bash
cp .env.example .env
# Add your exchange API key + secret
```

### 3. Train the models (run this first!)

```bash
# Train on BTC/USDT with 90 days of data
python main.py train

# Train on specific pair with more data
python main.py train --symbol ETH/USDT --days 180
```

On HPC with GPU this takes ~5-15 minutes. On CPU, ~30-60 minutes.

### 4. Paper trade (test without risk)

```bash
python main.py paper
```

This runs the full AI pipeline against live market data but simulates orders with realistic slippage and fees.

### 5. Backtest (historical performance)

```bash
python main.py backtest
python main.py backtest --train-window 8000 --test-window 1000
```

Uses walk-forward analysis (no look-ahead bias).

### 6. Analyze (one-shot market view)

```bash
python main.py analyze
python main.py analyze --symbol ETH/USDT
```

Shows current ML prediction, market regime, order flow, and whale activity.

### 7. Go live

```bash
# Testnet first (always!)
python main.py live --sandbox

# Real money (10-second safety countdown)
python main.py live
```

## ML Model Details

### Ensemble (4 models)

| Model | Weight | Strength |
|---|---|---|
| XGBoost | 35% | Tabular features, fast inference |
| LightGBM | 35% | Tabular features, handles missing values |
| Bi-LSTM + Attention | 20% | Sequential patterns, temporal dependencies |
| Transformer | 10% | Long-range dependencies, self-attention |

### Feature Engineering (200+ features)

- **Price Action**: Multi-horizon returns, log returns, gaps, ranges
- **Volatility**: Realized, Parkinson, Garman-Klass, ATR, vol ratios
- **Volume**: VWAP, OBV, accumulation/distribution, MFI, volume spikes
- **Momentum**: ROC, MA distances, EMA crossovers, MACD, Stochastic, CCI, Williams %R
- **Statistical**: Skewness, kurtosis, autocorrelation, Hurst exponent, Z-scores
- **Structure**: Candle patterns, body/wick ratios, consecutive direction
- **S/R Levels**: Distance to rolling highs/lows, range position
- **Multi-Timeframe**: Higher TF trend, momentum, structure
- **Order Flow**: Orderbook imbalance, wall detection, spoofing detection
- **Trade Flow**: CVD, whale detection, clustering (iceberg), aggression ratio

### RL Position Sizer (PPO)

The reinforcement learning agent observes:
- Model confidence and predicted return
- Portfolio value and current position
- Volatility and drawdown
- Win rate and consecutive losses
- Market regime

And outputs optimal position size [-1, 1] to maximize risk-adjusted returns.

## Dynamic Portfolio Scaling

The bot automatically adjusts aggressiveness as your portfolio grows:

| Portfolio Size | Max Leverage | Max Position | Risk/Trade |
|---|---|---|---|
| $0 – $1K | 2x | 5% | 1% |
| $1K – $10K | 5x | 10% | 2% |
| $10K – $50K | 8x | 15% | 3% |
| $50K+ | 10x | 20% | 4% |

## Risk Management

- **Trailing stops** — locks in profits, adjusts to regime
- **Daily loss limit** — halts trading at 5% daily drawdown
- **Total drawdown limit** — kill switch at 15% from peak
- **Cooldown** — 5 min pause after losing trades
- **Min risk-reward** — only takes 2:1+ R/R setups
- **Kelly Criterion** — mathematically optimal bet sizing (fractional, 25%)
- **Liquidation protection** — monitors leveraged positions

## Project Structure

```
├── main.py                         # CLI entry point (train/paper/live/backtest/analyze)
├── config.yaml                     # Full configuration
├── requirements.txt                # Dependencies
├── bot/
│   ├── config_loader.py            # YAML + .env config
│   ├── exchange.py                 # ccxt exchange wrapper
│   ├── logger.py                   # Logging setup
│   ├── ml/
│   │   ├── data_pipeline.py        # Multi-TF data collection + storage
│   │   ├── features.py             # 200+ feature engineering
│   │   ├── model.py                # XGBoost + LightGBM + LSTM + Transformer ensemble
│   │   ├── rl_agent.py             # PPO position sizer + Kelly Criterion
│   │   └── trainer.py              # End-to-end training pipeline
│   ├── trading/
│   │   ├── signal_engine.py        # Core decision engine (brain of the bot)
│   │   ├── position_sizer.py       # Dynamic sizing with portfolio scaling
│   │   ├── paper_trader.py         # Paper trading with realistic simulation
│   │   └── live_trader.py          # Live trading with safety rails
│   ├── analysis/
│   │   ├── orderflow.py            # Order flow + whale detection
│   │   └── regime.py               # Market regime classification
│   └── utils/
│       ├── backtester.py           # Walk-forward backtesting
│       └── metrics.py              # Sharpe, Sortino, Calmar, profit factor, etc.
├── data/                           # Cached market data + trade journals
└── models/                         # Trained model artifacts
```

## HPC / GPU Usage

The bot automatically detects and uses CUDA GPUs:

```bash
# Check GPU is detected
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Train with all cores + GPU
python main.py train --days 365
```

XGBoost and LightGBM use all CPU cores (`n_jobs=-1`). LSTM and Transformer train on GPU if available.

## Disclaimer

This bot is for educational and research purposes. Cryptocurrency trading involves substantial risk. Past performance does not guarantee future results. Always test thoroughly in paper mode before using real funds. Never trade with money you can't afford to lose.
