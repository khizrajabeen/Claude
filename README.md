# AI Crypto Trading Bot v2.1

Production-grade AI trading bot with multi-coin training, multi-strategy auto-selection, news sentiment trading, and dynamic portfolio scaling.

**No RSI. No Bollinger Bands. No lagging indicators. Pure ML + order flow + news.**

## What Makes This Different

| Amateur Bots | This Bot |
|---|---|
| RSI/BB/MACD signals | XGBoost + LightGBM + LSTM + Transformer ensemble |
| One coin | Trains on 15+ coins simultaneously |
| One strategy | 5 strategies, auto-picks best for conditions |
| No news awareness | Real-time news sentiment analysis |
| Fixed position sizes | RL agent (PPO) + Kelly Criterion |
| Fixed leverage | Portfolio-tier scaling ($100 → 2x, $50K → 10x) |
| No order flow | Whale detection, CVD, iceberg detection |
| Manual data download | Auto-discovers top coins, auto-downloads data |

## Quick Start

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure API keys
cp .env.example .env    # Add your exchange API key + secret

# 3. Download data (auto-discovers top coins)
python main.py download --auto-discover

# 4. Train (uses GPU if available)
python main.py train

# 5. Paper trade (zero risk)
python main.py paper

# 6. Check news sentiment
python main.py news

# 7. Go live (testnet first!)
python main.py live --sandbox
```

## All Commands

```
python main.py <mode> [options]

Modes:
  train          Train ML models on all coins + all strategies
  download       Download/refresh market data
  paper          Paper trade with AI signals
  live           Live trade (real money)
  backtest       Walk-forward backtest
  analyze        One-shot market analysis
  news           Check crypto news sentiment

Options:
  --config FILE        Config file (default: config.yaml)
  --symbol PAIR        Override trading pair (e.g. ETH/USDT)
  --sandbox            Force testnet mode
  --days N             Override lookback days
  --auto-discover      Auto-detect top coins by volume
  --single-coin        Train on primary coin only
  --no-multi-strategy  Skip multi-strategy training
  --train-window N     Backtest training window (candles)
  --test-window N      Backtest testing window (candles)
```

## 5 Auto-Selected Strategies

The bot trains 5 separate ML models and auto-picks the best one based on current market conditions:

| Strategy | Best For | Key Features |
|---|---|---|
| **Momentum** | Trending markets (ADX>30) | ROC, EMA crossovers, MACD, higher-TF trend |
| **Mean Reversion** | Ranging/sideways | Z-scores, Stochastic, Williams %R, orderbook imbalance |
| **Breakout** | Quiet→volatile transitions | Volatility ratios, ATR, S/R distance |
| **Scalp** | High volume, short TF | 1-min returns, orderbook spread, trade flow |
| **News** | Strong sentiment signals | Headline sentiment, article volume, consensus |

The `StrategySelector` scores each strategy by:
- **Regime fitness** (is this strategy suited to current market?)
- **Model confidence** (how sure is the ML prediction?)
- **Historical performance** (has this strategy been working?)
- **Predicted return** (how big is the expected move?)

## News Sentiment Trading

Aggregates crypto news from 6+ sources in real-time:
- CoinTelegraph, CoinDesk, Bitcoin Magazine, CryptoNews
- Reddit r/CryptoCurrency, r/Bitcoin

Features:
- Time-weighted sentiment scoring (recent news weighted higher)
- Coin-specific keyword detection (15+ coins)
- Whale/institutional language detection
- Auto-trade triggers on strong consensus

```bash
python main.py news                     # All coins
python main.py news --symbol ETH/USDT   # Specific coin
```

## Multi-Coin Training

Training on multiple coins simultaneously gives the model **more data and better generalization** than single-coin training.

```bash
# Train on configured coins (config.yaml → data.symbols)
python main.py train

# Auto-discover top 15 coins by volume and train on all
python main.py train --auto-discover

# Download data first, then train
python main.py download --auto-discover
python main.py train

# Single coin only
python main.py train --single-coin --symbol ETH/USDT
```

## Dynamic Portfolio Scaling

Automatically adjusts aggressiveness as your portfolio grows:

| Portfolio | Max Leverage | Max Position | Risk/Trade |
|---|---|---|---|
| $0 – $1K | 2x | 5% | 1% |
| $1K – $10K | 5x | 10% | 2% |
| $10K – $50K | 8x | 15% | 3% |
| $50K+ | 10x | 20% | 4% |

## Risk Management

- **Trailing stops** — regime-adjusted (wider in trends, tighter in ranges)
- **Daily loss limit** — halts at 5% daily drawdown
- **Total drawdown limit** — kill switch at 15% from peak
- **Cooldown** — 5 min pause after losing trades
- **Min R/R ratio** — only takes 2:1+ setups
- **Kelly Criterion** — fractional (25%) for mathematically optimal sizing
- **Liquidation protection** — monitors leveraged positions

## Architecture

```
main.py                              CLI (7 modes)
├── bot/ml/
│   ├── auto_data.py                 Auto coin discovery + data download
│   ├── data_pipeline.py             Multi-TF data collection
│   ├── features.py                  200+ feature engineering
│   ├── model.py                     XGBoost + LightGBM + LSTM + Transformer
│   ├── strategy_selector.py         5-strategy auto-selection
│   ├── rl_agent.py                  PPO position sizer + Kelly
│   └── trainer.py                   Multi-coin, multi-strategy trainer
├── bot/trading/
│   ├── signal_engine.py             Brain: ML + news + flow → trades
│   ├── position_sizer.py            Dynamic sizing with tier scaling
│   ├── paper_trader.py              Paper trading (slippage, fees)
│   └── live_trader.py               Live trading (exchange SL/TP, kill switch)
├── bot/analysis/
│   ├── orderflow.py                 Whale detection, CVD, icebergs
│   ├── regime.py                    Market regime classification
│   └── news_sentiment.py            Multi-source news aggregation
└── bot/utils/
    ├── backtester.py                Walk-forward backtesting
    └── metrics.py                   Sharpe, Sortino, Calmar, etc.
```

## HPC / GPU Usage

```bash
# Check GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"

# Train with all cores + GPU (365 days, all coins)
python main.py train --days 365 --auto-discover
```

XGBoost/LightGBM use all CPU cores. LSTM/Transformer train on GPU.

## Disclaimer

For educational and research purposes. Cryptocurrency trading involves substantial risk. Past performance does not guarantee future results. Always test in paper mode first. Never trade with money you can't afford to lose.
