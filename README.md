# Daily Crypto Trading Bot

A crypto trading bot built around a **trading day**. Each day it reads the
news and the tape, writes a plan, works that plan inside a bounded entry
window, manages the book against volatility-scaled stops for the rest of
the session, closes out, and writes the day to disk — so tomorrow starts
from a real record rather than from zero.

Everything runs in paper mode against live public market data. No API keys
are needed to run it.

```bash
pip install -r requirements.txt

python main.py briefing      # what the market and the news look like right now
python main.py plan          # ... and what the bot would trade
python main.py replay --days 60   # replay the same logic over history
python main.py day           # run one full trading day (paper)
python main.py run           # run day after day
python main.py report        # the record so far
```

---

## The daily cycle

```
00:00 UTC  BRIEFING   read yesterday's record, refresh news, read every
                      symbol: ATR, trend, regime, liquidity, funding
   ↓
00:00-02:00 ENTRY     work the plan; re-check prices before each fill
   ↓
until 23:30 MANAGE    every 5 min: mark, accrue funding, trail stops,
                      honour exits, watch the circuit breakers
   ↓
23:30      FLATTEN    close the intraday book (or carry, if configured)
   ↓
23:30+     REPORT     write trades.csv, days.csv, state.json
   ↓
           next day starts from that record
```

The day boundary is UTC midnight by default. Crypto never closes, so a
"day" is a convention — this one is anchored to the boundary exchanges use
for daily candles and for perpetual funding, which settles at 00:00, 08:00
and 16:00 UTC. All of it is configurable under `session:`.

### What carries into tomorrow

`state/` holds everything the next session needs:

| File | What it holds |
|---|---|
| `state.json` | cash, open positions, streaks, cooldowns, peak equity |
| `trades.csv` | every closed round trip, with R-multiple, MAE/MFE, costs |
| `days.csv` | one row per trading day: equity, return, W/L, fees, drawdown |
| `briefings/<date>.json` | that morning's read and the plan it produced |
| `session.lock` | held while a bot is trading, so two cannot share a directory |

One date gets one row: re-running a day after a crash corrects that day's
record rather than adding a second one, and a second bot pointed at the
same directory is refused rather than allowed to interleave its writes.

Tomorrow's briefing reads yesterday's closing equity, the positions it
inherited, and the rolling win rate and expectancy from `trades.csv` —
which then feed the day's risk budget. A bad stretch sizes the next day
down; a good one lets it back to the configured base, never above it.

---

## Strategies

Five independent return drivers, not one formula with five terms. The
distinction is the point: the evidence on managed futures is that combining
weakly correlated drivers is what shrinks drawdown, far more than improving
any single signal. A trend model and a carry model lose money at different
times; two trend models lose money together.

| Strategy | What it trades | Why it is here |
|---|---|---|
| `trend` | Volatility-scaled time-series momentum, blended over 1d / 3d / 2w | The CTA workhorse. Convex payoff — it is positioned for the persistent moves that make a crash a crash |
| `xsmom` | Cross-sectional momentum, 12-1 style | Roughly market-neutral by construction. It can **short a rising asset** that is rising less than its peers, which is exactly what makes it a diversifier rather than a trend clone |
| `breakout` | Donchian channel break, filtered on volatility compression | Flat during a grind that never makes a new high; already positioned when a quiet range snaps |
| `reversion` | Short-term reversal on statistically stretched moves | Makes money in the chop that whipsaws trend. Suppressed entirely when ADX says a real trend is running |
| `carry` | Perpetual funding, taking the paid side | Uncorrelated with price direction — but see the caveat below |

Adding a strategy is a file in `bot/strategies/` and a line in config. Each
one sees the same market context and knows nothing about the others; sizing
and capital allocation are not its business.

**On carry, honestly:** published work puts the crypto carry Sharpe above 6
over 2020–2023, falling through 2024 and turning negative in 2025 as
delta-neutral yield products crowded the trade. Recent samples show average
funding below the threshold at which it covers costs, so a disciplined rule
simply does not fire. This implementation stands aside rather than chasing
a yield that no longer clears fees — and on a spot venue there are no
funding rates at all, so it reports itself inactive and contributes
nothing.

## Portfolio construction

Strategies produce opinions; the portfolio layer decides how much each one
is worth and how hard to press overall.

**Risk parity across strategies.** Weights are inverse to each strategy's
own return volatility, so a noisy driver does not dominate the book's
variance just by being noisy. Equal *capital* is not equal *risk*, and it
is risk that produces drawdown.

**A correlation haircut.** Equal-risk weighting still over-allocates to a
cluster of strategies all saying the same thing. Each weight is reduced by
how correlated that strategy is with the rest, pushing capital toward the
genuinely independent drivers.

**Per-strategy caps.** Inverse volatility over-funds drivers whose risk
lives in the tail rather than in daily variance. Carry is the textbook
case — it looks almost riskless right up until it isn't — so it is capped
at 15% regardless of how calm its returns look.

Until there is enough realised history to measure either, everything is
weighted equally. Estimating a covariance matrix from a fortnight of data
produces confident nonsense.

**Volatility targeting** sits on top, scaling every position together. This
is the best-evidenced single lever on drawdown: left-tail events cluster in
high-volatility periods, so a vol-targeted book is already de-levered when
they arrive. Published effects take equity Sharpe from ~0.40 to ~0.50 with
materially smaller maximum drawdowns, and crypto shows the same pattern
with more positive skew. Two refinements: exposure is cut immediately on a
volatility spike but restored slowly, and a separate drawdown throttle
scales exposure down as the account falls from its peak.

## How a trade gets taken

1. **Strategies speak.** Each returns signed signals with a conviction in
   [0, 1], comparable across strategies.
2. **The allocator resolves them.** Weighted contributions are summed per
   symbol, so two strategies disagreeing cancel rather than opening two
   opposed positions in the same name. Agreement is reported alongside
   conviction.
3. **Tilts and vetoes.** News sentiment and order-book imbalance nudge the
   result; neither can open a trade, and a tilt that would flip the
   strategies' direction causes the bot to stand aside instead. A setup
   fighting the higher timeframe is vetoed — except for market-neutral
   strategies, since shorting a laggard in a rising market is the
   construction, not a mistake.
4. **Risk sizes what survives.** Fixed dollar risk at an ATR stop, scaled
   by the portfolio's volatility multiplier:

   ```
   stop distance = atr_stop_mult × ATR
   quantity      = (equity × risk_per_trade_pct × exposure_scale) / stop distance
   ```

   Then capped by notional limits, a per-position volatility budget, and
   the exchange's lot rules.
5. **Portfolio gates.** Each candidate is checked against a running book so
   limits see the cumulative effect of the plan: portfolio heat (total open
   risk if every stop hit at once), a correlation cap, net beta, the daily
   loss breaker, the drawdown halt and the loss cooldown.

Every trade records which strategy drove it, so returns are attributed back
and tomorrow's weights are computed from what actually happened.

## Execution costs

A backtest that ignores costs is a story. This one charges:

- maker/taker fees in basis points, separately
- slippage: a base term, plus half the spread, plus market impact
- **impact via the square-root law** — `coefficient × daily_vol × √(order/ADV)`,
  so a $1,500 order in a $50M book costs ~4bps and a $2M order in a $5M book
  costs ~100bps
- **perpetual funding**, accrued at each 00:00/08:00/16:00 UTC settlement a
  position is held through: longs pay a positive rate, shorts receive it

Margin is reserved on open, so two positions cannot spend the same dollar,
and equity is cash plus unrealised PnL, so the breakers see a losing open
position rather than only realised damage.

---

## Replay

`python main.py replay --days 60` runs the **same `DailySession`** over
historical bars, with a simulated clock and an exchange that only answers
with bars at or before that clock. There is no separate backtest engine to
drift out of sync with the live path.

Two limitations, stated in every report rather than hidden:

- **News is neutral in replay.** Public RSS feeds serve no history, so a
  replay cannot reconstruct what the wire said on a past morning. Sentiment
  is evaluated forward, in paper trading, not backwards.
- **Intrabar order is unknown.** When one bar contains both the stop and the
  target, the stop is assumed to fill first — the pessimistic assumption is
  the only defensible one without tick data, and the optimistic one is how
  backtests lie.

Replays write to `state/replay/` and never touch live records.

`--fast` on the live modes compresses the session clock so a whole day runs
in minutes. It is for exercising the lifecycle, not for measuring anything:
the clock moves but live market data does not move with it, so bar-driven
exits will not fire. Use `replay` to measure.

### A measured result

45 days of hourly bars, six majors on OKX, default settings:

```
Equity        : $10,000.00 → $10,069.30 (+0.69%)
Trades        : 47 (19W / 28L, 40.4%)
Expectancy    : +0.043R per trade
Avg win/loss  : +1.37R / 0.86R
Profit factor : 1.13
Sharpe (daily): 0.97
Max drawdown  : 4.79%
Costs         : fees $89.07 | funding $7.38
Exits         : stop_loss 22, take_profit 10, end_of_day 15
```

Read that honestly: **roughly break-even after costs.** An expectancy of
+0.043R over 47 trades is well inside noise — it is not evidence of an
edge, and 45 days is nowhere near the sample needed to claim one. What the
numbers do show is that the mechanics are sound: stop-outs cost about −1R,
targets pay about +1.9R, and drawdown stayed inside its budget. The risk
plumbing works. Finding an actual edge to run through it is separate work.

---

## Machine learning (optional, off by default)

The daily session needs none of this. When enabled it adds a fifth view to
the edge blend.

```bash
pip install -r requirements-ml.txt
python main.py train --days 180
```

- **Triple-barrier labels** (`bot/ml/labeling.py`). Each bar is labelled by
  which barrier price touches first — target, stop, or the clock — with
  barriers set from the same ATR the live risk layer uses. A label therefore
  means "a trade opened here would have won / lost / timed out", instead of
  "was the next bar up?", which is mostly a coin flip.
- **Sample weights by label uniqueness.** Two labels spanning the same bars
  are not two independent facts. Overlapping labels are down-weighted.
- **Purged K-fold with an embargo** (`bot/ml/validation.py`). Training rows
  whose label window overlaps the test fold are dropped, and a further band
  after each fold is embargoed for serial correlation. The test suite
  demonstrates both that purged folds do not leak and that a naive split
  does.
- **Everything fits inside the fold.** Feature selection and the scaler are
  fit on training rows only.

The trainer reports accuracy against the majority-class rate and says
plainly when there is no edge. A model that cannot beat that baseline is
reported as useless rather than dressed up.

Sequence models (LSTM, transformer) are deliberately absent. On a few
thousand crypto bars they have far more capacity than the data supports;
gradient-boosted trees on well-constructed features are the defensible
choice at this sample size.

---

## Configuration

`config.yaml` is commented throughout. Put secrets in `.env` or
`config.local.yaml` (both gitignored). Config is validated at startup and
fails loudly on combinations that cannot work — risk per trade above the
heat budget, a daily stop above the hard halt, a malformed pair.

The exchange defaults to Kraken. `binance`, `okx`, `kucoin`, `coinbase` and
`bybit` all work; symbols are mapped onto whatever quote currency the venue
actually lists. Note that Kraken's USDT books for smaller majors are thin
enough that the liquidity filter will skip them — OKX and KuCoin have
deeper books and longer history.

---

## Tests

```bash
python -m pytest tests/ -q      # 118 tests, no network
```

The suite pins the claims this README makes: that margin cannot be
double-spent, that dollar risk is constant across volatility, that a
stop-out costs about −1R, that funding only accrues across settlements,
that purged folds do not leak while naive splits do, that news windows
genuinely differ, and that a second process resumes exactly where the first
one stopped.

---

## Live trading

Not wired up. `main.py live` refuses and says so. The daily session drives
the paper broker only; connecting it to real order placement is
deliberately a separate, reviewed change — and `bot/exchange.py` already
has the order methods it would need, including resting stop orders so a
stop survives the bot going offline.

Before anyone considers it: the replay above is break-even, the sample is
small, and forward paper results are the only evidence worth acting on.

---

## Layout

```
bot/
  daily/       schedule, briefing, plan, session, journal
  risk/        position sizing, portfolio gates, circuit breakers
  trading/     paper broker, shared position/trade records
  analysis/    indicators, regime detection, news sentiment
  ml/          labeling, purged validation, model, trainer
  utils/       historical replay, performance metrics
tests/         118 tests
```
