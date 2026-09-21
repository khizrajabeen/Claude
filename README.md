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

Tomorrow's briefing reads yesterday's closing equity, the positions it
inherited, and the rolling win rate and expectancy from `trades.csv` —
which then feed the day's risk budget. A bad stretch sizes the next day
down; a good one lets it back to the configured base, never above it.

---

## How a trade gets taken

**1. The edge score.** A regime-weighted blend of four views, each
normalised to roughly [-1, 1]:

| View | Inputs |
|---|---|
| Trend | EMA structure gated by ADX/DI, confirmed on the higher timeframe |
| Momentum | MACD histogram in ATR units, overnight move |
| Mean reversion | z-score, RSI stretch, position in the Donchian range |
| News | sentiment tilt, scaled by consensus |
| Order book | top-of-book imbalance (small weight — confirmation only) |

Weights shift with the regime. A z-score of −2 is a buy in a range and a
falling knife in a downtrend, so the same reading gets different treatment
depending on which regime the detector reports.

**2. Filters and vetoes.** Thin books, wide spreads and volatility outside a
sane band are dropped before scoring matters. A setup that fights the
higher timeframe is vetoed, as is one that strong news opposes.

**3. Sizing.** Risk is fixed in dollars and the stop is placed at a multiple
of ATR:

```
stop distance = atr_stop_mult × ATR
quantity      = (equity × risk_per_trade_pct) / stop distance
```

When volatility expands the stop widens and the position shrinks, so dollar
risk stays constant across assets and regimes. That is what stops losses
from clustering in exactly the periods that hurt most. Size is then capped
by a notional limit, a per-position volatility budget, and whatever the
exchange's lot rules allow.

**4. Portfolio gates.** Each candidate is checked against a running book, so
limits see the cumulative effect of the plan:

- **portfolio heat** — total open risk if every stop hit at once (4%)
- **correlation cap** — crypto majors trade as one factor, so same-side
  crowding is capped (3)
- **net beta** — beta-weighted exposure against equity (1.5×)
- **daily loss breaker** — measured on equity, unrealised included (2%)
- **drawdown halt** — hard stop against peak equity (15%)
- **loss cooldown** — no new entries for a while after consecutive losses

**5. Management.** Stops only ever ratchet forward: to just past entry once
1R is banked (with a pad so fees cannot turn a winner into a loser), then
trailing 2.5 ATR behind the extreme. There is a time stop for trades that
go nowhere and a maximum holding period.

---

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

### A measured result

45 days of hourly bars, six majors on OKX, default settings:

```
Equity        : $10,000.00 → $10,052.57 (+0.53%)
Trades        : 46 (18W / 28L, 39.1%)
Expectancy    : +0.036R per trade
Avg win/loss  : +1.42R / 0.86R
Profit factor : 1.11
Max drawdown  : 4.79%
Costs         : fees $86.72 | funding $7.16
Exits         : stop_loss 22, take_profit 10, end_of_day 14
```

Read that honestly: **roughly break-even after costs.** An expectancy of
+0.036R over 46 trades is well inside noise — it is not evidence of an
edge. What the numbers do show is that the mechanics are sound: stop-outs
cost about −1R, targets pay about +1.9R, and drawdown stayed inside the
budget. The risk plumbing works. Finding an actual edge to run through it
is separate work, and 45 days is nowhere near enough sample to claim one.

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
python -m pytest tests/ -q      # 109 tests, no network
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
tests/         109 tests
```
