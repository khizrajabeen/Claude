# Meridian — a daily-cycle trading bot

A crypto trading bot built around a **trading day**. Each day it screens
the venue, reads the news and the tape, writes a plan, works that plan
inside bounded entry slots, manages the book against volatility-scaled
stops, closes out, and writes the day to disk — so tomorrow starts from a
real record rather than from zero.

There is a dashboard. The bot publishes its state as JSON and a static
page reads it, so the record is legible without a terminal.

Everything runs in paper mode against live public market data. No API keys
are needed to run it.

> **Where this actually stands.** The current roster returns **+16.2% over
> 90 days** in replay, but read that with the caution it deserves: it is a
> simulation over historical bars with modelled fees, slippage and
> funding, chosen by looking at those same bars. Earlier configurations of
> this same bot returned −0.66% over the same period. What is trustworthy
> here is the *measurement* — purged cross-validation, standard errors on
> every result, out-of-sample splitting, and replays that run the same
> code the live path does. Read the [results](#measured-results) before
> running anything with money.

```bash
pip install -r requirements.txt

python main.py screen        # rank every market on the venue by turnover
python main.py briefing      # what the market and the news look like now
python main.py plan          # ... and what the bot would trade
python main.py day           # run one full trading day (paper)
python main.py run           # run day after day
python main.py replay --days 90   # replay the same logic over history
python main.py pnl --months 3     # per-day and per-asset-class P&L
python main.py publish            # write the dashboard's data files
python main.py bench --days 150   # compare strategy configurations
```

Nothing needs configuring. The bot picks its own decision timeframe per
instrument per day, weights its strategies by measured risk, scales total
exposure to its own realised volatility, and benches drivers that stop
earning.

---

## The roster, and how it was chosen

Each strategy was run **alone** over the same 90 days of crypto history,
on one shared download, so the only thing differing between runs was the
strategy:

| strategy | return | trades | expectancy | max DD | win% | PF | |
|---|---:|---:|---:|---:|---:|---:|---|
| `clenow` | **+14.91%** | 39 | +0.581R | 1.33% | 69.2 | 5.58 | kept |
| `turtle` | **+10.77%** | 69 | +0.255R | 1.29% | 60.9 | 2.09 | kept |
| `lorentzian` | +3.80% | 50 | −0.002R | 4.02% | 48.0 | 1.37 | kept, on sufferance |
| `supertrend` | −1.17% | 214 | −0.070R | 5.58% | 42.1 | 1.01 | dropped |
| `nwenvelope` | −3.23% | 19 | −0.308R | 3.25% | 31.6 | 0.20 | dropped |
| `holygrail` | −3.26% | 55 | −0.206R | 5.14% | 30.9 | 0.76 | dropped |
| `smc` | −6.45% | 162 | −0.321R | 9.40% | 30.9 | 0.65 | dropped |

The two published trend systems carry the book, at roughly a fifth of the
drawdown the full seven-strategy roster suffered. The four dropped ones
traded about 450 times between them to lose money — `smc` and
`supertrend` alone accounted for 376 of those trades. That is the same
lesson every measurement in this project has returned: **trading less is
the only durable edge found so far.**

`lorentzian` is kept on sufferance. Its return is positive but its
expectancy is essentially zero, so the gain came from a handful of large
winners rather than an edge per trade. It earns its place as a different
kind of driver beside two correlated trend systems, and the autopilot
benches it automatically if that stops being true.

Nothing is deleted. The dropped strategies keep their code, their tests
and their bench variants, because "it lost money over one 90-day window
on one universe" is a finding, not a proof.

---

## The dashboard

```bash
python main.py publish --screen     # writes web/data/*.json
python -m http.server 8000 -d web   # then open http://localhost:8000
```

Three pages, no framework and no build step: an overview, a portfolio
dashboard (equity curve, open positions, per-asset-class attribution,
allocation, recent trades, market screen) and a settings page for API
keys.

Two things it deliberately will not do. It will not invent numbers — a
missing data file produces an explanation of how to generate it, not a
plausible-looking figure. And it will not pretend to be fresh: data older
than two hours turns the status indicator amber, because a dashboard
whose data quietly went stale looks exactly like one that is working.

Keys entered on the settings page stay in that browser and go nowhere
else, which the page says plainly. For a bot that trades unattended they
are the wrong place; the server reads its own `.env`, and the page has an
**Export .env** button that writes it.

`deploy/` has a systemd timer, a Dockerfile and a compose file for
running the cycle around the clock and publishing to GitHub Pages. See
[deploy/README.md](deploy/README.md) — including the part where Pages
serves files and cannot run the bot.

---

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

Seven, in two families. Six earlier ones were removed after measuring
poorly across repeated benches — a house trend model, cross-sectional
momentum, a Donchian breakout, short-term reversion, funding carry and
dual momentum. That removal is recorded rather than quietly done, because
"we tried it and it did not work" is information.

### selective — survived the out-of-sample split

Published systems, implemented to their stated rules so a comparison is
against the real thing rather than a paraphrase.

| Strategy | Source | The rules, as published |
|---|---|---|
| `turtle` | Dennis & Eckhardt, 1983 | 20-bar and 55-bar Donchian entries; N = ATR(20); 2N stop; a unit every 0.5N to four units; 10-bar opposite-channel exit. Includes the filter most implementations drop — skip an S1 breakout if the previous one would have won — with the 55-bar failsafe always taken |
| `clenow` | Clenow, *Following the Trend* | EMA(50) above EMA(100) plus price at a 100-bar extreme; 3 ATR stop |
| `holygrail` | Raschke & Connors, *Street Smarts* | ADX(14) above 30, then wait for the retracement to the 20 EMA. The only pullback entry here — the others all buy strength, so they fire at the same moment and are effectively one bet |

### lux — indicator-style, measured not trusted

| Strategy | What it does |
|---|---|
| `supertrend` | SuperTrend across a range of multipliers, each scored on what it would actually have earned, then k-means into three groups with the best group's centroid traded. Seeded at the quartiles so the result is deterministic |
| `smc` | Smart Money Concepts: BOS/CHoCH structure breaks, liquidity sweeps, fair value gaps, premium/discount positioning — each component reported separately |
| `nwenvelope` | Nadaraya-Watson envelope, endpoint estimator only. The default form repaints |
| `lorentzian` | kNN over historical market states. Instead of applying a rule it asks what happened the last time the market looked like this, and lets the closest neighbours vote |

**Lorentzian Classification** is the most-used open-source ML indicator on
TradingView and the only genuinely different thing in the book. Features
are RSI(14), WaveTrend, CCI(20), ADX(20) and RSI(9); distance is
`sum(log(1 + |dx|))`. The logarithm is the point — under Euclidean
distance a single volatility spike decides who counts as a neighbour.

Two details this implementation gets right that most do not. Features are
normalised on a **trailing** window, not the whole series, because
rescaling against a future minimum is a leak. And neighbours are drawn
only from bars whose outcome had already resolved, so no bar votes on a
future it could not have seen. Both have tests.

### Confirmation vs contrarian

From LuxAlgo's Oscillator Matrix, which separates signals that ride a move
from ones that fade it. The distinction is not cosmetic: a fade enters
earlier and is right more often about the turn, but it is betting against
whatever is currently working — so when it is wrong, it is wrong into a
move that is still running.

Every signal carries a stance. `smc` picks per-signal, since a structure
break confirms while a sweep reversal opposes. The risk layer cuts size on
fades in proportion to how much of the combined view is one.

## Autopilot

Nothing here needs configuring. The bot manages its own roster from what
the strategies have actually earned:

- **Bench the losers.** A strategy with a materially negative record over a
  real sample stops being funded.
- **Keep a probe.** Benched is not deleted — a small allocation stays alive,
  because a strategy switched fully off produces no record and can never
  earn its way back.
- **Tilt for the tape.** Trend systems and reversal systems fail in each
  other's weather, so the roster leans toward whichever suits the current
  regime.

The obvious failure mode is chasing whatever worked last month. Three
guards: nothing is judged before 25 trades, the bench threshold sits at
−0.15R rather than at zero, and no more than half the roster can be benched
at once — when most strategies are losing, the settings or the market are
the problem, not the selection.

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

### Measured results

`python main.py bench --days 150` replays cumulative configurations over
identical bars, so each step's contribution is attributable. Six majors on
OKX, hourly:

```
  variant         return%   maxDD%  trades  expect R     ± SE       t  real?
  single            -1.57     8.36     301    -0.035    0.068   -0.51     no
  multi-equal       -0.14     7.18     363    -0.025    0.063   -0.40     no
  multi-parity       0.60     7.27     357    -0.016    0.063   -0.26     no
  full               0.71     7.71     357    -0.017    0.063   -0.26     no
  turtle             2.62     1.08      22     0.221    0.282    0.78     no
  clenow             5.08     0.50      13     0.683    0.368    1.85     no
  holygrail          0.99     0.74      17     0.428    0.323    1.33     no
  dualmom           -5.17     6.41     156    -0.106    0.093   -1.14     no
  published         -6.31     6.92     181    -0.098    0.087   -1.13     no
  everything        -3.00     6.56     353    -0.037    0.064   -0.58     no
  autopilot         -3.62     6.51     354    -0.054    0.062   -0.86     no
```

**Read the `t` column before anything else.** Not one variant clears
|t| ≥ 2. Every expectancy in that table is compatible with having no edge
at all, including the ones that look spectacular.

Three things it does show:

**The layers work in the direction the research predicts.** Going from one
strategy to five cut drawdown from 8.36% to 7.18% and lifted return by 1.4
points; risk parity added another 0.7. That is the diversification and
risk-budgeting claim, reproduced. The differences are not significant, but
they point the right way and they point the right way for the stated
reason.

**Clenow's +5.08% at 0.50% drawdown is not a result.** It took 13 trades.
At the observed variance it needs roughly 16 before the edge means
anything, and turtle — which looks nearly as good — needs about 145. A
Sharpe computed on thirteen trades measures the sample size, not the
strategy. This is precisely why the bench prints standard errors.

**The real finding is about costs, not about gurus.** The systems that made
money took 13–22 trades in 150 days. The blend took 357 for the same gross
edge, paying the round trip seventeen times more often. Two changes follow
from that, both from first principles rather than fitted to this sample: a
cost gate that refuses any trade whose expected move is under 3× the
modelled round trip, and overnight carry as the default, since flattening
nightly pays the spread again every morning for a book whose published
holding periods are measured in days.

Combining everything made things *worse*, not better — `published` at
−6.31% is dragged down by `dualmom`'s 156 losing trades. Diversification
helps when the drivers are individually sound; it launders nothing.

### The changes that followed did not help

Both follow-up changes — the cost gate and overnight carry — are defensible
from first principles. Re-running the identical bench with them on:

```
  variant         return%   maxDD%  trades  expect R     ± SE       t   (was)
  single            -3.65     9.40     236    -0.103    0.072   -1.42   -1.57
  multi-equal       -4.88     8.08     324    -0.069    0.065   -1.06   -0.14
  multi-parity      -4.50     8.03     251    -0.110    0.072   -1.52   +0.60
  full              -5.29     8.74     250    -0.118    0.072   -1.64   +0.71
  autopilot         -1.71     8.08     212    -0.064    0.081   -0.79   -3.62
```

The blend got **worse**, not better. The cost gate did what it was designed
to do — `full` went from 357 trades to 250 — and the result still
deteriorated. Only `autopilot` improved, from −3.62% to −1.71%, which is
consistent with the conviction fix finally letting benching bite.

These changes are kept anyway, and that is a judgement call worth stating
outright. The reasoning behind them is sound and the counter-evidence is
not significant: every |t| in both tables is below 2, so reverting on this
sample would be fitting to noise — the exact error the purged validation
exists to prevent. Both are switchable (`risk.min_edge_cost_ratio`,
`session.carry_overnight`) precisely because the measurement does not
settle it.

What the two tables together actually establish is narrower than anyone
would like: **this system has no demonstrated edge.** The infrastructure
that lets you know that — purged folds, standard errors, identical-data
replays — is the part that is trustworthy. The strategies are not yet.

### Out of sample: does picking the winners survive?

`clenow`, `turtle` and `holygrail` led the first bench, so the `selective`
variant bundles them. But they were chosen *from* that sample — re-running
it would only confirm the choice made from it. `bench --oos` cuts 300 days
of history at one instant across every symbol and reports both halves:

```
  variant                   in-sample            OUT-OF-SAMPLE
                    return%    trades     return%  trades      t
  full               -10.49       319       -1.06     244  -0.81
  selective           -4.04        33       +3.87      42   0.95
  luxalgo            -10.47       220       -0.10     145  -0.12
  selective+lux      -10.36       175       +2.94     142   0.10
  autopilot          -10.61       170       +2.19     229  -0.02
```

**No variant is positive in both halves.** The first period was hostile to
everything; the second was kinder to most.

Two things are still worth reading out of it:

**`selective` ranked first in both halves** — least bad in the hard period,
best in the easy one. Consistent *relative* ranking across a split is
weaker evidence than a significant return, but it is not nothing, and it is
the only result here that survived the split at all. It also traded 33 and
42 times against `full`'s 319 and 244, which is the same "trade less"
signal the first bench produced, now visible on data that did not choose it.

**It is still not significant.** t = 0.95 on 42 trades, and it lost 4% in
the first half. This is a reason to keep watching `selective`, not a reason
to fund it.

### Did LuxAlgo help? Two tests, two different answers.

The LuxAlgo-style strategies are implemented in full and measured on the
same footing as everything else. Two out-of-sample splits, on different
sample lengths:

**300 days, split in half:**

```
  variant                   in-sample            OUT-OF-SAMPLE
                    return%    trades     return%  trades      t
  selective           -4.04        33       +3.87      42   0.95
  luxalgo            -10.47       220       -0.10     145  -0.12
  selective+lux      -10.36       175       +2.94     142   0.10
```

**400 days, split in half, after the prune:**

```
  variant                   in-sample            OUT-OF-SAMPLE
                    return%    trades     return%  trades      t
  selective           -3.64        53       +1.50      47   0.38
  lux                -10.47       204       -0.27     161   0.10
  everything          -9.19       221       +4.33     167   0.81
  autopilot           -8.49       145       +1.14     179   0.05
```

On the first test, adding the LuxAlgo strategies to the selective three
made things **worse** (+3.87% → +2.94%). On the second, the same
combination made things **better** (+1.50% → +4.33%).

**That contradiction is the finding.** With t-statistics of 0.38 and 0.81,
neither test can distinguish these variants from each other or from zero,
and so two honest runs on two samples reversed the ranking. Anyone
reporting either number alone would be reporting noise as a result. It is
the single clearest demonstration in this repository of why the
significance column exists.

What holds across both: `lux` on its own is the weakest family — worst
in-sample in both tests and roughly flat out of sample. That is consistent
with the published evidence on Smart Money Concepts, where 648 backtests
across four markets found no ICT/SMC signal with a significant forward
edge and nothing beating buy-and-hold. It is kept enabled because the user
asked to keep measuring it, and because the code is correct even where the
signal is not.

Also consistent across both: **no variant is positive in both halves**,
and the selective three trade a fraction as often as everything else
(47–53 trades against 161–221) for comparable or better results.

### What the prune actually bought

Not performance — the tables above cannot support that claim. What it
bought is measurable in other ways:

- **Seven strategies instead of thirteen**, with the six removals recorded
  and their reasons stated.
- **A 7x faster replay.** Profiling found 51% of replay time in one place:
  every price lookup ran a boolean mask over the whole index and allocated
  a fresh frame, thousands of times per simulated day. Twenty days went
  from 8.4s to 1.2s, which is the difference between a bench being a
  coffee break and an afternoon — and therefore between running one
  experiment and running ten.
- **Two look-ahead traps closed**, both found by implementing the LuxAlgo
  indicators properly: swing points that are only knowable once the right
  shoulder prints, and a Nadaraya-Watson fit that repaints unless the
  endpoint estimator is used. Either would have made a naive version look
  excellent.
- **A warm-up correctness fix.** The replay warmed up on `filters.min_bars`
  (120) rather than on what the enabled strategies need (724 with
  `lorentzian` in the roster), so it was replaying a stretch where the
  deepest strategies were starved. A starved strategy returns nothing,
  which is indistinguishable from having no opportunity — so the measured
  result silently belonged to whichever half of the roster was awake.

## Machine learning: measured, then left switched off

```bash
pip install -r requirements-ml.txt
python main.py compare --days 200     # the bake-off
python main.py train --days 200       # fit the winner, if there is one
```

Nine models across five families — a majority-class baseline, two linear
models, two tree ensembles, three boosting variants and a shallow net —
all given the same features, the same triple-barrier labels, the same
sample weights and the same purged folds, with scaling fit inside each
fold so nothing gets to peek.

The literature disagrees about what should win. Gu, Kelly and Xiu find
trees and neural networks beat linear models on US equities via nonlinear
interactions; more recent work on tabular financial features finds
gradient boosting matching deep learning at a fraction of the cost. The
zoo exists to test that rather than assume it.

**28,800 rows, 109 features, 5 purged folds, six majors on hourly bars:**

```
  model          family         acc  vs base     AUC   logloss
  hist_gbm       boosting    0.4844  -0.0265  0.5004    0.7735
  xgboost        boosting    0.4838  -0.0271  0.5000    0.7766
  majority       baseline    0.4764  -0.0345  0.5000    0.6951
  lightgbm       boosting    0.4842  -0.0267  0.4980    0.7732
  logistic       linear      0.4707  -0.0402  0.4938    0.7355
  random_forest  trees       0.4808  -0.0301  0.4928    0.7070
  mlp            neural      0.4803  -0.0306  0.4833    0.9929
```

**Nothing beats the baseline.** Every AUC sits between 0.483 and 0.500 —
a coin flip — and every model's accuracy is *below* the majority-class
rate. The fold-to-fold spread (±0.02 to ±0.04) swallows every difference
between them.

The literature's ordering does faintly appear — boosting ahead of linear,
linear ahead of the net — but it is inside the noise and means nothing
here. The one substantive result is the log-loss column: the MLP is not
merely no better, it is *worse calibrated* than the baseline (0.99 against
0.70). Since position sizing consumes probabilities, a confidently wrong
model is more dangerous than an uncertain one.

This is the expected outcome for generic technical features on
short-horizon crypto bars, and it is exactly what the purged validation is
for. An unpurged split would have reported an edge that is not there —
which is how most published crypto-ML results are produced. `model.enabled`
stays `false`, and the trainer says so in plain words rather than shipping
a model that cannot beat guessing.

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
python -m pytest tests/ -q      # 526 tests, no network
```

The suite pins the claims this README makes: that margin cannot be
double-spent, that dollar risk is constant across volatility, that a
stop-out costs about −1R, that funding only accrues across settlements,
that purged folds do not leak while naive splits do, that news windows
genuinely differ, and that a second process resumes exactly where the first
one stopped.

It also pins the awkward ones: that a benched strategy cannot shout louder
than a trusted one, that a great-looking result on fourteen trades is
reported as *not* significant, that spot crypto cannot be sold short,
that a trade which sold half at 2R and then stopped at breakeven is
recorded as the winner it was, that leverage never puts the stop outside
the liquidation price, that no credential reaches the published
dashboard, and that the bake-off finds a planted signal but invents none
from noise.

Tests requiring optional extras skip rather than fail — `requirements-ml.txt`
installs what the ML bake-off needs.

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
  strategies/  clenow, turtle, lorentzian (enabled)
               holygrail, supertrend, smc, nwenvelope, news (measured, off)
  markets/     instruments, trading calendars, timeframe ladder, screener
  data/        one router over crypto, Alpaca and daily-equity providers
  portfolio/   risk-parity allocator, vol targeting, autopilot, tracker
  daily/       schedule + entry slots, briefing, plan, session, journal
  risk/        position sizing, portfolio gates, circuit breakers
  trading/     paper broker (scale in/out), shared position/trade records
  analysis/    indicators, regime detection, news sentiment, beta book
  ml/          labeling, purged validation, model zoo, bake-off, trainer
  utils/       replay, variant bench, metrics, P&L attribution, publisher
web/           landing page, dashboard, settings — plain files, no build
deploy/        systemd timer, Dockerfile, compose, runbook
tests/         526 tests
```

---

## Reading this repository honestly

A few things worth knowing before the numbers persuade you of anything:

**The roster was chosen by looking at the same data it is measured on.**
The +16.2% figure is in-sample in that sense. The out-of-sample splitting
machinery exists (`--oos`) precisely because that distinction matters.

**Single replays cannot resolve small differences.** Four variants of one
change, over identical data, landed between −6.03% and +0.90% — because
refusing one trade frees cash and heat, which changes which *other*
trades get taken, which moves the equity curve that sizes everything
after. Run over three windows the same comparison gave −1.55% ± 2.19 and
−1.83% ± 2.59: a difference smaller than its own spread.

**Several things here are documented negative results**, kept because
they were paid for: the ML bake-off (nine models, every AUC 0.483–0.500,
below baseline), the cost-in-R gate (swept and switched off — the numbers
were noise), and four strategies that lost money and kept their code so
the finding stays repeatable.

**It trades on paper by default.** Backtests flatter: they fill at prices
nobody queued for and they know which markets survived. Nothing here is
financial advice.
