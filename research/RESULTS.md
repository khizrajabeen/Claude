# Daily strategy experiment — 2026-09-25

Candidate: `momentum_90_200`. Long BTC/USD and ETH/USD when the completed daily close exceeds both its close 90 days ago and its 200-day simple moving average; otherwise cash. Exit the entire position when the condition turns false. No leverage, pyramids, scale-outs, trailing stops, fixed profit targets or intrabar stop claims. The alternative is a 55-day high / 20-day low breakout.

## Historical comparison

Two fixed candidates, no parameter sweep. Development: 2022–2023. Held-out comparison: 2024–2025. Recent monitoring: 2026 through the latest completed candle. Momentum beat breakout in development, then was checked on the later periods. These labels describe this experiment only; the underlying historical markets are not unknown to the world.

Each independent segment starts flat with $10,000. Each new position invests 20% of current equity; holdings may drift above that weight. Signals execute at the following daily open. Crypto buy fees reduce received quantity; sell fees reduce cash proceeds. Base costs: 25 bps fee plus 10 bps slippage per side. Stress costs: 25 bps fee plus 25 bps slippage per side. Remaining cash earns zero. Terminal positions are liquidated with costs. Buy-and-hold begins with the same 20% allocation to each asset; it is not volatility/exposure matched thereafter.

| Window | Candidate | Slippage/side | Return | Max drawdown | Round trips |
|---|---|---:|---:|---:|---:|
| development | breakout_55_20 | 10 bps | -0.30% | 11.39% | 10 |
| development | breakout_55_20 | 25 bps | -0.89% | 11.52% | 10 |
| development | momentum_90_200 | 10 bps | +11.29% | 14.17% | 22 |
| development | momentum_90_200 | 25 bps | +9.82% | 14.83% | 22 |
| development | buy_hold | 10 bps | -9.51% | 28.04% | 2 |
| development | buy_hold | 25 bps | -9.60% | 28.02% | 2 |
| holdout | breakout_55_20 | 10 bps | +31.97% | 13.97% | 10 |
| holdout | breakout_55_20 | 25 bps | +31.12% | 14.10% | 10 |
| holdout | momentum_90_200 | 10 bps | +37.17% | 12.44% | 23 |
| holdout | momentum_90_200 | 25 bps | +35.21% | 12.43% | 23 |
| holdout | buy_hold | 10 bps | +26.94% | 24.90% | 2 |
| holdout | buy_hold | 25 bps | +26.74% | 24.88% | 2 |
| recent | breakout_55_20 | 10 bps | -0.15% | 4.85% | 6 |
| recent | breakout_55_20 | 25 bps | -0.51% | 4.98% | 6 |
| recent | momentum_90_200 | 10 bps | +5.12% | 2.39% | 2 |
| recent | momentum_90_200 | 25 bps | +4.98% | 2.39% | 2 |
| recent | buy_hold | 10 bps | -2.86% | 19.83% | 2 |
| recent | buy_hold | 25 bps | -2.97% | 19.81% | 2 |

Cash benchmark: 0% for all periods.

Positive historical returns do not establish a profitable forward strategy. Recent momentum results contain only two round trips, including terminal liquidations. BTC/ETH were selected as currently liquid assets; no claim of a survivorship-free broad-universe test. No statistical significance or annualized return guarantee is claimed. Daily prices do not reconstruct queue position, quote spreads, intraday execution or outages.

## Reproduce

```bash
python -m bot.utils.daily_research --raw research/daily-v1/bars.json --out research/reproduced
python -m pytest -q
```

Raw source: https://data.alpaca.markets/v1beta3/crypto/us/bars

Data SHA256: `80b408dcbdf4dc7c63226f2e6c8ffa78c0fc34ace69389f29cd4192a1fe10316`.

## Paper deployment

The selected config is `research/paper_daily.json`. `python -m bot.trading.daily_paper` plans; add `--submit` to place Alpaca PAPER orders. Both commands require the paper API keys in environment variables or an ignored `.env`. The runner rejects `ALPACA_PAPER=false` and cannot route to a live host.

The sleeve starts with $1,000 and limits each new entry to 20% of the lesser of sleeve equity and $1,000. Dollar caps, the 50 bps spread gate, up to roughly two-hour scheduled execution delay and fees estimated pending activity reconciliation differ from the historical assumptions. Paper outcomes must be measured separately. Exit checks run on daily signals, so this is not an intraday stop-loss system.

The new `Trade` workflow replaces the legacy simulated-clock invocation. It runs several bounded real-clock cycles after UTC midnight, uses repository secrets `ALPACA_API_KEY_ID` and `ALPACA_API_SECRET_KEY`, hardcodes paper mode, serializes runs, and saves order-recovery state. Manual dispatch defaults to plan-only. Scheduled runs submit paper orders. The workflow must be on the repository default branch for the schedule to run.

Existing account holdings are not adopted or liquidated. Use a dedicated empty paper account for this experiment. If the existing account has holdings, the runner stops with an explicit error. Never delete recovery state while positions/orders exist. Never run the old account-trading job alongside this runner.

Unique client order IDs and persisted intentions handle retry after a lost acknowledgement. A partial/unresolved order blocks new exposure. Each cycle saves decisions, quote observations, acknowledgements, fill observations and its data/config manifest. GitHub artifacts retain those records for 90 days; download them before expiration for longer retention. This is not permanent archival storage.

Venue positions are authoritative. Sleeve cash uses observed cumulative fills and a conservative estimated sell fee; reconcile actual CFEE/FEE activities before treating paper P&L as audited. The existing dashboard is not repurposed: it reports the legacy journal, not this new sleeve. Use Alpaca paper orders and the new run artifacts for the experiment.

## Research sources

- https://www.nber.org/papers/w24877 — historical cryptocurrency time-series momentum evidence; does not validate these parameters.
- https://www.nber.org/papers/w25882 — historical cryptocurrency factor evidence.
- https://docs.alpaca.markets/us/docs/crypto-fees — current lowest-tier taker fee 0.25%, with fees in the received asset.
- https://docs.alpaca.markets/us/docs/paper-trading — simulation limitations, including absent market impact and latency slippage.
- https://docs.alpaca.markets/us/docs/working-with-orders — client-order-ID recovery.

## Implementation status

Four reviewed defects corrected: completed-candle visibility in the legacy replay, fixed dollar-risk normalization, separately reported scale-out fees, and stop shortfall based on executed exit quantity. Existing records cannot recover missing historical scale-out fees; retain them as legacy records and rerun rather than silently rewriting history.

Historical data retrieval and mock paper lifecycle tests succeeded. The deployment workflow uses repository-held Alpaca secrets; account connectivity and order execution must be verified from the resulting Actions run, not inferred from these tests. The selected configuration is frozen as daily-v1; a positive backtest is not a guarantee of forward profitability.
