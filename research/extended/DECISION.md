# Paper experiment decision — 2026-09-26

Retain `daily-v1`, BTC/USD and ETH/USD, `momentum_90_200`. No challenger is promoted and live trading remains disabled. This freezes a forward experiment; it does not certify a profitable strategy.

## Completed comparison

[Results](RESULTS.md) and [machine-readable summary](summary.json) contain 90 fixed comparisons: 15 strategy/universe/side combinations, three chronological periods, and two cost scenarios. These are exploratory comparisons, not 90 independent trials. Source data and each trade ledger/equity curve are preserved in the [research run artifact](https://github.com/khizrajabeen/Claude/actions/runs/36243373969) (90-day retention).

Base-cost portfolio returns, including idle cash:

| Candidate | 2022–2023 | 2024–2025 | 2026 through completed September 25 crypto bar |
|---|---:|---:|---:|
| BTC/ETH momentum 90/200 | +11.30% | +37.17% | +5.07% |
| BTC/ETH momentum 20/100 | -0.85% | +14.17% | +3.61% |
| BTC/ETH breakout 55/20 | -0.30% | +31.97% | -0.20% |
| BTC/ETH two-period reversion | +1.43% | -8.27% | +0.99% |
| BTC/ETH/LTC/LINK momentum 90/200 | -2.24% | +22.98% | +9.67% |

BTC/ETH baseline returns under higher slippage were +9.82%, +35.22%, and +4.93%. Its base-cost maximum drawdowns were 14.16%, 12.44%, and 2.39%. The recent period has only two round trips, including terminal liquidation; it is not sufficient evidence of a durable edge. SOL was excluded for missing daily history, including a 417-day gap; prices were not filled across gaps.

SPY/QQQ momentum 90/200 returned +3.05%, +12.24%, +3.95% long-only versus +1.79%, +9.56%, +2.70% long/short. Faster long/short momentum returned +4.22%, +6.15%, -0.38%; under stressed slippage and short carry it returned -0.32%, +3.68%, -2.11%. ETF buy-and-hold outperformed the tested active ETF rules in the latter two periods. ETF returns omit cash dividends and use IEX data; short carry is a modeled scenario and historical borrow availability is not established. These results do not justify adding short orders to the paper baseline. Alpaca spot crypto does not support short selling.

## Forward execution evidence

The [paper run](https://github.com/khizrajabeen/Claude/actions/runs/36221990641), attempts 3 and 4, records both original buys as `filled`: BTC cumulative fill notional $196.10 and ETH $196.02. Requested notionals were $200.00 and $199.89. Venue-held quantities differ from filled quantities because crypto buy fees are taken in crypto. No pending intents remained; the next cycle returned `no_action`. These are execution checks, not realized profit.

The paper allocation remains $1,000 maximum starting sleeve capital, at most 20% per new asset entry, no borrowing, no pyramiding, and a hardcoded Alpaca paper endpoint. Holdings can drift above initial allocation weights. Signals use completed daily bars, not an intraday profit target. Scheduled cycles run at 08:15/08:30/08:45 and 09:15/09:30/09:45 Asia/Shanghai (00:15–01:45 UTC); GitHub scheduling can be delayed.

## Evaluation discipline

- Preserve the strategy/configuration and log each decision, order, cumulative fill, and equity observation. GitHub artifacts expire after 90 days; they are not permanent archival storage.
- Do not replace the strategy because one recent period or one short paper sample looks better. Historical validation periods were already inspected and are not untouched holdouts.
- Before any future promotion, compare net returns and drawdown against cash and a matched-exposure buy-and-hold benchmark over the same forward dates; include inactivity, turnover, all fees, and fill differences.
- Review missing bars, failed scheduled cycles, stale quotes, and unresolved orders as operational failures, independently of performance.
- Daily profits are not promised. The immediate goal is a reproducible forward record; live trading stays gated.
