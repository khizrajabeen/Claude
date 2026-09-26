# Extended strategy research

Excluded markets: {"SOL/USD": "Missing daily history; largest gap 417 days 00:00:00"}

- Later historical periods were already inspected in prior work; they are reused validation, not pristine holdouts.
- No strategy is guaranteed to profit daily. Cash benchmark is zero; uninvested cash earns zero.
- Universe chosen today; common-history comparison is not survivorship-free.
- 40% of equity allocated across new entries; weights drift; all entries use next daily open.
- ETF returns exclude cash dividends; shorts pay modeled 3% borrow + 1% dividend drag annually (10% + 1% under stress). These are scenarios, not measured borrow quotes.
- IEX data may differ from consolidated prices; historical short availability is not reconstructed.
- Crypto shorting is unsupported by Alpaca and is not submitted or simulated as an executable Alpaca strategy.
- Multiple comparisons are exploratory; the frozen paper baseline is not automatically replaced.

| Universe | Rule | Side | Window | Costs | Return | Drawdown | Trades |
|---|---|---|---|---|---:|---:|---:|
| crypto_2 | momentum_90_200 | long | development | base | +11.30% | 14.16% | 22 |
| crypto_2 | momentum_90_200 | long | development | stress | +9.82% | 14.82% | 22 |
| crypto_2 | momentum_90_200 | long | reused_validation | base | +37.17% | 12.44% | 23 |
| crypto_2 | momentum_90_200 | long | reused_validation | stress | +35.22% | 12.43% | 23 |
| crypto_2 | momentum_90_200 | long | recent | base | +5.07% | 2.39% | 2 |
| crypto_2 | momentum_90_200 | long | recent | stress | +4.93% | 2.39% | 2 |
| crypto_2 | momentum_20_100 | long | development | base | -0.85% | 16.63% | 45 |
| crypto_2 | momentum_20_100 | long | development | stress | -3.49% | 18.55% | 45 |
| crypto_2 | momentum_20_100 | long | reused_validation | base | +14.17% | 21.56% | 67 |
| crypto_2 | momentum_20_100 | long | reused_validation | stress | +9.66% | 22.64% | 67 |
| crypto_2 | momentum_20_100 | long | recent | base | +3.61% | 4.19% | 19 |
| crypto_2 | momentum_20_100 | long | recent | stress | +2.43% | 4.87% | 19 |
| crypto_2 | breakout_55_20 | long | development | base | -0.30% | 11.39% | 10 |
| crypto_2 | breakout_55_20 | long | development | stress | -0.89% | 11.52% | 10 |
| crypto_2 | breakout_55_20 | long | reused_validation | base | +31.97% | 13.97% | 10 |
| crypto_2 | breakout_55_20 | long | reused_validation | stress | +31.12% | 14.10% | 10 |
| crypto_2 | breakout_55_20 | long | recent | base | -0.20% | 4.85% | 6 |
| crypto_2 | breakout_55_20 | long | recent | stress | -0.55% | 4.98% | 6 |
| crypto_2 | reversion_2_200 | long | development | base | +1.43% | 6.25% | 79 |
| crypto_2 | reversion_2_200 | long | development | stress | -3.26% | 8.48% | 79 |
| crypto_2 | reversion_2_200 | long | reused_validation | base | -8.27% | 10.91% | 114 |
| crypto_2 | reversion_2_200 | long | reused_validation | stress | -14.32% | 16.18% | 114 |
| crypto_2 | reversion_2_200 | long | recent | base | +0.99% | 1.18% | 8 |
| crypto_2 | reversion_2_200 | long | recent | stress | +0.50% | 1.35% | 8 |
| crypto_2 | buy_hold | long | development | base | -9.51% | 28.04% | 2 |
| crypto_2 | buy_hold | long | development | stress | -9.60% | 28.02% | 2 |
| crypto_2 | buy_hold | long | reused_validation | base | +26.94% | 24.90% | 2 |
| crypto_2 | buy_hold | long | reused_validation | stress | +26.74% | 24.88% | 2 |
| crypto_2 | buy_hold | long | recent | base | -2.90% | 19.83% | 2 |
| crypto_2 | buy_hold | long | recent | stress | -3.01% | 19.81% | 2 |
| crypto_4 | momentum_90_200 | long | development | base | -2.24% | 17.50% | 54 |
| crypto_4 | momentum_90_200 | long | development | stress | -3.82% | 18.12% | 54 |
| crypto_4 | momentum_90_200 | long | reused_validation | base | +22.98% | 14.07% | 41 |
| crypto_4 | momentum_90_200 | long | reused_validation | stress | +21.43% | 14.18% | 41 |
| crypto_4 | momentum_90_200 | long | recent | base | +9.67% | 3.93% | 5 |
| crypto_4 | momentum_90_200 | long | recent | stress | +9.49% | 3.93% | 5 |
| crypto_4 | momentum_20_100 | long | development | base | -12.13% | 21.49% | 106 |
| crypto_4 | momentum_20_100 | long | development | stress | -14.86% | 23.35% | 106 |
| crypto_4 | momentum_20_100 | long | reused_validation | base | +0.16% | 22.17% | 126 |
| crypto_4 | momentum_20_100 | long | reused_validation | stress | -3.54% | 23.06% | 126 |
| crypto_4 | momentum_20_100 | long | recent | base | +8.94% | 5.13% | 29 |
| crypto_4 | momentum_20_100 | long | recent | stress | +7.98% | 5.30% | 29 |
| crypto_4 | breakout_55_20 | long | development | base | -3.42% | 16.64% | 18 |
| crypto_4 | breakout_55_20 | long | development | stress | -3.93% | 16.96% | 18 |
| crypto_4 | breakout_55_20 | long | reused_validation | base | +19.70% | 14.44% | 19 |
| crypto_4 | breakout_55_20 | long | reused_validation | stress | +18.97% | 14.54% | 19 |
| crypto_4 | breakout_55_20 | long | recent | base | +2.75% | 5.14% | 10 |
| crypto_4 | breakout_55_20 | long | recent | stress | +2.44% | 5.26% | 10 |
| crypto_4 | reversion_2_200 | long | development | base | -1.29% | 6.04% | 133 |
| crypto_4 | reversion_2_200 | long | development | stress | -5.14% | 8.22% | 133 |
| crypto_4 | reversion_2_200 | long | reused_validation | base | -13.38% | 15.00% | 194 |
| crypto_4 | reversion_2_200 | long | reused_validation | stress | -18.24% | 19.52% | 194 |
| crypto_4 | reversion_2_200 | long | recent | base | +0.52% | 1.75% | 15 |
| crypto_4 | reversion_2_200 | long | recent | stress | +0.07% | 1.88% | 15 |
| crypto_4 | buy_hold | long | development | base | -12.21% | 28.73% | 4 |
| crypto_4 | buy_hold | long | development | stress | -12.30% | 28.70% | 4 |
| crypto_4 | buy_hold | long | reused_validation | base | +12.04% | 26.68% | 4 |
| crypto_4 | buy_hold | long | reused_validation | stress | +11.88% | 26.66% | 4 |
| crypto_4 | buy_hold | long | recent | base | -0.71% | 20.03% | 4 |
| crypto_4 | buy_hold | long | recent | stress | -0.83% | 20.02% | 4 |
| etfs | momentum_90_200 | long | development | base | +3.05% | 5.70% | 19 |
| etfs | momentum_90_200 | long | development | stress | +2.27% | 6.03% | 19 |
| etfs | momentum_90_200 | long | reused_validation | base | +12.24% | 4.76% | 14 |
| etfs | momentum_90_200 | long | reused_validation | stress | +11.60% | 4.76% | 14 |
| etfs | momentum_90_200 | long | recent | base | +3.95% | 3.38% | 8 |
| etfs | momentum_90_200 | long | recent | stress | +3.61% | 3.38% | 8 |
| etfs | momentum_90_200 | long/short | development | base | +1.79% | 7.60% | 41 |
| etfs | momentum_90_200 | long/short | development | stress | -2.10% | 9.13% | 41 |
| etfs | momentum_90_200 | long/short | reused_validation | base | +9.56% | 7.56% | 17 |
| etfs | momentum_90_200 | long/short | reused_validation | stress | +8.26% | 7.98% | 17 |
| etfs | momentum_90_200 | long/short | recent | base | +2.70% | 3.98% | 11 |
| etfs | momentum_90_200 | long/short | recent | stress | +2.10% | 4.36% | 11 |
| etfs | momentum_20_100 | long | development | base | +2.52% | 6.59% | 28 |
| etfs | momentum_20_100 | long | development | stress | +1.38% | 7.05% | 28 |
| etfs | momentum_20_100 | long | reused_validation | base | +9.31% | 4.24% | 36 |
| etfs | momentum_20_100 | long | reused_validation | stress | +7.74% | 4.45% | 36 |
| etfs | momentum_20_100 | long | recent | base | +2.78% | 2.93% | 23 |
| etfs | momentum_20_100 | long | recent | stress | +1.83% | 3.44% | 23 |
| etfs | momentum_20_100 | long/short | development | base | +4.22% | 5.62% | 61 |
| etfs | momentum_20_100 | long/short | development | stress | -0.32% | 7.41% | 61 |
| etfs | momentum_20_100 | long/short | reused_validation | base | +6.15% | 6.44% | 45 |
| etfs | momentum_20_100 | long/short | reused_validation | stress | +3.68% | 6.77% | 45 |
| etfs | momentum_20_100 | long/short | recent | base | -0.38% | 3.71% | 34 |
| etfs | momentum_20_100 | long/short | recent | stress | -2.11% | 4.59% | 34 |
| etfs | buy_hold | long | development | base | +0.44% | 12.08% | 2 |
| etfs | buy_hold | long | development | stress | +0.36% | 12.07% | 2 |
| etfs | buy_hold | long | reused_validation | base | +19.10% | 9.77% | 2 |
| etfs | buy_hold | long | reused_validation | stress | +18.98% | 9.77% | 2 |
| etfs | buy_hold | long | recent | base | +6.25% | 4.23% | 2 |
| etfs | buy_hold | long | recent | stress | +6.15% | 4.23% | 2 |
