"""Funding carry — get paid to take the unpopular side of a perpetual.

A perpetual swap has no expiry, so it is tethered to spot by a funding
payment every eight hours. When the crowd is long, longs pay shorts, and a
short collects that yield whether or not price moves. This is the crypto
version of the classic futures basis trade.

Two honest caveats, both load-bearing:

  * **The edge has compressed.** Published work puts the carry Sharpe above
    6 over 2020-2023, falling through 2024 and turning negative in 2025 as
    delta-neutral yield products crowded the trade. Recent samples show
    average funding below the threshold at which it covers costs, meaning a
    disciplined carry rule simply does not fire. That is the correct
    behaviour, not a broken strategy, and this implementation is built to
    stand aside rather than to chase a yield that no longer clears fees.
  * **Directional carry is not the basis trade.** A true cash-and-carry is
    long spot and short the perp, leaving almost no price risk. Running a
    naked short to collect funding leaves all of it. This strategy is
    therefore weighted low by default and doubles as a crowding signal:
    sustained extreme funding has historically preceded the sharpest
    reversals, because it means one side is leveraged and vulnerable.

On a spot-only venue there are no funding rates, so this yields nothing and
says so once.
"""

from __future__ import annotations

import logging

import numpy as np

from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal

logger = logging.getLogger("trading_bot")

SETTLEMENTS_PER_YEAR = 3 * 365  # 8-hourly


class CarryStrategy(BaseStrategy):
    """Take the paid side of funding, and fade extreme crowding."""

    name = "carry"

    def __init__(self, config: dict):
        super().__init__(config)
        # Per-settlement rate at which carry is worth the price risk.
        # 1bp per 8h is ~11% a year before costs.
        self.entry_bps = float(self._param("entry_bps", 1.0))
        self.extreme_bps = float(self._param("extreme_bps", 5.0))
        self.max_bps = float(self._param("max_bps", 15.0))
        self.contrarian = bool(self._param("contrarian_on_extreme", True))
        self._warned = False

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        rates = {s: r for s, r in ctx.funding.items() if r}
        if not rates:
            if not self._warned:
                logger.info(
                    "Carry: no funding rates available (spot venue or "
                    "unsupported endpoint) — strategy inactive"
                )
                self._warned = True
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            rate = rates.get(symbol)
            if rate is None:
                continue

            bps = rate * 10_000  # per settlement
            annual_pct = rate * SETTLEMENTS_PER_YEAR * 100

            if abs(bps) < self.entry_bps:
                continue  # yield does not clear the cost of holding

            # Positive funding: longs pay, so the paid side is short.
            side = -1 if bps > 0 else 1
            reason = f"funding {bps:+.2f}bp/8h ({annual_pct:+.1f}%/yr), collect as " \
                     f"{'short' if side < 0 else 'long'}"

            if self.contrarian and abs(bps) >= self.extreme_bps:
                # At extremes the crowding read and the carry read agree:
                # fading a leveraged crowd is the same trade as collecting
                # from it, so conviction rises rather than the sign flipping.
                reason += " — crowded positioning"

            # Conviction saturates: twice the funding is not twice the edge,
            # because the tail risk of the naked leg grows with it.
            span = max(1e-9, self.max_bps - self.entry_bps)
            excess = (abs(bps) - self.entry_bps) / span
            score = side * float(np.clip(0.35 + 0.65 * np.tanh(2 * excess), 0.0, 1.0))

            signal = self.signal(
                symbol, score, reason=reason, horizon_bars=24,
                funding_bps=round(bps, 3), annual_pct=round(annual_pct, 2),
            )
            if signal:
                signals.append(signal)

        return signals
