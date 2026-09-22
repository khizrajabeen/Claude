"""News-driven entries, for instruments whose tape is too slow to signal.

Crypto prints a bar every fifteen minutes and a technical strategy has
plenty to work with. A US stock prints one bar a day and gets one entry
slot per session, so over a 90-day replay the equity leg managed nine
trades — not because the strategies disliked the setups but because there
were barely any setups to have an opinion about. A daily Donchian channel
needs twenty sessions to say anything, and by then the month is over.

News is the opposite: it arrives continuously, it is per-company, and it
is the thing that actually moves a single stock on a given day. So for
equities the story is the signal and the chart is the filter, rather than
the other way round.

Three things keep this from being a headline-chasing machine:

  **Coverage before conviction.** A +0.9 sentiment score drawn from two
  stories is a strong opinion weakly held. The signal uses the analyzer's
  signal_strength — consensus times coverage times magnitude — so a lone
  excitable headline cannot open a position.

  **Freshness matters, but staleness is the danger.** A story that broke
  this morning has not been fully priced; one the tape has had all day to
  digest has. The signal is scaled by how much of the sentiment is recent.

  **The chart still holds a veto.** A bullish story on an instrument in a
  primary downtrend is not a reason to buy it. Direction must agree with
  the trend read, which is the same gate every other strategy passes.
"""

from __future__ import annotations

import logging

from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal

logger = logging.getLogger("trading_bot")


class NewsDriven(BaseStrategy):
    """Trades single-name news for instruments the chart cannot signal on."""

    name = "news"
    stance = "confirmation"

    def __init__(self, config: dict):
        super().__init__(config)
        self.min_articles = int(self._param("min_articles", 6))
        self.min_score = float(self._param("min_score", 0.12))
        self.min_strength = float(self._param("min_strength", 0.05))
        self.fresh_weight = float(self._param("fresh_weight", 0.4))
        self.require_trend_agreement = bool(
            self._param("require_trend_agreement", True))
        # Which asset classes this runs on. Crypto is left to the chart:
        # its tape is fast enough to signal on, and crypto headlines are
        # mostly about the whole market rather than one coin.
        self.asset_classes = set(self._param(
            "asset_classes", ["equity", "etf"]))

    def required_bars(self) -> int:
        # Only enough to have a price and a trend read; the signal itself
        # comes from the wire, not the chart.
        return 30

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            read = ctx.reads.get(symbol)
            if read is None:
                continue
            if self.asset_classes and str(read.asset_class) not in self.asset_classes:
                continue

            articles = int(getattr(read, "news_articles", 0) or 0)
            score = float(getattr(read, "news_score", 0.0) or 0.0)
            strength = float(getattr(read, "news_strength", 0.0) or 0.0)
            if articles < self.min_articles:
                continue
            if abs(score) < self.min_score or strength < self.min_strength:
                continue

            direction = 1 if score > 0 else -1

            # A bullish story on something in a primary downtrend is not a
            # reason to buy it.
            if self.require_trend_agreement:
                trend = int(getattr(read, "primary_trend", 0) or 0)
                if trend and trend != direction:
                    continue

            # How much of the opinion is recent. A story already a day old
            # has been priced; one that broke this morning has not.
            fresh = float(getattr(read, "news_short_score", 0.0) or 0.0)
            freshness = 0.0
            if score:
                # Same sign and at least as strong = fully fresh.
                freshness = max(0.0, min(1.0, (fresh / score)))

            conviction = min(1.0, abs(score) * 2.0)
            coverage = min(1.0, articles / 20.0)
            base = 0.45 * conviction + 0.35 * strength + 0.20 * coverage
            score_out = direction * base * (
                1.0 - self.fresh_weight + self.fresh_weight * freshness
            )

            signal = self.signal(
                symbol, score_out,
                reason=(f"news {score:+.2f} over {articles} stories "
                        f"(strength {strength:.2f}, "
                        f"{freshness * 100:.0f}% still fresh)"),
                horizon_bars=int(self._param("horizon_bars", 5)),
                news_score=round(score, 3),
                news_articles=articles,
                news_strength=round(strength, 3),
                freshness=round(freshness, 3),
            )
            if signal:
                signals.append(signal)

        return signals
