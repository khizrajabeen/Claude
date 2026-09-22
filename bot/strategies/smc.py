"""Smart Money Concepts, implemented mechanically so it can be judged.

The framework, as LuxAlgo's indicator formalises it:

    BOS      a break of the last swing high/low in the trend's direction —
             continuation
    CHoCH    a break against the prevailing structure — the first evidence
             of a reversal
    FVG      a three-bar imbalance where price skipped a range entirely
    Sweep    price takes out a prior extreme and immediately reverses,
             i.e. stops were run
    Premium / discount
             where price sits in the current structural range; buy the
             discount half, sell the premium half

**What the evidence says.** The most careful published study of these
signals — 648 backtests across four markets — found no ICT/SMC signal
produced a statistically significant forward edge, and nothing beat
buy-and-hold. Mechanical order blocks and fair value gaps came out "real
but modest", capturing a fraction of the index's own drift. Reported win
rates of 50-65% come from discretionary implementations; stringent
mechanical testing puts the raw hit rate nearer 38-45%.

So this is here to be measured, not believed. The prior is that it adds
little; the bench and the significance tests decide. What it does bring
structurally is a genuinely different entry logic — a sweep-and-reverse
fires where every trend system is silent, so even a weak signal can
diversify.

The one non-negotiable detail is swing confirmation. A swing high is only
knowable `length` bars after it prints, and structure indicators that
ignore that are reading the future. `swing_points` shifts accordingly, and
a look-ahead test pins it.
"""

from __future__ import annotations

import numpy as np

from bot.analysis import indicators as ind
from bot.strategies.base import BaseStrategy, MarketContext, StrategySignal


class SmartMoneyConcepts(BaseStrategy):
    """Structure breaks, liquidity sweeps and imbalances."""

    name = "smc"

    def __init__(self, config: dict):
        super().__init__(config)
        self.swing_length = int(self._param("swing_length", 10))
        self.sweep_lookback = int(self._param("sweep_lookback", 30))
        self.sweep_reversal_atr = float(self._param("sweep_reversal_atr", 0.5))
        self.use_fvg = bool(self._param("use_fvg", True))
        self.fvg_lookback = int(self._param("fvg_lookback", 20))
        self.use_premium_discount = bool(self._param("use_premium_discount", True))
        self.max_extension_atr = float(self._param("max_extension_atr", 2.0))
        self.dealing_range_bars = int(self._param("dealing_range_bars", 60))

    def required_bars(self) -> int:
        return max(self.swing_length * 8 + self.sweep_lookback,
                   self.dealing_range_bars) + 60

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        if not self.enabled:
            return []

        signals: list[StrategySignal] = []
        for symbol in ctx.tradable():
            df = ctx.frames.get(symbol)
            if df is None or len(df) < self.required_bars():
                continue

            atr_now = ind.last_value(ind.atr(df, 14))
            if atr_now <= 0:
                continue

            swing_high, swing_low = ind.swing_points(df, self.swing_length)
            last_high = ind.last_value(swing_high)
            last_low = ind.last_value(swing_low)
            if not last_high or not last_low or last_high <= last_low:
                continue

            price = float(df["close"].iloc[-1])
            components: dict[str, float] = {}
            notes: list[str] = []

            structure = self._structure(df, swing_high, swing_low, price, atr_now)
            if structure:
                components["structure"] = structure["score"]
                notes.append(structure["note"])

            sweep = self._liquidity_sweep(df, atr_now)
            if sweep:
                components["sweep"] = sweep["score"]
                notes.append(sweep["note"])

            if self.use_fvg:
                gap = self._fair_value_gap(df, price, atr_now)
                if gap:
                    components["fvg"] = gap["score"]
                    notes.append(gap["note"])

            if self.use_premium_discount:
                zone = self._premium_discount(df, price)
                if zone:
                    components["zone"] = zone["score"]
                    notes.append(zone["note"])

            if not components:
                continue

            # Equal weight across whichever components fired. Weighting them
            # would need evidence this framework does not have.
            raw = float(np.mean(list(components.values())))
            score = float(np.clip(raw, -1.0, 1.0))

            signal = self.signal(
                symbol, score,
                reason="SMC: " + "; ".join(notes),
                horizon_bars=self.swing_length * 3,
                **{k: round(v, 3) for k, v in components.items()},
            )
            if signal:
                signals.append(signal)

        return signals

    # ── Components ────────────────────────────────────────────

    def _structure(self, df, swing_high, swing_low, price: float,
                   atr_now: float) -> dict | None:
        """Break of structure, and whether it continues or reverses trend."""
        highs = swing_high.dropna()
        lows = swing_low.dropna()
        if len(highs) < 2 or len(lows) < 2:
            return None

        last_high = float(highs.iloc[-1])
        prior_high = float(highs.iloc[-2])
        last_low = float(lows.iloc[-1])
        prior_low = float(lows.iloc[-2])

        # Prevailing structure: higher highs and higher lows is an uptrend.
        trend = 0
        if last_high > prior_high and last_low > prior_low:
            trend = 1
        elif last_high < prior_high and last_low < prior_low:
            trend = -1

        if price > last_high:
            side, level = 1, last_high
        elif price < last_low:
            side, level = -1, last_low
        else:
            return None

        extension = abs(price - level) / atr_now
        if extension > self.max_extension_atr:
            return None  # the break already happened without us

        kind = "BOS" if (trend == 0 or side == trend) else "CHoCH"
        # A change of character is the more informative event but the less
        # reliable one, so it is not credited more.
        base = 0.6 if kind == "BOS" else 0.5
        freshness = float(np.clip(1.0 - extension / self.max_extension_atr, 0.0, 1.0))
        return {
            "score": side * (base * 0.5 + base * 0.5 * freshness),
            "note": f"{kind} at {level:.4f} ({extension:.1f} ATR past)",
        }

    def _liquidity_sweep(self, df, atr_now: float) -> dict | None:
        """A prior extreme taken out, then immediately rejected.

        The signal is the rejection, not the break: a wick through a level
        that closes back inside says the stops beyond it were run and the
        move had no follow-through.
        """
        window = df.tail(self.sweep_lookback + 1)
        if len(window) < self.sweep_lookback:
            return None

        recent = window.iloc[-1]
        prior = window.iloc[:-1]
        prior_high = float(prior["high"].max())
        prior_low = float(prior["low"].min())

        high, low, close = float(recent["high"]), float(recent["low"]), float(recent["close"])

        # Swept the highs and closed back below them: bearish.
        if high > prior_high and close < prior_high:
            rejection = (high - close) / atr_now
            if rejection >= self.sweep_reversal_atr:
                return {
                    "score": -float(np.clip(0.4 + 0.3 * rejection, 0.0, 1.0)),
                    "note": f"swept highs at {prior_high:.4f}, rejected {rejection:.1f} ATR",
                }

        if low < prior_low and close > prior_low:
            rejection = (close - low) / atr_now
            if rejection >= self.sweep_reversal_atr:
                return {
                    "score": float(np.clip(0.4 + 0.3 * rejection, 0.0, 1.0)),
                    "note": f"swept lows at {prior_low:.4f}, rejected {rejection:.1f} ATR",
                }
        return None

    def _fair_value_gap(self, df, price: float, atr_now: float) -> dict | None:
        """An unfilled imbalance price is returning into."""
        direction, top, bottom = ind.fair_value_gaps(df)
        recent = direction.tail(self.fvg_lookback)
        active = recent[recent != 0]
        if active.empty:
            return None

        stamp = active.index[-1]
        side = int(active.iloc[-1])
        gap_top = float(top.loc[stamp])
        gap_bottom = float(bottom.loc[stamp])
        if not np.isfinite(gap_top) or not np.isfinite(gap_bottom):
            return None

        # The trade is price coming back *into* the gap, not price leaving it.
        inside = gap_bottom <= price <= gap_top
        if not inside:
            return None

        size_atr = (gap_top - gap_bottom) / atr_now
        return {
            "score": side * float(np.clip(0.35 + 0.25 * size_atr, 0.0, 1.0)),
            "note": f"{'bullish' if side > 0 else 'bearish'} FVG "
                    f"{gap_bottom:.4f}-{gap_top:.4f} ({size_atr:.1f} ATR)",
        }

    def _premium_discount(self, df, price: float) -> dict | None:
        """Where price sits in the current dealing range.

        The range is the high and low of a bounded recent window, not the
        last confirmed swing pair. In a sustained trend the last confirmed
        swings can be hundreds of bars old, which put price at "638% of
        range" — a number that means the range had stopped describing the
        market. A rolling window is what "current dealing range" actually
        refers to.

        Stripped of the vocabulary this is mean reversion within a range,
        so it is also clamped: being far outside the range is not a
        stronger signal, it is evidence the range has broken.
        """
        window = df.tail(self.dealing_range_bars)
        if len(window) < self.dealing_range_bars // 2:
            return None

        high = float(window["high"].max())
        low = float(window["low"].min())
        span = high - low
        if span <= 0:
            return None

        position = (price - low) / span
        if not (-0.1 <= position <= 1.1):
            return None  # price has left the range; this is not a fade

        position = float(np.clip(position, 0.0, 1.0))
        # 0 at the low (discount, favour longs), 1 at the high (premium).
        score = (0.5 - position) * 1.2
        label = "discount" if position < 0.45 else "premium" if position > 0.55 else "equilibrium"
        return {
            "score": float(np.clip(score, -1.0, 1.0)),
            "note": f"{label} ({position * 100:.0f}% of range)",
        }
