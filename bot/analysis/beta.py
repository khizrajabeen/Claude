"""How much of an instrument's move is really the market's move.

Altcoins are, to a first approximation, levered Bitcoin. Measured over 199
daily observations on the fifteen pairs this bot trades:

    pair        beta to BTC    R^2
    ETH/USDT           1.26   0.80
    SOL/USDT           1.23   0.72
    XRP/USDT           1.24   0.73
    LINK/USDT          1.20   0.66
    SUI/USDT           1.44   0.54
    ADA/USDT           1.40   0.61
    ...
    average pairwise correlation among the alts: 0.62

So five long alts is not five bets. It is about six units of Bitcoin with
some noise on top, and a book that counts it as five independent positions
has understated its true exposure by a factor of five.

The betas used to be a hard-coded table. That table had XRP at 1.00
against a measured 1.24 and ADA at 1.10 against 1.40, and had no entry at
all for SUI, UNI, NEAR, ARB, FIL or BCH — every one of which fell back to
a default. Measuring them from the bars the bot already fetches costs one
covariance per symbol per day and cannot go stale.

Two properties are kept, not one. Beta says how much of the market's move
this instrument takes; R^2 says how much of *its* move is the market's.
A coin with beta 1.0 and R^2 0.2 is a genuinely different bet from one
with beta 1.0 and R^2 0.8, and only the second is a proxy for the index.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")

# The thing each asset class is measured against. Crypto against Bitcoin,
# equities against the S&P proxy — the same split the risk groups use,
# because a beta against the wrong factor is worse than no beta at all.
CLASS_BENCHMARK = {
    "crypto_spot": "BTC/USDT",
    "crypto_perp": "BTC/USDT:USDT",
    "equity": "SPY",
    "etf": "SPY",
}

# Fallbacks when the named benchmark is not in the universe: the most
# liquid instrument of that class stands in.
DEFAULT_BETA = 1.0


class BetaBook:
    """Rolling beta and R-squared of each instrument against its benchmark."""

    def __init__(self, config: dict):
        signals = config.get("signals", {})
        self.window = int(signals.get("beta_window", 120))
        self.min_observations = int(signals.get("beta_min_observations", 40))
        self._beta: dict[str, float] = {}
        self._r2: dict[str, float] = {}
        self._benchmarks: dict[str, str] = {}

    # ── Measurement ───────────────────────────────────────────

    def update(self, frames: dict[str, pd.DataFrame], instruments) -> None:
        """Recompute from whatever frames the briefing already fetched.

        Instruments with too little overlapping history keep the neutral
        default rather than a beta estimated from a fortnight of noise.
        """
        by_class: dict[str, list] = {}
        for instrument in instruments:
            by_class.setdefault(instrument.asset_class.value, []).append(instrument)

        for klass, members in by_class.items():
            benchmark = self._pick_benchmark(klass, members, frames)
            if benchmark is None:
                continue
            base = _returns(frames.get(benchmark), self.window)
            if base is None or len(base) < self.min_observations:
                continue

            for instrument in members:
                symbol = instrument.symbol
                self._benchmarks[symbol] = benchmark
                if symbol == benchmark:
                    self._beta[symbol], self._r2[symbol] = 1.0, 1.0
                    continue
                series = _returns(frames.get(symbol), self.window)
                if series is None:
                    continue
                pair = pd.concat([series, base], axis=1).dropna()
                if len(pair) < self.min_observations:
                    continue
                y = pair.iloc[:, 0].to_numpy(dtype=float)
                x = pair.iloc[:, 1].to_numpy(dtype=float)
                variance = float(np.var(x))
                if variance <= 0:
                    continue
                self._beta[symbol] = float(np.cov(y, x)[0, 1] / variance)
                correlation = float(np.corrcoef(y, x)[0, 1])
                self._r2[symbol] = correlation ** 2 if correlation == correlation else 0.0

    def _pick_benchmark(self, klass: str, members: list,
                        frames: dict[str, pd.DataFrame]) -> str | None:
        """The named benchmark if present, else the longest series available."""
        named = CLASS_BENCHMARK.get(klass)
        if named and named in frames:
            return named
        candidates = [i.symbol for i in members
                      if i.symbol in frames and not frames[i.symbol].empty]
        if not candidates:
            return None
        return max(candidates, key=lambda s: len(frames[s]))

    # ── Lookup ────────────────────────────────────────────────

    def beta(self, symbol: str, default: float = DEFAULT_BETA) -> float:
        return self._beta.get(symbol, default)

    def r2(self, symbol: str, default: float = 0.0) -> float:
        return self._r2.get(symbol, default)

    def benchmark(self, symbol: str) -> str | None:
        return self._benchmarks.get(symbol)

    def known(self, symbol: str) -> bool:
        return symbol in self._beta

    def as_dict(self) -> dict:
        return {s: {"beta": round(b, 3), "r2": round(self._r2.get(s, 0.0), 3),
                    "benchmark": self._benchmarks.get(s, "")}
                for s, b in sorted(self._beta.items())}

    # ── Portfolio view ────────────────────────────────────────

    def effective_positions(self, symbols: list[str]) -> float:
        """How many independent bets a set of positions really is.

        Counting positions is only honest when they are independent. With
        an average pairwise correlation of rho, n positions carry about
        the risk of n / (1 + (n-1) * rho) independent ones — five alts at
        rho 0.62 is roughly 1.5 bets, not five. Returned so a caller can
        say "the book is more concentrated than its position count
        suggests" without re-deriving it.
        """
        n = len(symbols)
        if n <= 1:
            return float(n)
        # R^2 against a common benchmark is a floor on pairwise
        # correlation: two instruments that each track the market closely
        # must track each other closely.
        shared = [self.r2(s) for s in symbols if self.known(s)]
        if not shared:
            return float(n)
        rho = float(np.mean(shared))
        rho = min(max(rho, 0.0), 0.99)
        return n / (1.0 + (n - 1) * rho)


def _returns(df: pd.DataFrame | None, window: int) -> pd.Series | None:
    if df is None or df.empty or "close" not in df:
        return None
    series = df["close"].pct_change().dropna().tail(window)
    return series if len(series) else None
