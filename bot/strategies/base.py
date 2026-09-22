"""The strategy interface.

Each strategy is a separate *return driver*, not a variation on one. That
distinction is the whole point: the published evidence on managed futures
is that combining weakly-correlated drivers is what shrinks drawdown, more
than any improvement to a single signal. A trend model and a carry model
lose money at different times; two trend models lose money together.

A strategy answers one question — "what would you hold, and how strongly?"
— and knows nothing about sizing, stops or the portfolio. Position size is
the risk layer's job, and how much capital each strategy gets is the
allocator's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import pandas as pd


@dataclass
class MarketContext:
    """Everything a strategy is allowed to see on a given day."""

    day: str
    reads: dict                          # symbol -> SymbolRead
    frames: dict[str, pd.DataFrame]      # symbol -> decision-timeframe OHLCV
    higher_frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    funding: dict[str, float] = field(default_factory=dict)
    market_tone: float = 0.0
    equity: float = 0.0
    timeframe: str = "1h"

    def tradable(self) -> list[str]:
        return [s for s, r in self.reads.items() if r.tradable and s in self.frames]

    def close(self, symbol: str) -> pd.Series | None:
        df = self.frames.get(symbol)
        return df["close"] if df is not None and not df.empty else None


@dataclass
class StrategySignal:
    """One strategy's opinion on one symbol."""

    strategy: str
    symbol: str
    direction: int          # +1 long, -1 short, 0 flat
    strength: float         # 0..1 conviction, comparable across strategies
    reason: str = ""
    horizon_bars: int = 0
    # "confirmation" rides the prevailing move; "contrarian" fades it.
    # The distinction is not cosmetic: a fade has less lag but far more
    # exposure to a large adverse move, because the thing it is betting
    # against is exactly the thing that is currently working. The risk
    # layer sizes the two differently.
    kind: str = "confirmation"
    meta: dict = field(default_factory=dict)

    @property
    def signed(self) -> float:
        return self.direction * self.strength

    @property
    def contrarian(self) -> bool:
        return self.kind == "contrarian"

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy, "symbol": self.symbol,
            "direction": self.direction, "strength": round(self.strength, 4),
            "kind": self.kind, "reason": self.reason, "meta": self.meta,
        }


@runtime_checkable
class Strategy(Protocol):
    """What every strategy must provide."""

    name: str

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:
        """Signals for this day. Symbols with no view are simply omitted."""
        ...


class BaseStrategy:
    """Shared plumbing: config access, enable flag, strength clamping."""

    name: str = "base"
    # Strategies that can hold both sides at once are marked so the
    # allocator does not treat a hedged pair as crowding.
    market_neutral: bool = False
    # Whether this strategy rides moves or fades them. Trend and breakout
    # systems confirm; envelope fades and sweep reversals oppose.
    stance: str = "confirmation"

    def __init__(self, config: dict):
        self.config = config
        self.params = config.get("strategies", {}).get(self.name, {})
        self.enabled = bool(self.params.get("enabled", True))
        self.min_strength = float(self.params.get("min_strength", 0.15))

    def generate(self, ctx: MarketContext) -> list[StrategySignal]:  # pragma: no cover
        raise NotImplementedError

    def required_bars(self) -> int:
        """Bars of history this strategy needs to produce any signal.

        Declared rather than discovered, because a strategy starved of
        history fails silently: it returns no signals and looks merely
        opinionless. Startup validation compares this against the configured
        fetch depth so the failure is loud instead.
        """
        return 120

    # ── helpers ───────────────────────────────────────────────

    def signal(self, symbol: str, score: float, reason: str = "",
               horizon_bars: int = 0, kind: str | None = None,
               **meta) -> StrategySignal | None:
        """Build a signal from a signed score, or None if it is too weak.

        Scores arrive in roughly [-1, 1]; strength is the magnitude, so a
        strategy that is barely leaning does not compete with one that is
        certain.
        """
        strength = min(1.0, abs(float(score)))
        if strength < self.min_strength:
            return None
        return StrategySignal(
            strategy=self.name,
            symbol=symbol,
            direction=1 if score > 0 else -1,
            strength=strength,
            reason=reason,
            horizon_bars=horizon_bars,
            kind=kind or self.stance,
            meta=meta,
        )

    def _param(self, key: str, default):
        return self.params.get(key, default)
