"""Shared trade/position records.

These are the objects the paper trader, the live trader and the journal all
agree on, so a position can be written to disk at the end of one day and
resumed at the start of the next without any lossy translation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    return datetime.fromisoformat(str(value))


@dataclass
class Position:
    """An open position, sized so that a stop-out costs `risk_usd`."""

    id: str
    symbol: str
    side: str  # "long" | "short"
    entry_price: float
    quantity: float
    leverage: float
    opened_at: datetime
    stop_price: float
    take_profit: float = 0.0
    initial_stop: float = 0.0
    risk_usd: float = 0.0
    margin: float = 0.0
    entry_fee: float = 0.0
    funding_paid: float = 0.0
    atr_at_entry: float = 0.0
    best_price: float = 0.0
    worst_price: float = 0.0
    moved_to_breakeven: bool = False
    opened_on_day: str = ""
    bars_held: int = 0
    strategy: str = ""
    entry_reason: str = ""
    asset_class: str = "crypto_spot"
    venue: str = ""
    timeframe: str = ""
    # Scaling in and out. `units` counts how many entries built this
    # position; `original_quantity` is what it opened with, so a
    # partial exit can be measured against the position that was sized,
    # not against whatever is left of it.
    units: int = 1
    original_quantity: float = 0.0
    scaled_out_at: list = field(default_factory=list)
    # Which configured rungs have already fired, so each fires once.
    scaled_out_at_levels: list = field(default_factory=list)
    realized_pnl: float = 0.0
    tags: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.initial_stop:
            self.initial_stop = self.stop_price
        if not self.best_price:
            self.best_price = self.entry_price
        if not self.worst_price:
            self.worst_price = self.entry_price
        if not self.original_quantity:
            self.original_quantity = self.quantity

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def notional(self) -> float:
        return self.entry_price * self.quantity

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.entry_price) * self.quantity * self.direction

    def unrealized_r(self, price: float) -> float:
        """Profit in units of initial risk — the only scale worth comparing
        across assets."""
        risk_per_unit = abs(self.entry_price - self.initial_stop)
        if risk_per_unit <= 0:
            return 0.0
        return (price - self.entry_price) * self.direction / risk_per_unit

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opened_at"] = _iso(self.opened_at)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Position":
        d = dict(d)
        d["opened_at"] = _parse(d.get("opened_at")) or datetime.now(timezone.utc)
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Trade:
    """A closed round trip."""

    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    leverage: float
    opened_at: datetime
    closed_at: datetime
    pnl: float
    pnl_pct: float
    r_multiple: float
    fees: float
    funding: float
    slippage_cost: float
    exit_reason: str
    risk_usd: float = 0.0
    opened_on_day: str = ""
    closed_on_day: str = ""
    strategy: str = ""
    entry_reason: str = ""
    asset_class: str = "crypto_spot"
    venue: str = ""
    timeframe: str = ""
    mae_r: float = 0.0  # worst excursion, in R
    mfe_r: float = 0.0  # best excursion, in R

    # ── Audit trail ──────────────────────────────────────────
    # Enough to reconstruct how a trade reached its R without the
    # journal. A reviewer asked how a -2.66R stop-out was possible on a
    # position sized for -1R, and the record could not answer: it
    # carried the averaged entry and the exit, and neither the units
    # added nor the stop that R was measured against. Two quite
    # different mechanisms — a pyramid that grew the risk, and a fill
    # well past the stop — produce an identical row.
    units: int = 1                  # 1 unless the position was pyramided
    original_quantity: float = 0.0  # size before any add: the R denominator
    initial_stop: float = 0.0       # the stop R is measured against
    final_stop: float = 0.0         # where the stop actually sat at exit
    # Profit banked BEFORE this exit, by scaling out. Without it the row
    # cannot be reconciled: `pnl` includes these legs while
    # `(exit - entry) * quantity` does not, so all 43 trades in the last
    # replay failed an independent recomputation by up to $48.
    realized_before_exit: float = 0.0
    exit_quantity: float = 0.0      # size on the final leg only

    @property
    def holding_minutes(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds() / 60.0

    def to_dict(self) -> dict:
        d = asdict(self)
        d["opened_at"] = _iso(self.opened_at)
        d["closed_at"] = _iso(self.closed_at)
        d["holding_minutes"] = round(self.holding_minutes, 2)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Trade":
        d = dict(d)
        d.pop("holding_minutes", None)
        d["opened_at"] = _parse(d.get("opened_at")) or datetime.now(timezone.utc)
        d["closed_at"] = _parse(d.get("closed_at")) or datetime.now(timezone.utc)
        known = {f for f in cls.__dataclass_fields__}
        out = {}
        for k, v in d.items():
            if k not in known:
                continue
            if k in ("opened_at", "closed_at"):
                out[k] = v
            elif k in ("id", "symbol", "side", "exit_reason", "opened_on_day",
                       "closed_on_day", "strategy", "entry_reason",
                       "asset_class", "venue", "timeframe"):
                out[k] = "" if v is None else str(v)
            else:
                out[k] = float(v) if v not in (None, "") else 0.0
        return cls(**out)
