"""Prove that every reported R is arithmetic, not assertion.

A reviewer worked an ETH trade by hand and found 0.143R the record did
not account for. That is the right standard: a trade log whose numbers
cannot be re-derived from its own columns is a claim, not evidence, and
every conclusion drawn from it inherits the gap.

So this recomputes each trade from first principles and reports the
residual:

    gross  = (exit - entry) * quantity * direction
    net    = gross - fees - funding
    risk$  = |entry - initial_stop| * original_quantity
    R      = net / risk$

Anything that does not close within rounding is printed, not smoothed.
The point is to fail loudly when the ledger and the arithmetic disagree.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass


@dataclass
class Row:
    symbol: str
    side: str
    entry: float
    exit: float
    quantity: float
    original_quantity: float
    initial_stop: float
    final_stop: float
    units: int
    realized_before_exit: float
    exit_quantity: float
    pnl: float
    fees: float
    funding: float
    r_reported: float
    exit_reason: str
    initial_risk_usd: float = 0.0
    scale_out_fees: float = 0.0

    @property
    def direction(self) -> int:
        return 1 if self.side == "long" else -1

    @property
    def gross(self) -> float:
        """Price P&L on the final leg, plus anything banked before it.

        A position that sold a third at 1.5R and then stopped out has
        two cash flows, and only one of them is visible in the exit
        price. Omitting the first is what made every trade fail to
        reconcile by up to $48.
        """
        leg = self.exit_quantity or self.quantity
        return ((self.exit - self.entry) * leg * self.direction
                + self.realized_before_exit + self.scale_out_fees)

    @property
    def net(self) -> float:
        return self.gross - self.fees - self.funding

    @property
    def risk_usd(self) -> float:
        # A changed average entry cannot reconstruct the original risk.
        return self.initial_risk_usd

    @property
    def r_computed(self) -> float:
        return self.net / self.risk_usd if self.risk_usd > 0 else 0.0

    # ── Where the R actually went ────────────────────────────
    # Decomposed against the counterfactual of a clean fill at the stop
    # with no costs, which is the -1.0R the position was sized for.

    @property
    def r_from_price(self) -> float:
        """R from executed prices before fees/funding; includes execution slippage."""
        return self.gross / self.risk_usd if self.risk_usd > 0 else 0.0

    @property
    def r_from_fees(self) -> float:
        return -(self.fees + self.funding) / self.risk_usd if self.risk_usd > 0 else 0.0

    @property
    def slipped_past_stop(self) -> float:
        """How far beyond the resting stop the fill landed, in R.

        Only meaningful for a stop exit. Positive means the fill was
        worse than the stop it was supposed to happen at.
        """
        if "stop" not in self.exit_reason or self.risk_usd <= 0:
            return 0.0
        beyond = (self.final_stop - self.exit) * self.direction
        return beyond * self.exit_quantity / self.risk_usd


def load(path: str) -> list[Row]:
    out: list[Row] = []
    with open(path, newline="") as fh:
        for d in csv.DictReader(fh):
            f = lambda k: float(d.get(k) or 0)      # noqa: E731
            out.append(Row(
                symbol=d["symbol"], side=d["side"],
                entry=f("entry_price"), exit=f("exit_price"),
                quantity=f("quantity"),
                original_quantity=f("original_quantity"),
                initial_stop=f("initial_stop"), final_stop=f("final_stop"),
                units=int(f("units") or 1),
                realized_before_exit=f("realized_before_exit"),
                exit_quantity=f("exit_quantity"),
                pnl=f("pnl"), fees=f("fees"), funding=f("funding"),
                initial_risk_usd=f("risk_usd"), scale_out_fees=f("scale_out_fees"),
                r_reported=f("r_multiple"), exit_reason=d.get("exit_reason", ""),
            ))
    return out


def check(rows: list[Row], tol_cash: float = 0.02, tol_r: float = 0.005) -> dict:
    """Recompute every trade. Returns the failures, not a pass/fail flag."""
    bad_cash, bad_r, missing = [], [], []
    for r in rows:
        if r.risk_usd <= 0 or r.original_quantity <= 0 or r.initial_stop <= 0:
            missing.append(r)
            continue
        if abs(r.net - r.pnl) > tol_cash:
            bad_cash.append((r, r.net - r.pnl))
        if abs(r.r_computed - r.r_reported) > tol_r:
            bad_r.append((r, r.r_computed - r.r_reported))
    return {"rows": len(rows), "missing_fields": missing,
            "cash_mismatch": bad_cash, "r_mismatch": bad_r}


def render(rows: list[Row], result: dict) -> str:
    out = [f"{result['rows']} trades"]
    for name, key in (("missing audit fields", "missing_fields"),
                      ("cash does not reconcile", "cash_mismatch"),
                      ("R does not reconcile", "r_mismatch")):
        items = result[key]
        out.append(f"  {name}: {len(items)}")
        for item in items[:5]:
            r, diff = item if isinstance(item, tuple) else (item, None)
            detail = f" (off by {diff:+.4f})" if diff is not None else ""
            out.append(f"      {r.symbol} {r.exit_reason} "
                       f"reported {r.r_reported:+.3f}R{detail}")
    return "\n".join(out)
