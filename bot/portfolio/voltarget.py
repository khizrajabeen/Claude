"""Portfolio-level volatility targeting.

The single best-evidenced lever on drawdown. Scaling exposure by recent
realised volatility raises Sharpe modestly and cuts maximum drawdown
substantially, because left-tail events cluster in high-volatility periods
— exactly when a vol-targeted book is already de-levered. The published
effect on equities takes Sharpe from roughly 0.40 to about 0.50, and the
same pattern holds for crypto, with more positive skew and less time under
water.

Two refinements on the plain version:

  * **Asymmetric response.** Leverage is cut quickly when volatility spikes
    and restored slowly when it falls, so the book de-risks into a shock
    rather than after it.
  * **A drawdown throttle.** Separately from volatility, exposure is scaled
    down as the account falls from its peak, which bounds how far a bad
    stretch can run before the hard halt has to fire.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")


@dataclass
class ExposureDecision:
    """How hard to press, and why."""

    scale: float
    realized_vol: float
    target_vol: float
    drawdown_pct: float
    reasons: list

    def to_dict(self) -> dict:
        return {
            "scale": round(self.scale, 4),
            "realized_vol_pct": round(self.realized_vol * 100, 3),
            "target_vol_pct": round(self.target_vol * 100, 3),
            "drawdown_pct": round(self.drawdown_pct, 3),
            "reasons": self.reasons,
        }


class VolatilityTargeter:
    """Turns recent realised volatility into an exposure multiplier."""

    def __init__(self, config: dict):
        self.config = config
        vol = config.get("vol_target", {})
        self.enabled = bool(vol.get("enabled", True))
        self.target_annual = float(vol.get("target_annual_pct", 15.0)) / 100
        self.min_scale = float(vol.get("min_scale", 0.25))
        self.max_scale = float(vol.get("max_scale", 1.5))
        self.window_days = int(vol.get("window_days", 20))
        self.min_days = int(vol.get("min_days", 8))
        self.cut_speed = float(vol.get("cut_speed", 1.0))      # immediate de-risk
        self.restore_speed = float(vol.get("restore_speed", 0.25))
        self.drawdown_throttle = bool(vol.get("drawdown_throttle", True))
        self.throttle_start_pct = float(vol.get("throttle_start_pct", 4.0))
        self.throttle_floor = float(vol.get("throttle_floor", 0.35))
        self._previous_scale: float | None = None

    def decide(self, daily_returns: pd.Series | list, drawdown_pct: float = 0.0
               ) -> ExposureDecision:
        """Exposure multiplier for today."""
        reasons: list[str] = []

        if not self.enabled:
            return ExposureDecision(1.0, 0.0, self.target_annual, drawdown_pct,
                                    ["vol targeting disabled"])

        series = pd.Series(list(daily_returns), dtype=float).dropna()
        if len(series) < self.min_days:
            # Not enough history to measure anything. Start at target rather
            # than guessing, and say so.
            scale = 1.0
            reasons.append(f"only {len(series)} days of history — holding at 1.0x")
            realized = 0.0
        else:
            window = series.tail(self.window_days)
            daily_vol = float(window.std())
            realized = daily_vol * np.sqrt(365)
            if realized <= 1e-9:
                scale = self.max_scale
                reasons.append("realised vol ~0 — capped at max scale")
            else:
                scale = self.target_annual / realized
                reasons.append(
                    f"realised {realized * 100:.1f}% vs target {self.target_annual * 100:.1f}%"
                )

        scale = float(np.clip(scale, self.min_scale, self.max_scale))
        scale = self._smooth(scale, reasons)

        if self.drawdown_throttle and drawdown_pct > self.throttle_start_pct:
            # Linear from 1.0 at the throttle start down to the floor at the
            # hard halt level.
            halt = float(self.config.get("risk", {}).get("max_total_drawdown_pct", 15.0))
            span = max(1e-9, halt - self.throttle_start_pct)
            progress = min(1.0, (drawdown_pct - self.throttle_start_pct) / span)
            throttle = 1.0 - (1.0 - self.throttle_floor) * progress
            scale *= throttle
            reasons.append(f"drawdown {drawdown_pct:.1f}% throttles to {throttle:.2f}x")

        scale = float(np.clip(scale, self.min_scale, self.max_scale))
        return ExposureDecision(scale, realized, self.target_annual, drawdown_pct, reasons)

    def _smooth(self, target: float, reasons: list) -> float:
        """Cut fast, restore slow.

        Volatility spikes arrive faster than they decay, so a symmetric
        smoother is always late in the direction that costs money.
        """
        previous = self._previous_scale
        if previous is None:
            self._previous_scale = target
            return target

        speed = self.cut_speed if target < previous else self.restore_speed
        smoothed = previous + speed * (target - previous)
        if abs(smoothed - target) > 1e-6 and speed < 1.0:
            reasons.append(f"eased from {previous:.2f}x toward {target:.2f}x")
        self._previous_scale = smoothed
        return smoothed

    def reset(self) -> None:
        self._previous_scale = None
