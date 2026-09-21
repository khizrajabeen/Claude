"""Per-strategy performance history.

The allocator needs to know how volatile each strategy's returns have been
and how much they move together. That has to come from realised results,
not from assumptions, so every closed trade is attributed to the strategy
that opened it and rolled up into a daily return series per strategy.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")


class StrategyTracker:
    """Rolling per-strategy returns, volatility and correlation."""

    def __init__(self, config: dict, journal=None):
        self.config = config
        self.journal = journal
        portfolio = config.get("portfolio", {})
        self.window_days = int(portfolio.get("lookback_days", 60))
        self.min_days = int(portfolio.get("min_history_days", 10))

    # ── Attribution ───────────────────────────────────────────

    def daily_returns(self, trades: list, equity: float) -> pd.DataFrame:
        """Daily PnL per strategy, as a fraction of equity.

        Returns an empty frame when there is nothing to measure, which the
        allocator reads as "weight everything equally" rather than guessing.
        """
        if not trades or equity <= 0:
            return pd.DataFrame()

        buckets: dict[tuple[str, str], float] = defaultdict(float)
        for trade in trades:
            strategy = (trade.strategy or "unattributed").split("+")[0]
            day = trade.closed_on_day or trade.closed_at.date().isoformat()
            buckets[(day, strategy)] += trade.pnl

        if not buckets:
            return pd.DataFrame()

        rows = [{"day": day, "strategy": strategy, "pnl": pnl}
                for (day, strategy), pnl in buckets.items()]
        frame = pd.DataFrame(rows)
        wide = frame.pivot_table(index="day", columns="strategy", values="pnl",
                                 aggfunc="sum").sort_index()
        # Days a strategy did not trade are flat, not missing.
        wide = wide.fillna(0.0) / equity
        return wide.tail(self.window_days)

    # ── Risk inputs ───────────────────────────────────────────

    def volatilities(self, returns: pd.DataFrame) -> dict[str, float]:
        """Daily return volatility per strategy."""
        if returns.empty:
            return {}
        out = {}
        for column in returns.columns:
            series = returns[column]
            if series.count() < self.min_days:
                continue
            vol = float(series.std())
            if np.isfinite(vol) and vol > 0:
                out[column] = vol
        return out

    def correlations(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Correlation between strategy return streams."""
        if returns.empty or returns.shape[1] < 2 or len(returns) < self.min_days:
            return pd.DataFrame()
        corr = returns.corr()
        return corr.fillna(0.0)

    def summary(self, returns: pd.DataFrame) -> dict:
        """Per-strategy stats, for the report and for the allocator's log."""
        if returns.empty:
            return {}
        out = {}
        for column in returns.columns:
            series = returns[column]
            traded = series[series != 0]
            vol = float(series.std())
            mean = float(series.mean())
            out[column] = {
                "days": int(series.count()),
                "active_days": int(traded.count()),
                "total_return_pct": round(float(series.sum()) * 100, 4),
                "daily_vol_pct": round(vol * 100, 4),
                "sharpe": round(mean / vol * np.sqrt(365), 3) if vol > 0 else 0.0,
                "hit_rate": round(float((traded > 0).mean()) * 100, 2)
                if traded.count() else 0.0,
            }
        return out
