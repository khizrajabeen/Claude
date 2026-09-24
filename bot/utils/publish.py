"""Publishing the bot's state as JSON for a dashboard to read.

The dashboard is a static page. It cannot open a socket to the trading
process, and it must not hold exchange credentials to fetch its own data.
So the bot writes what it knows to a handful of small JSON files and the
page reads those — the same records that drive the terminal reports, in
the same shapes.

Everything written here is derived from the journal. Nothing is computed
twice: if the terminal says the account is down 0.66% and the dashboard
says something else, one of them is lying, and the way to guarantee they
agree is for both to read one source.

No credentials, keys or account identifiers are written. The files are
published to a public page; treating them as public is the only safe
assumption.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("trading_bot")

SCHEMA_VERSION = 1

# Written with a trailing newline and sorted keys so a republish that
# changes nothing produces no diff — otherwise every export churns the
# repository and the page rebuilds for nothing.
_DUMP = {"indent": 2, "sort_keys": True, "default": str}


class Publisher:
    """Writes the dashboard's data files from the journal."""

    def __init__(self, config: dict, out_dir: str | Path | None = None):
        self.config = config
        default = Path(config.get("web", {}).get("data_dir", "web/data"))
        self.dir = Path(out_dir) if out_dir else default

    # ── Entry point ───────────────────────────────────────────

    def publish(self, journal=None, screen=None, broker=None) -> dict:
        """Write every file. Returns what was written, for the caller to log."""
        from bot.daily.journal import Journal

        journal = journal or Journal(self.config)
        self.dir.mkdir(parents=True, exist_ok=True)

        state = journal.load_state()
        days = journal.load_days()
        trades = journal.load_trades()

        written = {
            "portfolio.json": self._portfolio(state, days, trades),
            "positions.json": self._positions(state),
            "trades.json": self._trades(trades),
            "daily.json": self._daily(trades, days, state),
            "strategies.json": self._strategies(journal),
            "meta.json": self._meta(state, days),
        }
        if screen is not None:
            written["screen.json"] = self._screen(screen)

        # The live account, and last prices for anything the browser
        # cannot fetch itself. The dashboard is a public static page: it
        # has no server, so it cannot hold an API key, so it cannot ask
        # Alpaca anything. Whatever needs a key has to be written here,
        # on the machine that has one.
        account = self._account(broker)
        if account:
            written["account.json"] = account
        quotes = self._quotes(broker)
        if quotes:
            written["quotes.json"] = quotes

        for name, payload in written.items():
            self._write(name, payload)
        logger.info("Published %d file(s) to %s", len(written), self.dir)
        return written

    def _account(self, broker) -> dict | None:
        """The venue's own view of the account.

        Distinct from portfolio.json, which is the bot's ledger. Both are
        published because a disagreement between them is the single most
        useful thing this dashboard can show: it means something happened
        that the bot does not know about.
        """
        if broker is None or not hasattr(broker, "snapshot"):
            return None
        try:
            snap = broker.snapshot()
        except Exception as e:
            logger.warning("Could not snapshot the account: %s", e)
            return None
        snap["schema"] = SCHEMA_VERSION
        try:
            snap["drift"] = broker.reconcile()
        except Exception:
            snap["drift"] = []
        return snap

    def _quotes(self, broker) -> dict | None:
        """Last price for every instrument in the universe.

        Crypto the browser can fetch itself from a public venue. Equities
        it cannot — every feed worth using needs a key. These are stamped
        with the time they were taken so the page can say how old they
        are instead of implying they are live.
        """
        if broker is None:
            return None
        try:
            from bot.data.router import DataRouter
            from bot.markets import build_universe
            router = DataRouter(self.config)
            rows = {}
            for instrument in build_universe(self.config):
                price = router.price(instrument)
                if price:
                    rows[instrument.symbol] = {
                        "price": round(float(price), 8),
                        "asset_class": instrument.asset_class.value,
                        "venue": instrument.venue,
                    }
        except Exception as e:
            logger.warning("Could not collect quotes: %s", e)
            return None
        if not rows:
            return None
        return {"schema": SCHEMA_VERSION, "as_of": _now(), "quotes": rows}

    def _write(self, name: str, payload) -> None:
        path = self.dir / name
        text = json.dumps(payload, **_DUMP) + "\n"
        # Only rewrite when the content actually changed, so an unchanged
        # export leaves the working tree clean.
        if path.exists() and path.read_text() == text:
            return
        path.write_text(text)

    # ── The files ─────────────────────────────────────────────

    def _portfolio(self, state, days: list, trades: list) -> dict:
        """Headline numbers: the four tiles across the top of the page."""
        from bot.utils.metrics import summarize_trades

        start = float(state.initial_equity or
                      self.config.get("paper", {}).get("initial_balance", 0.0))
        latest = days[-1] if days else {}
        equity = _float(latest.get("ending_equity"), state.cash)
        summary = summarize_trades(trades, starting_equity=start or None)

        # Month to date, by the day records rather than by trade dates: an
        # open position's mark belongs to the day it was marked on.
        month = latest.get("day", "")[:7]
        month_days = [d for d in days if str(d.get("day", "")).startswith(month)]
        month_start = _float((month_days[0] if month_days else latest)
                             .get("starting_equity"), equity)

        return {
            "schema": SCHEMA_VERSION,
            "as_of": _now(),
            "currency": "USD",
            "equity": round(equity, 2),
            "starting_equity": round(start, 2),
            "cash": round(float(state.cash or 0.0), 2),
            "total_return_pct": _pct(equity, start),
            "daily_pnl": round(_float(latest.get("realized_pnl"))
                               + _float(latest.get("unrealized_pnl")), 2),
            "daily_return_pct": round(_float(latest.get("return_pct")), 3),
            "month_return_pct": _pct(equity, month_start),
            "peak_equity": round(float(state.peak_equity or 0.0), 2),
            "max_drawdown_pct": round(_float(summary.get("max_drawdown_pct")), 3),
            "open_positions": len(state.positions or []),
            "trades": int(summary.get("total_trades") or 0),
            "win_rate": round(_float(summary.get("win_rate")), 1),
            "expectancy_r": round(_float(summary.get("expectancy_r")), 3),
            "profit_factor": round(_float(summary.get("profit_factor")), 2),
            "sharpe_daily": round(_float(summary.get("sharpe_daily")), 2),
            "fees": round(_float(summary.get("total_fees")), 2),
            "halted_reason": state.halted_reason or "",
        }

    def _positions(self, state) -> dict:
        """What is open right now, and how it is doing."""
        rows = []
        for position in state.positions or []:
            data = position.to_dict() if hasattr(position, "to_dict") else dict(position)
            rows.append({
                "symbol": data.get("symbol"),
                "side": data.get("side"),
                "asset_class": data.get("asset_class"),
                "quantity": _round(data.get("quantity"), 8),
                "entry_price": _round(data.get("entry_price"), 8),
                "stop_price": _round(data.get("stop_price"), 8),
                "take_profit": _round(data.get("take_profit"), 8),
                "leverage": _round(data.get("leverage"), 2),
                "opened_at": data.get("opened_at"),
                "opened_on_day": data.get("opened_on_day"),
                "strategy": data.get("strategy"),
                "units": data.get("units", 1),
                "risk_usd": _round(data.get("risk_usd"), 2),
            })
        return {"schema": SCHEMA_VERSION, "as_of": _now(), "positions": rows}

    def _trades(self, trades: list, limit: int = 200) -> dict:
        """The most recent closed trades, newest first."""
        rows = []
        for trade in list(trades)[-limit:]:
            data = trade.to_dict() if hasattr(trade, "to_dict") else dict(trade)
            rows.append({
                "symbol": data.get("symbol"),
                "side": data.get("side"),
                "asset_class": data.get("asset_class"),
                "strategy": data.get("strategy"),
                "entry_price": _round(data.get("entry_price"), 8),
                "exit_price": _round(data.get("exit_price"), 8),
                "pnl": _round(data.get("pnl"), 2),
                "r_multiple": _round(data.get("r_multiple"), 3),
                "exit_reason": data.get("exit_reason"),
                "opened_on_day": data.get("opened_on_day"),
                "closed_on_day": data.get("closed_on_day"),
            })
        rows.reverse()
        return {"schema": SCHEMA_VERSION, "as_of": _now(), "trades": rows}

    def _daily(self, trades: list, days: list, state) -> dict:
        """The equity curve and the per-asset-class split, one source."""
        from bot.utils.pnl import period_report

        report = period_report(trades, days,
                               starting_equity=state.initial_equity or None)
        curve = [
            {"day": row["day"], "equity": row["equity"],
             "return_pct": row["return_pct"],
             "realized_pnl": row["realized_pnl"],
             "by_class": row["by_class"]}
            for row in report.get("daily", [])
        ]
        return {
            "schema": SCHEMA_VERSION,
            "as_of": _now(),
            "period": report.get("period", {}),
            "curve": curve,
            "by_asset_class": report.get("by_asset_class", {}),
            "by_symbol": report.get("by_symbol", {}),
            "winning_days": report.get("winning_days", 0),
            "losing_days": report.get("losing_days", 0),
            "flat_days": report.get("flat_days", 0),
        }

    def _strategies(self, journal) -> dict:
        try:
            stats = journal.strategy_stats()
        except Exception as e:
            logger.debug("No strategy stats: %s", e)
            stats = {}
        return {"schema": SCHEMA_VERSION, "as_of": _now(), "strategies": stats}

    def _screen(self, candidates) -> dict:
        rows = [c.to_dict() if hasattr(c, "to_dict") else dict(c)
                for c in candidates]
        return {"schema": SCHEMA_VERSION, "as_of": _now(), "markets": rows}

    def _meta(self, state, days: list) -> dict:
        """What this bot is configured to do — no keys, no identifiers."""
        from bot.markets import build_universe

        universe = build_universe(self.config)
        by_class: dict[str, list[str]] = {}
        for instrument in universe:
            by_class.setdefault(instrument.asset_class.value, []).append(
                instrument.symbol)

        session = self.config.get("session", {})
        risk = self.config.get("risk", {})
        return {
            "schema": SCHEMA_VERSION,
            "as_of": _now(),
            "mode": "paper" if self.config.get("paper", {}).get(
                "enabled", True) else "live",
            "strategies": list(self.config.get("strategies", {}).get("enabled", [])),
            "universe": by_class,
            "risk": {
                "risk_per_trade_pct": risk.get("risk_per_trade_pct"),
                "max_portfolio_heat_pct": risk.get("max_portfolio_heat_pct"),
                "max_open_positions": risk.get("max_open_positions"),
                "max_daily_loss_pct": risk.get("max_daily_loss_pct"),
            },
            "session": {
                "day_open": session.get("day_open"),
                "flatten_at": session.get("flatten_at"),
                "max_new_positions_per_day": session.get("max_new_positions_per_day"),
            },
            "days_recorded": len(days),
            "last_day": (days[-1].get("day") if days else None),
            "current_day": state.current_day,
        }


# ── helpers ──────────────────────────────────────────────────

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round(value, places: int):
    try:
        return round(float(value), places)
    except (TypeError, ValueError):
        return None


def _pct(now: float, then: float) -> float:
    return round((now - then) / then * 100, 3) if then else 0.0
