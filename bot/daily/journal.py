"""Durable record keeping across trading days.

Everything the bot needs to resume tomorrow lives on disk:

    state/state.json              cash, open positions, streaks, cooldowns
    state/trades.csv              append-only ledger of closed round trips
    state/days.csv                one row per completed trading day
    state/briefings/<day>.json    the morning briefing and the day's plan

Writes are atomic (temp file + rename) so a crash mid-save cannot leave a
half-written state file behind.
"""

from __future__ import annotations

import csv
import json
import logging
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from bot.trading.models import Position, Trade

logger = logging.getLogger("trading_bot")

STATE_VERSION = 2

TRADE_COLUMNS = [
    "id", "symbol", "side", "entry_price", "exit_price", "quantity", "leverage",
    "opened_at", "closed_at", "holding_minutes", "pnl", "pnl_pct", "r_multiple",
    "fees", "funding", "slippage_cost", "risk_usd", "exit_reason",
    "opened_on_day", "closed_on_day", "strategy", "entry_reason", "mae_r", "mfe_r",
]

DAY_COLUMNS = [
    "day", "starting_equity", "ending_equity", "realized_pnl", "unrealized_pnl",
    "return_pct", "trades_opened", "trades_closed", "wins", "losses",
    "win_rate", "avg_r", "gross_profit", "gross_loss", "profit_factor",
    "fees", "funding", "max_drawdown_pct", "peak_equity", "positions_carried",
    "halted_reason", "news_bias", "regime",
]


@dataclass
class BotState:
    """Everything that survives a restart or a day boundary."""

    version: int = STATE_VERSION
    cash: float = 0.0
    initial_equity: float = 0.0
    peak_equity: float = 0.0
    positions: list[Position] = field(default_factory=list)
    last_completed_day: str | None = None
    current_day: str | None = None
    day_start_equity: float = 0.0
    trades_opened_today: int = 0
    trades_closed_today: int = 0
    realized_pnl_today: float = 0.0
    consecutive_losses: int = 0
    consecutive_wins: int = 0
    cooldown_until: str | None = None
    halted_reason: str | None = None
    trade_counter: int = 0
    symbol_stats: dict = field(default_factory=dict)
    updated_at: str | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["positions"] = [p.to_dict() for p in self.positions]
        d["updated_at"] = datetime.now(timezone.utc).isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "BotState":
        d = dict(d)
        d["positions"] = [Position.from_dict(p) for p in d.get("positions", [])]
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class DaySummary:
    """One completed trading day, as written to days.csv."""

    day: str
    starting_equity: float
    ending_equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    return_pct: float = 0.0
    trades_opened: int = 0
    trades_closed: int = 0
    wins: int = 0
    losses: int = 0
    win_rate: float = 0.0
    avg_r: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    fees: float = 0.0
    funding: float = 0.0
    max_drawdown_pct: float = 0.0
    peak_equity: float = 0.0
    positions_carried: int = 0
    halted_reason: str = ""
    news_bias: float = 0.0
    regime: str = ""

    def to_row(self) -> dict:
        d = asdict(self)
        return {c: d.get(c, "") for c in DAY_COLUMNS}


class JournalLocked(Exception):
    """Another process already holds this journal."""


class Journal:
    """Reads and writes the bot's persistent records."""

    def __init__(self, config: dict):
        self.config = config
        root = config.get("journal", {}).get("dir", "state")
        self.dir = Path(root)
        self.briefing_dir = self.dir / "briefings"
        self.state_path = self.dir / "state.json"
        self.trades_path = self.dir / "trades.csv"
        self.days_path = self.dir / "days.csv"
        self.lock_path = self.dir / "session.lock"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.briefing_dir.mkdir(parents=True, exist_ok=True)
        self._locked = False

    # ── Single-writer lock ────────────────────────────────────

    def acquire(self) -> "Journal":
        """Claim this journal for the current process.

        Two bots sharing one directory interleave their writes and each
        records the same day, which is how one trading day ends up as three
        rows. Taking the lock makes the second one fail loudly instead.
        """
        if self.lock_path.exists():
            try:
                pid = int(self.lock_path.read_text().split()[0])
            except (ValueError, IndexError, OSError):
                pid = None
            if pid is not None and _process_alive(pid):
                raise JournalLocked(
                    f"{self.dir} is in use by process {pid}. Stop it first, or "
                    f"point journal.dir somewhere else."
                )
            logger.warning("Clearing a stale lock from process %s", pid)

        self.lock_path.write_text(f"{os.getpid()} {datetime.now(timezone.utc).isoformat()}\n")
        self._locked = True
        return self

    def release(self) -> None:
        if self._locked:
            self.lock_path.unlink(missing_ok=True)
            self._locked = False

    def __enter__(self) -> "Journal":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()

    # ── State ─────────────────────────────────────────────────

    def load_state(self) -> BotState:
        """Load persisted state, or seed a fresh one from config."""
        initial = float(self.config.get("paper", {}).get("initial_balance", 10000.0))
        if not self.state_path.exists():
            logger.info("No prior state — starting fresh at $%.2f", initial)
            return BotState(cash=initial, initial_equity=initial, peak_equity=initial)

        try:
            with open(self.state_path) as f:
                raw = json.load(f)
            state = BotState.from_dict(raw)
        except (OSError, ValueError, TypeError) as e:
            logger.error("State file unreadable (%s) — starting fresh", e)
            return BotState(cash=initial, initial_equity=initial, peak_equity=initial)

        if state.version != STATE_VERSION:
            logger.warning(
                "State version %s != %s — fields may be defaulted",
                state.version, STATE_VERSION,
            )
        logger.info(
            "Resumed state: cash=$%.2f | %d open position(s) | last day %s",
            state.cash, len(state.positions), state.last_completed_day,
        )
        return state

    def save_state(self, state: BotState) -> None:
        _atomic_write_json(self.state_path, state.to_dict())

    # ── Ledgers ───────────────────────────────────────────────

    def append_trade(self, trade: Trade) -> None:
        _append_csv(self.trades_path, TRADE_COLUMNS, trade.to_dict())

    def append_day(self, summary: DaySummary) -> None:
        """Record a day, replacing any existing row for the same date.

        Re-running a day — after a crash, or because the bot was restarted —
        must correct that day's record, not add a second one. Two rows for
        one date make every downstream number wrong.
        """
        _upsert_csv(self.days_path, DAY_COLUMNS, summary.to_row(), key="day")
        logger.info(
            "Day %s recorded: %+.2f (%.2f%%) | %dW/%dL | equity $%.2f",
            summary.day, summary.realized_pnl, summary.return_pct,
            summary.wins, summary.losses, summary.ending_equity,
        )

    def load_trades(self, limit: int | None = None) -> list[Trade]:
        rows = _read_csv(self.trades_path)
        if limit is not None:
            rows = rows[-limit:]
        trades = []
        for row in rows:
            try:
                trades.append(Trade.from_dict(row))
            except (ValueError, TypeError) as e:
                logger.debug("Skipping malformed trade row: %s", e)
        return trades

    def load_days(self, limit: int | None = None) -> list[dict]:
        rows = _read_csv(self.days_path)
        if limit is not None:
            rows = rows[-limit:]
        out = []
        for row in rows:
            parsed = {}
            for k, v in row.items():
                if k in ("day", "halted_reason", "regime"):
                    parsed[k] = v
                else:
                    try:
                        parsed[k] = float(v) if v not in (None, "") else 0.0
                    except ValueError:
                        parsed[k] = v
            out.append(parsed)
        return out

    # ── Briefings ─────────────────────────────────────────────

    def save_briefing(self, day: date | str, payload: dict) -> Path:
        path = self.briefing_dir / f"{day}.json"
        _atomic_write_json(path, payload)
        return path

    def load_briefing(self, day: date | str) -> dict | None:
        path = self.briefing_dir / f"{day}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    # ── Derived stats (what tomorrow reads from yesterday) ────

    def rolling_stats(self, window: int = 50) -> dict:
        """Win rate / payoff / expectancy over the most recent trades.

        Feeds the Kelly cap and the adaptive risk multiplier, so the bot
        sizes down after a bad stretch and up after a good one instead of
        treating every day as its first.
        """
        trades = self.load_trades(limit=window)
        if not trades:
            return {
                "sample": 0, "win_rate": 0.0, "avg_win_r": 0.0, "avg_loss_r": 0.0,
                "expectancy_r": 0.0, "profit_factor": 0.0, "kelly": 0.0,
            }

        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]
        win_rate = len(wins) / len(trades)

        avg_win_r = sum(t.r_multiple for t in wins) / len(wins) if wins else 0.0
        avg_loss_r = abs(sum(t.r_multiple for t in losses) / len(losses)) if losses else 0.0
        expectancy_r = win_rate * avg_win_r - (1 - win_rate) * avg_loss_r

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0.0

        # Kelly on the R distribution: f = W - (1-W)/payoff.
        payoff = avg_win_r / avg_loss_r if avg_loss_r > 0 else 0.0
        kelly = win_rate - (1 - win_rate) / payoff if payoff > 0 else 0.0

        return {
            "sample": len(trades),
            "win_rate": round(win_rate, 4),
            "avg_win_r": round(avg_win_r, 4),
            "avg_loss_r": round(avg_loss_r, 4),
            "expectancy_r": round(expectancy_r, 4),
            "profit_factor": round(profit_factor, 4),
            "kelly": round(max(0.0, min(1.0, kelly)), 4),
        }

    def symbol_stats(self, window: int = 200) -> dict:
        """Per-symbol expectancy, used to drop persistently losing symbols."""
        out: dict[str, dict] = {}
        for t in self.load_trades(limit=window):
            s = out.setdefault(t.symbol, {"trades": 0, "wins": 0, "r_sum": 0.0, "pnl": 0.0})
            s["trades"] += 1
            s["wins"] += 1 if t.pnl > 0 else 0
            s["r_sum"] += t.r_multiple
            s["pnl"] += t.pnl
        for s in out.values():
            n = s["trades"]
            s["win_rate"] = round(s["wins"] / n, 4) if n else 0.0
            s["avg_r"] = round(s["r_sum"] / n, 4) if n else 0.0
        return out

    def strategy_stats(self, window: int = 500) -> dict:
        """Per-strategy record: trade count, expectancy in R, hit rate.

        This is what the autopilot judges a strategy on, so it deliberately
        reports the sample size alongside the number — an expectancy over
        six trades is not evidence of anything.
        """
        out: dict[str, dict] = {}
        for trade in self.load_trades(limit=window):
            name = (trade.strategy or "unattributed").split("+")[0]
            record = out.setdefault(name, {"trades": 0, "wins": 0, "r_sum": 0.0,
                                           "pnl": 0.0})
            record["trades"] += 1
            record["wins"] += 1 if trade.pnl > 0 else 0
            record["r_sum"] += trade.r_multiple
            record["pnl"] += trade.pnl

        for record in out.values():
            count = record["trades"]
            record["avg_r"] = round(record["r_sum"] / count, 4) if count else 0.0
            record["win_rate"] = round(record["wins"] / count * 100, 2) if count else 0.0
            record["pnl"] = round(record["pnl"], 2)
        return out

    def last_day(self) -> dict | None:
        days = self.load_days(limit=1)
        return days[-1] if days else None


# ── file helpers ─────────────────────────────────────────────

def _process_alive(pid: int) -> bool:
    """Whether a pid is still running, without signalling it."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by someone else
    return True

def _atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _upsert_csv(path: Path, columns: list[str], row: dict, key: str) -> None:
    """Append a row, or replace the existing row with the same key."""
    existing = _read_csv(path)
    replaced = False
    for i, current in enumerate(existing):
        if current.get(key) == str(row.get(key)):
            existing[i] = {c: row.get(c, "") for c in columns}
            replaced = True
            break
    if not replaced:
        _append_csv(path, columns, row)
        return

    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
            writer.writeheader()
            for current in existing:
                writer.writerow({c: current.get(c, "") for c in columns})
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _append_csv(path: Path, columns: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))
