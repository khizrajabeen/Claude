"""Historical replay of the daily session.

This is not a second trading engine. It downloads history once, wraps it in
an exchange that only ever answers with bars at or before the simulated
clock, and then runs the *same* ``DailySession`` that trades live. Whatever
the replay measures is what the live path does, because it is the same code.

Two honest limitations, stated rather than hidden:

  * News is replayed as neutral. Public RSS feeds do not serve history, so
    a replay cannot reconstruct what the wire said on a past morning. The
    news tilt is disabled for replays and the report says so; sentiment is
    evaluated forward, in paper trading, not backwards.
  * Intrabar order is unknown. When one bar contains both the stop and the
    target, the stop is assumed to fill first.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.analysis.indicators import TIMEFRAME_SECONDS
from bot.daily.journal import Journal
from bot.daily.schedule import build_schedule, next_schedule
from bot.daily.session import DailySession, SimulatedClock
from bot.exchange import ExchangeClient
from bot.trading.broker import PaperBroker
from bot.utils.metrics import summarize_trades

logger = logging.getLogger("trading_bot")


class ReplayExchange:
    """Serves historical bars as though they were live, clipped to `now`.

    Every read goes through the clock, so no part of the session can see a
    bar that had not printed yet.
    """

    def __init__(self, frames: dict[str, dict[str, pd.DataFrame]], clock: SimulatedClock,
                 limits: dict | None = None, volumes: dict | None = None):
        self.frames = frames          # symbol -> timeframe -> DataFrame
        self.clock = clock
        self._limits = limits or {}
        self._volumes = volumes or {}
        # Price lookups dominate a replay — thousands per simulated day.
        # Caching the last visible close per instant, and slicing by
        # position rather than by boolean mask, took the replay from 8.4s
        # to under 2s for twenty days.
        self._price_cache: dict[tuple[str, object], float] = {}
        self._closes = {
            symbol: {tf: df["close"].to_numpy(dtype=float)
                     for tf, df in timeframes.items()}
            for symbol, timeframes in frames.items()
        }

    def resolve_symbol(self, symbol: str) -> str | None:
        return symbol if symbol in self.frames else None

    def _visible_count(self, symbol: str, timeframe: str) -> int:
        """How many bars have printed by the clock's current instant.

        `searchsorted` on the sorted index is O(log n); the boolean mask it
        replaces was O(n) and allocated a fresh frame on every call.
        """
        df = self.frames.get(symbol, {}).get(timeframe)
        if df is None or df.empty:
            return 0
        return int(df.index.searchsorted(self.clock.now(), side="right"))

    def _visible(self, symbol: str, timeframe: str) -> pd.DataFrame:
        count = self._visible_count(symbol, timeframe)
        if count == 0:
            return pd.DataFrame()
        return self.frames[symbol][timeframe].iloc[:count]

    def fetch_ohlcv(self, symbol: str, timeframe: str, limit: int = 500,
                    since: int | None = None) -> pd.DataFrame:
        return self._visible(symbol, timeframe).tail(limit)

    def fetch_ohlcv_paged(self, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
        """History is already local in a replay; paging is just a tail."""
        return self._visible(symbol, timeframe).tail(bars)

    def get_current_price(self, symbol: str) -> float:
        key = (symbol, self.clock.now())
        cached = self._price_cache.get(key)
        if cached is not None:
            return cached

        for timeframe in sorted(self.frames.get(symbol, {}),
                                key=lambda tf: TIMEFRAME_SECONDS.get(tf, 3600)):
            count = self._visible_count(symbol, timeframe)
            if count:
                price = float(self._closes[symbol][timeframe][count - 1])
                # One instant's prices are all that is ever needed at once;
                # keeping more would grow without bound over a long replay.
                if len(self._price_cache) > 4096:
                    self._price_cache.clear()
                self._price_cache[key] = price
                return price
        raise LookupError(f"no replay price for {symbol}")

    def market_limits(self, symbol: str) -> dict:
        return self._limits.get(symbol, {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0})

    def quote_volume_24h(self, symbol: str) -> float | None:
        """Rolling 24h quote volume from the bars themselves."""
        cached = self._volumes.get(symbol)
        if cached:
            return cached
        for timeframe in self.frames.get(symbol, {}):
            visible = self._visible(symbol, timeframe)
            if visible.empty:
                continue
            bars = max(1, int(86_400 / TIMEFRAME_SECONDS.get(timeframe, 3600)))
            window = visible.tail(bars)
            return float((window["close"] * window["volume"]).sum())
        return None

    def spread_bps(self, symbol: str) -> float | None:
        return None  # not reconstructible from OHLCV; the base slippage covers it

    def fetch_funding_rate(self, symbol: str) -> float | None:
        return None  # historical funding is not fetched in this build


class DailyReplay:
    """Replays the daily cycle over historical data."""

    def __init__(self, config: dict):
        # Replays get their own journal so they never overwrite live records.
        self.config = _replay_config(config)
        self.symbols = list(self.config["data"]["symbols"])
        self.timeframe = self.config["data"]["timeframe"]
        self.htf = self.config["data"].get("higher_timeframe", "4h")

    def run(self, days: int = 30, frames: dict | None = None) -> dict:
        """Replay `days` trading days and return the aggregate result."""
        logger.info("═" * 62)
        logger.info("  HISTORICAL REPLAY — %d day(s)", days)
        logger.info("  News is neutral in replay (RSS serves no history)")
        logger.info("═" * 62)

        frames = frames if frames is not None else self._download(days)
        if not frames:
            logger.error("No historical data available — nothing to replay.")
            return {"days": 0, "error": "no data"}

        start, end = _data_span(frames, self.timeframe)
        if start is None:
            return {"days": 0, "error": "no data"}

        # Begin only once every enabled strategy has the history it needs.
        # Warming up on filters.min_bars instead silently replays a stretch
        # where the deepest strategies are starved — they return nothing,
        # which is indistinguishable from having no opportunity, so the
        # measured result belongs to whichever half of the roster happened
        # to be awake.
        warmup_bars = self._warmup_bars()
        bar_seconds = TIMEFRAME_SECONDS.get(self.timeframe, 3600)
        first_tradeable = start + timedelta(seconds=warmup_bars * bar_seconds)
        if first_tradeable >= end:
            logger.error(
                "Only %s of data but the roster needs %d bars of warm-up — "
                "download more history or shorten the deepest lookback",
                end - start, warmup_bars,
            )
            return {"days": 0, "error": "not enough history for the roster's warm-up"}

        clock = SimulatedClock(first_tradeable)
        exchange = ReplayExchange(frames, clock, limits=self._limits())
        journal = Journal(self.config)
        # Each replay starts from nothing. Accumulating runs in one file
        # would silently mix results from different parameters.
        _clear_directory(journal.dir)
        journal = Journal(self.config)
        broker = PaperBroker(self.config)
        state = journal.load_state()
        state.cash = broker.cash
        state.initial_equity = broker.cash
        state.peak_equity = broker.cash

        session = DailySession(self.config, exchange, broker, journal, state, clock=clock)
        # Drive prices from the replay exchange rather than a live ticker.
        session._price_source = lambda symbols, now: _replay_prices(exchange, symbols)

        schedule = build_schedule(self.config, clock.now())
        results = []
        while schedule.close_at <= end and len(results) < days:
            clock._now = max(clock.now(), schedule.open_at)
            result = session.run_day(schedule)
            results.append(result)
            schedule = next_schedule(schedule, self.config)
            clock._now = schedule.open_at

        trades = journal.load_trades()
        summary = summarize_trades(
            trades, starting_equity=self.config["paper"]["initial_balance"]
        )
        day_rows = journal.load_days()

        equity_start = self.config["paper"]["initial_balance"]
        equity_end = day_rows[-1]["ending_equity"] if day_rows else equity_start

        report = {
            "days": len(results),
            "period": {"from": str(start.date()), "to": str(end.date())},
            "symbols": self.symbols,
            "starting_equity": equity_start,
            "ending_equity": equity_end,
            "total_return_pct": round((equity_end - equity_start) / equity_start * 100, 3)
            if equity_start else 0.0,
            "trades": summary,
            "daily": day_rows,
            "journal_dir": str(journal.dir),
            "caveats": [
                "news sentiment is neutral in replay (no historical RSS)",
                "stop assumed to fill before target when a bar spans both",
                "spread modelled by the flat slippage term only",
            ],
        }
        self._log_report(report)
        return report

    def _warmup_bars(self) -> int:
        """Bars of history the deepest enabled strategy needs."""
        from bot.strategies import build_strategies

        needed = [s.required_bars() for s in build_strategies(self.config)]
        return max([int(self.config["filters"].get("min_bars", 120))] + needed)

    # ── Data ──────────────────────────────────────────────────

    def _download(self, days: int) -> dict:
        """Fetch enough history for `days` sessions plus indicator warm-up."""
        exchange = ExchangeClient(self.config)
        bar_seconds = TIMEFRAME_SECONDS.get(self.timeframe, 3600)
        warmup = self._warmup_bars()
        needed = int(days * 86_400 / bar_seconds) + warmup + 50

        frames: dict[str, dict[str, pd.DataFrame]] = {}
        for symbol in self.symbols:
            resolved = exchange.resolve_symbol(symbol)
            if resolved is None:
                logger.warning("%s not listed on %s — excluded from replay",
                               symbol, self.config["exchange"]["name"])
                continue
            try:
                base = _paged_ohlcv(exchange, resolved, self.timeframe, needed)
                if base.empty or len(base) < warmup + 24:
                    logger.warning("%s: only %d bars — excluded", symbol, len(base))
                    continue
                htf_needed = max(120, int(needed * bar_seconds
                                          / TIMEFRAME_SECONDS.get(self.htf, 14_400)) + 60)
                htf = _paged_ohlcv(exchange, resolved, self.htf, htf_needed)
                frames[symbol] = {self.timeframe: base, self.htf: htf}
                logger.info("  %-10s %d bars %s, %d bars %s",
                            symbol, len(base), self.timeframe, len(htf), self.htf)
            except Exception as e:
                logger.warning("Download failed for %s: %s", symbol, e)
        self._exchange = exchange
        return frames

    def _limits(self) -> dict:
        exchange = getattr(self, "_exchange", None)
        if exchange is None:
            return {}
        out = {}
        for symbol in self.symbols:
            try:
                resolved = exchange.resolve_symbol(symbol)
                if resolved:
                    out[symbol] = exchange.market_limits(resolved)
            except Exception:
                pass
        return out

    def _log_report(self, report: dict) -> None:
        s = report["trades"]
        logger.info("═" * 62)
        logger.info("  REPLAY RESULT — %d days (%s → %s)",
                    report["days"], report["period"]["from"], report["period"]["to"])
        logger.info("═" * 62)
        logger.info("  Equity        : $%.2f → $%.2f (%+.2f%%)",
                    report["starting_equity"], report["ending_equity"],
                    report["total_return_pct"])
        if not s.get("total_trades"):
            logger.info("  No trades were taken.")
            logger.info("═" * 62)
            return
        logger.info("  Trades        : %d (%dW / %dL, %.1f%%)",
                    s["total_trades"], s["wins"], s["losses"], s["win_rate"])
        logger.info("  Expectancy    : %+.3fR per trade", s["expectancy_r"])
        logger.info("  Avg win/loss  : %+.2fR / %-.2fR", s["avg_win_r"], s["avg_loss_r"])
        logger.info("  Profit factor : %.2f", s["profit_factor"])
        logger.info("  Sharpe (daily): %.2f", s["sharpe_daily"])
        logger.info("  Max drawdown  : %.2f%%", s["max_drawdown_pct"])
        logger.info("  Costs         : fees $%.2f | funding $%+.2f",
                    s["total_fees"], s["total_funding"])
        logger.info("  Avg MAE/MFE   : %.2fR / %.2fR", s["avg_mae_r"], s["avg_mfe_r"])
        logger.info("  Exits         : %s", s["exit_reasons"])
        logger.info("─" * 62)
        for caveat in report["caveats"]:
            logger.info("  caveat: %s", caveat)
        logger.info("═" * 62)


# ── helpers ──────────────────────────────────────────────────

def _replay_config(config: dict) -> dict:
    """A replay's config: no news, and its own records directory.

    The replay directory hangs off whatever journal directory is
    configured, so a replay can never write over live records — and a test
    pointing the journal at a temporary path keeps its replay there too.
    """
    import copy
    from pathlib import Path

    cfg = copy.deepcopy(config)
    cfg.setdefault("news", {})["enabled"] = False

    journal = cfg.setdefault("journal", {})
    live_dir = Path(journal.get("dir", "state"))
    journal["dir"] = str(journal.get("replay_dir") or live_dir / "replay")

    # A replay has no reason to wait between management passes.
    cfg.setdefault("session", {})["entry_stagger_seconds"] = 0
    return cfg


def _paged_ohlcv(exchange: ExchangeClient, symbol: str, timeframe: str,
                 needed: int) -> pd.DataFrame:
    """Kept as a thin alias: paging now lives on the exchange client, where
    the live briefing needs it too."""
    return exchange.fetch_ohlcv_paged(symbol, timeframe, needed)


def _clear_directory(path) -> None:
    """Empty a replay's records directory, leaving the directory itself."""
    import shutil
    from pathlib import Path

    path = Path(path)
    if not path.exists():
        return
    for child in path.iterdir():
        shutil.rmtree(child) if child.is_dir() else child.unlink()


def _data_span(frames: dict, timeframe: str):
    starts, ends = [], []
    for tfs in frames.values():
        df = tfs.get(timeframe)
        if df is not None and not df.empty:
            starts.append(df.index[0])
            ends.append(df.index[-1])
    if not starts:
        return None, None
    return max(starts), min(ends)


def _replay_prices(exchange: ReplayExchange, symbols: list[str]) -> dict[str, float]:
    prices = {}
    for symbol in symbols:
        try:
            prices[symbol] = exchange.get_current_price(symbol)
        except LookupError:
            continue
    return prices
