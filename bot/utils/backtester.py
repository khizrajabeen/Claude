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
  * Equities replay on daily bars, because that is the resolution the free
    feed serves. A stop on a stock is therefore checked once per session,
    not continuously — so equity stop-outs in a replay are optimistic on
    a gap and pessimistic on a wick. Crypto replays at its real ladder
    resolution and has no such gap.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

from bot.analysis.indicators import TIMEFRAME_SECONDS
from bot.daily.journal import Journal
from bot.daily.schedule import build_schedule, next_schedule
from bot.daily.session import DailySession, SimulatedClock
from bot.data import DataRouter
from bot.markets import build_universe
from bot.markets.timeframes import ladder_for
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

    def market_limits(self, instrument) -> dict:
        symbol = getattr(instrument, "symbol", instrument)
        return self._limits.get(
            symbol, {"min_qty": 0.0, "qty_step": 0.0, "min_notional": 0.0}
        )

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

    def spread_bps(self, instrument) -> float | None:
        return None  # not reconstructible from OHLCV; the base slippage covers it

    def fetch_funding_rate(self, symbol: str) -> float | None:
        return None  # historical funding is not fetched in this build

    # ── Router interface ──────────────────────────────────────
    # The session and briefing talk to a DataRouter, never to an exchange.
    # These methods are the same reads keyed by Instrument, so a replay is
    # a drop-in for the live router and nothing downstream knows which it
    # is holding.

    def provider_for(self, instrument):
        return self if instrument.symbol in self.frames else None

    def bars(self, instrument, timeframe: str | None = None,
             limit: int = 500) -> pd.DataFrame:
        tf = timeframe or instrument.timeframe
        available = self.frames.get(instrument.symbol, {})
        if tf not in available:
            # An instrument whose ladder does not carry this frame falls
            # back to its coarsest available one rather than going blind.
            if not available:
                return pd.DataFrame()
            tf = max(available, key=lambda f: TIMEFRAME_SECONDS.get(f, 3600))
        return self._visible(instrument.symbol, tf).tail(limit)

    def price(self, instrument) -> float | None:
        try:
            return self.get_current_price(instrument.symbol)
        except LookupError:
            return None

    def prices(self, instruments) -> dict[str, float]:
        out = {}
        for instrument in instruments:
            price = self.price(instrument)
            if price is not None:
                out[instrument.symbol] = price
        return out

    def quote_volume(self, instrument) -> float | None:
        return self.quote_volume_24h(instrument.symbol)

    def order_book(self, instrument, depth: int = 20) -> dict | None:
        return None  # OHLCV carries no book; imbalance is simply unavailable

    def funding_rate(self, instrument) -> float | None:
        return None


class DailyReplay:
    """Replays the daily cycle over historical data."""

    def __init__(self, config: dict, universe: list | None = None):
        # Replays get their own journal so they never overwrite live records.
        self.config = _replay_config(config)
        self.universe = universe if universe is not None else build_universe(self.config)
        self.by_symbol = {i.symbol: i for i in self.universe}
        self.symbols = [i.symbol for i in self.universe]
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

        # Only replay instruments history actually arrived for. Leaving a
        # dead symbol in the universe makes the report count days on which
        # it "found nothing", which is not the same as having looked.
        universe = [i for i in self.universe if i.symbol in frames]
        if not universe:
            universe = [
                self.by_symbol[s] for s in frames if s in self.by_symbol
            ] or list(self.universe)
        by_class: dict[str, list[str]] = {}
        for instrument in universe:
            by_class.setdefault(instrument.asset_class.value, []).append(instrument.symbol)
        logger.info("  Universe      : %s",
                    " | ".join(f"{k} x{len(v)}" for k, v in sorted(by_class.items())))

        start, end = _data_span(frames)
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
        open_instruments = list(universe)
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

        session = DailySession(self.config, exchange, broker, journal, state,
                               clock=clock, universe=universe)
        # Drive prices from the replay exchange rather than a live ticker,
        # and honour each calendar: a stock has no price at 03:00 UTC, so
        # quoting one there would let the replay manage a position through
        # hours it could never have traded in.
        session._price_source = lambda symbols, now: _replay_prices(
            exchange, symbols, self.by_symbol, now
        )

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
            "symbols": [i.symbol for i in universe],
            "universe": by_class,
            "starting_equity": equity_start,
            "ending_equity": equity_end,
            "total_return_pct": round((equity_end - equity_start) / equity_start * 100, 3)
            if equity_start else 0.0,
            "trades": summary,
            "by_asset_class": attribute_by_class(trades),
            "daily": day_rows,
            "journal_dir": str(journal.dir),
            "caveats": [
                "news sentiment is neutral in replay (no historical RSS)",
                "stop assumed to fill before target when a bar spans both",
                "spread modelled by the flat slippage term only",
                "equity stops are checked on daily bars (the feed's resolution)",
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
        """Fetch enough history for `days` sessions plus indicator warm-up.

        Every instrument gets its whole ladder, because the adaptive chooser
        may land on any rung of it on any day. Equities need far fewer bars
        for the same span than a 15-minute crypto frame does, so the count
        is computed per timeframe rather than once for the universe.
        """
        router = DataRouter(self.config)
        warmup = self._warmup_bars()

        frames: dict[str, dict[str, pd.DataFrame]] = {}
        for instrument in self.universe:
            ladder = ladder_for(instrument)
            got: dict[str, pd.DataFrame] = {}
            for timeframe in ladder:
                seconds = TIMEFRAME_SECONDS.get(timeframe, 3600)
                # An equity session is 6.5h of a 24h day, so a calendar
                # span of N days holds only ~N daily bars, not N*3.7.
                spans_per_day = (1.0 if seconds >= 86_400
                                 else 86_400 / seconds)
                if not instrument.asset_class.is_crypto and seconds >= 86_400:
                    # Weekends and holidays: ~252 sessions a year.
                    spans_per_day = 252 / 365
                needed = int(days * spans_per_day) + warmup + 50
                try:
                    df = router.bars(instrument, timeframe, needed)
                except Exception as e:
                    logger.warning("Download failed for %s %s: %s",
                                   instrument.symbol, timeframe, e)
                    continue
                if df is not None and not df.empty:
                    got[timeframe] = df

            base = got.get(instrument.timeframe) or next(iter(got.values()), None)
            if base is None or len(base) < 60:
                logger.warning("%s: %d bars — excluded from replay",
                               instrument.symbol, 0 if base is None else len(base))
                continue
            frames[instrument.symbol] = got
            logger.info("  %-16s %-11s %s",
                        instrument.symbol, instrument.asset_class.value,
                        ", ".join(f"{len(df)}x{tf}" for tf, df in got.items()))

        self._router = router
        return frames

    def _limits(self) -> dict:
        router = getattr(self, "_router", None)
        if router is None:
            return {}
        out = {}
        for instrument in self.universe:
            try:
                out[instrument.symbol] = router.market_limits(instrument)
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


def _paged_ohlcv(exchange, symbol: str, timeframe: str,
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


def _data_span(frames: dict, timeframe: str | None = None):
    """The window every instrument can be replayed over.

    The start is the *latest* first bar and the end the *earliest* last bar,
    so no instrument is asked for a day it has no history for. With mixed
    asset classes the span is taken over each instrument's coarsest frame:
    a stock's daily series and a perp's 15-minute series describe the same
    calendar days, and comparing their raw index bounds would hand the
    replay a start before the equity feed begins.
    """
    starts, ends = [], []
    for tfs in frames.values():
        candidates = [tfs[timeframe]] if timeframe and timeframe in tfs \
            else list(tfs.values())
        firsts = [df.index[0] for df in candidates if df is not None and not df.empty]
        lasts = [df.index[-1] for df in candidates if df is not None and not df.empty]
        if not firsts:
            continue
        starts.append(min(firsts))
        ends.append(max(lasts))
    if not starts:
        return None, None
    return max(starts), min(ends)


def attribute_by_class(trades: list[dict]) -> dict:
    """Per-asset-class P&L, so the classes can be compared on one page.

    This is the answer to "did the stocks or the perps make the money?",
    which a single blended equity curve cannot give.
    """
    from collections import defaultdict

    buckets: dict[str, list] = defaultdict(list)
    for trade in trades:
        buckets[str(_field(trade, "asset_class") or "unknown")].append(trade)

    out = {}
    for name, rows in sorted(buckets.items()):
        pnl = sum(_number(r, "pnl") for r in rows)
        r_multiples = [_number(r, "r_multiple") for r in rows
                       if _field(r, "r_multiple") not in (None, "")]
        wins = sum(1 for r in rows if _number(r, "pnl") > 0)
        out[name] = {
            "trades": len(rows),
            "pnl": round(pnl, 2),
            "wins": wins,
            "win_rate": round(wins / len(rows) * 100, 1) if rows else 0.0,
            "expectancy_r": round(sum(r_multiples) / len(r_multiples), 3)
            if r_multiples else 0.0,
            "fees": round(sum(_number(r, "fees") for r in rows), 2),
            "funding": round(sum(_number(r, "funding") for r in rows), 2),
            "symbols": sorted({str(_field(r, "symbol")) for r in rows}),
        }
    return out


def _field(trade, name: str):
    """One accessor for both shapes a trade arrives in.

    The journal hands back ``Trade`` objects; a CSV read or a JSON report
    hands back plain dicts. Attribution is asked for from both paths.
    """
    if isinstance(trade, dict):
        return trade.get(name)
    return getattr(trade, name, None)


def _number(trade, name: str) -> float:
    try:
        return float(_field(trade, name) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _replay_prices(exchange: ReplayExchange, symbols: list[str],
                   by_symbol: dict | None = None,
                   now: datetime | None = None) -> dict[str, float]:
    """Prices for every symbol whose market is open at `now`.

    Skipping the closed ones is what keeps an equity position from being
    stopped out overnight at a price no one could have traded at.
    """
    prices = {}
    for symbol in symbols:
        instrument = (by_symbol or {}).get(symbol)
        if instrument is not None and now is not None and not instrument.is_open(now):
            continue
        try:
            prices[symbol] = exchange.get_current_price(symbol)
        except LookupError:
            continue
    return prices
