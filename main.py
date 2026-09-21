#!/usr/bin/env python3
"""Daily crypto trading bot — entry point.

The bot runs on a trading-day cycle. Each day it reads the news and the
tape, builds a plan, works that plan inside a bounded entry window, manages
the book against ATR stops for the rest of the session, closes out, and
writes the day to disk so the next one starts from a real record.

    python main.py run                  # paper-trade the daily cycle
    python main.py run --days 1         # one day, then stop
    python main.py day                  # run today's session once and exit
    python main.py briefing             # morning read, no orders
    python main.py plan                 # briefing + the plan it would trade
    python main.py replay --days 30     # replay the cycle over history
    python main.py report               # records so far
    python main.py news                 # current sentiment
    python main.py live                 # real money (asks first)
"""

from __future__ import annotations

import argparse
import json
import signal as signal_module
import sys
from datetime import datetime, timedelta, timezone


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Daily crypto trading bot — news-aware, ATR-risked, journaled",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py briefing              What the market and the news look like
  python main.py plan                  ... and what the bot would trade
  python main.py day                   Run one full trading day (paper)
  python main.py run                   Run day after day (paper)
  python main.py replay --days 60      Replay the same logic over history
  python main.py report                Equity curve, trades, day-by-day
  python main.py live --yes            Real money. Read the code first.
        """,
    )
    parser.add_argument(
        "mode",
        choices=["run", "day", "briefing", "plan", "replay", "report", "news",
                 "live", "train", "backtest"],
        help="What to do",
    )
    parser.add_argument("--config", default="config.yaml", help="Config file")
    parser.add_argument("--symbol", help="Restrict the universe to one pair")
    parser.add_argument("--exchange", help="Override exchange name")
    parser.add_argument("--days", type=int, help="Number of days to run or replay")
    parser.add_argument("--fast", action="store_true",
                        help="Run the day on a simulated clock (no waiting)")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument("--yes", action="store_true", help="Skip the live-trading prompt")
    parser.add_argument("--reset", action="store_true",
                        help="Start from a clean journal (archives the old one)")
    return parser.parse_args(argv)


def build_context(config, args):
    """Wire up the pieces every trading mode needs."""
    from bot.daily.journal import Journal
    from bot.daily.session import Clock, DailySession, SimulatedClock
    from bot.exchange import ExchangeClient
    from bot.trading.broker import PaperBroker

    exchange = ExchangeClient(config)
    journal = Journal(config)
    state = journal.load_state()
    broker = PaperBroker(
        config, cash=state.cash, positions=state.positions,
        trade_counter=state.trade_counter,
    )
    clock = SimulatedClock(datetime.now(timezone.utc)) if args.fast else Clock()
    session = DailySession(config, exchange, broker, journal, state, clock=clock)
    return exchange, journal, state, broker, session


# ── Modes ────────────────────────────────────────────────────

def mode_briefing(config, logger, args):
    from bot.daily.schedule import build_schedule
    exchange, journal, state, broker, session = build_context(config, args)
    schedule = build_schedule(config, session.clock.now())
    briefing, _ = session.briefing_builder.build(
        symbols=session.symbols,
        day=str(schedule.day),
        equity=broker.equity(session._latest_prices(session.symbols, session.clock.now())),
        cash=broker.cash,
        carried_positions=broker.positions,
        rolling_stats=journal.rolling_stats(),
        yesterday=journal.last_day(),
        now=session.clock.now(),
    )
    if args.json:
        print(json.dumps(briefing.to_dict(), indent=2, default=str))
    return briefing


def mode_plan(config, logger, args):
    from bot.daily.schedule import build_schedule
    exchange, journal, state, broker, session = build_context(config, args)
    schedule = build_schedule(config, session.clock.now())
    session._begin_day(str(schedule.day))
    briefing, plan = session.open_day(schedule)
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, default=str))
    else:
        logger.info("Plan saved to %s", journal.briefing_dir / f"{schedule.day}.json")
    return plan


def mode_day(config, logger, args):
    exchange, journal, state, broker, session = build_context(config, args)
    _install_shutdown(session, logger)
    result = session.run_day()
    if args.json:
        print(json.dumps(result.summary.to_row(), indent=2, default=str))
    return result


def mode_run(config, logger, args):
    exchange, journal, state, broker, session = build_context(config, args)
    _install_shutdown(session, logger)
    results = session.run_forever(max_days=args.days)
    logger.info("Ran %d session(s)", len(results))
    return results


def mode_replay(config, logger, args):
    """Replay the daily cycle over historical bars."""
    from bot.utils.backtester import DailyReplay
    days = args.days or 30
    replay = DailyReplay(config)
    results = replay.run(days=days)
    if args.json:
        print(json.dumps(results, indent=2, default=str))
    return results


def mode_report(config, logger, args):
    from bot.daily.journal import Journal
    from bot.utils.metrics import summarize_trades

    journal = Journal(config)
    days = journal.load_days()
    trades = journal.load_trades()
    state = journal.load_state()

    if args.json:
        print(json.dumps({
            "state": state.to_dict(),
            "days": days,
            "rolling": journal.rolling_stats(),
            "by_symbol": journal.symbol_stats(),
            "summary": summarize_trades(trades, starting_equity=state.initial_equity),
        }, indent=2, default=str))
        return

    logger.info("═" * 62)
    logger.info("  TRADING RECORD")
    logger.info("═" * 62)
    if not days and not trades:
        logger.info("  Nothing recorded yet — run 'python main.py day' first.")
        return

    logger.info("  Cash          : $%.2f", state.cash)
    logger.info("  Open positions: %d", len(state.positions))
    logger.info("  Peak equity   : $%.2f", state.peak_equity)
    logger.info("  Days recorded : %d", len(days))

    stats = summarize_trades(trades, starting_equity=state.initial_equity)
    for key, label in [
        ("total_trades", "Trades"), ("win_rate", "Win rate %"),
        ("expectancy_r", "Expectancy R"), ("profit_factor", "Profit factor"),
        ("avg_win_r", "Avg win R"), ("avg_loss_r", "Avg loss R"),
        ("total_pnl", "Total PnL $"), ("total_fees", "Fees $"),
        ("max_drawdown_pct", "Max drawdown %"), ("sharpe_daily", "Sharpe (daily)"),
    ]:
        if key in stats:
            logger.info("  %-14s: %s", label, _fmt(stats[key]))

    if days:
        logger.info("─" * 62)
        logger.info("  %-12s %12s %10s %8s %7s", "Day", "Equity", "Return%", "Trades", "W/L")
        for d in days[-14:]:
            logger.info(
                "  %-12s %12.2f %9.2f%% %8d %3d/%-3d",
                d["day"], d["ending_equity"], d["return_pct"],
                int(d["trades_closed"]), int(d["wins"]), int(d["losses"]),
            )
    logger.info("═" * 62)


def mode_news(config, logger, args):
    from bot.analysis.news_sentiment import NewsSentimentAnalyzer
    news = NewsSentimentAnalyzer(config)
    news.refresh(force=True)

    symbols = [args.symbol] if args.symbol else config["data"]["symbols"]
    tone = news.market_bias()

    if args.json:
        print(json.dumps({
            "market": tone,
            "symbols": {s: news.sentiment_for(s, 24) for s in symbols},
        }, indent=2, default=str))
        return

    logger.info("═" * 62)
    logger.info("  CRYPTO NEWS — market tone %s (%.3f from %d stories)",
                tone["tone"], tone["score"], tone["articles"])
    logger.info("═" * 62)
    for symbol in symbols:
        s = news.sentiment_for(symbol, 24)
        bar_len = 20
        filled = int((s["score"] + 1) / 2 * bar_len)
        bar = "█" * filled + "░" * (bar_len - filled)
        logger.info("  %-10s [%s] %+.3f | %d articles (%dB/%dR) | tilt %+.2f",
                    symbol, bar, s["score"], s["articles"], s["bullish"],
                    s["bearish"], news.tilt_for(symbol))
        for h in s["headlines"][:3]:
            logger.info("      [%+.2f] %s", h["score"], h["title"][:72])
    logger.info("═" * 62)


def mode_live(config, logger, args):
    if not config["exchange"].get("api_key"):
        logger.error("No API key. Set EXCHANGE_API_KEY in .env before trading live.")
        sys.exit(1)
    if not args.yes:
        logger.warning("=" * 62)
        logger.warning("  LIVE TRADING — this places real orders with real money.")
        logger.warning("  Re-run with --yes once you have read the code and the config.")
        logger.warning("=" * 62)
        sys.exit(1)
    logger.error(
        "Live execution is not wired up in this build. The daily session runs "
        "against the paper broker only; connecting it to real order placement "
        "is deliberately a separate, reviewed change."
    )
    sys.exit(2)


def mode_train(config, logger, args):
    from bot.ml.trainer import TrainingPipeline
    pipeline = TrainingPipeline(config)
    return pipeline.run(days=args.days)


def mode_backtest(config, logger, args):
    return mode_replay(config, logger, args)


# ── Helpers ──────────────────────────────────────────────────

def _fmt(value):
    if isinstance(value, float):
        return f"{value:,.3f}"
    return str(value)


def _install_shutdown(session, logger):
    def shutdown(signum, frame):
        logger.info("Shutdown requested — finishing the current pass...")
        session.stop()

    signal_module.signal(signal_module.SIGINT, shutdown)
    signal_module.signal(signal_module.SIGTERM, shutdown)


def print_banner(logger, config, mode):
    session = config.get("session", {})
    risk = config.get("risk", {})
    logger.info("═" * 62)
    logger.info("  DAILY CRYPTO TRADING BOT")
    logger.info("═" * 62)
    logger.info("  Mode       : %s", mode.upper())
    logger.info("  Exchange   : %s (%s)", config["exchange"]["name"],
                config["exchange"].get("market_type", "spot"))
    logger.info("  Universe   : %s", ", ".join(config["data"]["symbols"]))
    logger.info("  Timeframe  : %s (trend filter %s)",
                config["data"]["timeframe"], config["data"].get("higher_timeframe"))
    logger.info("  Day        : open %s | entries +%sm | flat %s",
                session.get("day_open"), session.get("entry_window_minutes"),
                session.get("flatten_at"))
    logger.info("  Risk       : %.2f%%/trade | heat %.1f%% | daily stop %.1f%%",
                float(risk.get("risk_per_trade_pct", 0)),
                float(risk.get("max_portfolio_heat_pct", 0)),
                float(risk.get("max_daily_loss_pct", 0)))
    logger.info("  Stops      : %.1fxATR | target %.1fR | trail %.1fxATR",
                float(config["stops"].get("atr_stop_mult", 0)),
                float(config["stops"].get("target_r_multiple", 0)),
                float(config["stops"].get("trail_atr_mult", 0)))
    logger.info("  News       : %s", "on" if config.get("news", {}).get("enabled") else "off")
    logger.info("═" * 62)


def _archive_journal(config, logger):
    import shutil
    from pathlib import Path
    root = Path(config.get("journal", {}).get("dir", "state"))
    if root.exists() and any(root.iterdir()):
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        dest = root.parent / f"{root.name}.{stamp}"
        shutil.move(str(root), str(dest))
        logger.info("Archived previous records to %s", dest)


def main(argv=None):
    args = parse_args(argv)

    from bot.config_loader import load_config
    from bot.logger import setup_logger

    config = load_config(args.config)
    if args.symbol:
        config["data"]["symbols"] = [args.symbol]
    if args.exchange:
        config["exchange"]["name"] = args.exchange

    logger = setup_logger(config)
    if args.reset:
        _archive_journal(config, logger)
    print_banner(logger, config, args.mode)

    modes = {
        "run": mode_run,
        "day": mode_day,
        "briefing": mode_briefing,
        "plan": mode_plan,
        "replay": mode_replay,
        "report": mode_report,
        "news": mode_news,
        "live": mode_live,
        "train": mode_train,
        "backtest": mode_backtest,
    }
    return modes[args.mode](config, logger, args)


if __name__ == "__main__":
    main()
