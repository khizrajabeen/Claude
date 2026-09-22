"""Profit and loss, split by asset class and by day.

A single blended equity curve cannot answer the question this bot exists
to answer: did the perps or the stocks make the money? So the record is
cut two ways — one row per trading day, and one block per asset class
over the whole window.

Everything here is derived from the trade records rather than kept as
extra state. `trades.csv` already carries the asset class and the day each
position closed, so there is nothing to migrate and nothing that can drift
out of sync with the trades it describes. It also means these tables can
be run over a journal written before this module existed.

Two conventions worth stating, because they are choices rather than
facts:

  * A day is credited with the trades that *closed* on it. A position
    opened Monday and closed Thursday is Thursday's P&L, not a quarter of
    each. That matches how realised P&L is booked everywhere else in the
    bot and how an exchange statement reads.
  * Significance is reported alongside every expectancy, because a class
    with four trades and a fine average has told you nothing. The t-stat
    is on the mean R per trade, and it is compared against the critical
    value for the sample actually in hand rather than a flat 1.96. At
    four trades that threshold is 3.18, not 2 — using 2 everywhere calls
    a four-trade run significant, which is exactly the error the column
    exists to prevent.
"""

from __future__ import annotations

import math
from collections import defaultdict

CLASS_ORDER = ["crypto_perp", "crypto_spot", "equity", "etf", "futures", "unknown"]

# Two-sided 95% critical values of Student's t by degrees of freedom. The
# normal approximation (1.96) is only honest once the sample is large;
# below about thirty trades it understates how easily noise clears the
# bar. Indexed by df = n - 1.
T_CRITICAL_95 = {
    1: 12.71, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160,
    14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093,
    20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042, 40: 2.021,
    60: 2.000, 120: 1.980,
}


def t_critical(n: int) -> float:
    """The |t| a sample of `n` trades must clear to be called real."""
    df = max(1, n - 1)
    if df in T_CRITICAL_95:
        return T_CRITICAL_95[df]
    for bound in sorted(T_CRITICAL_95):
        if df < bound:
            return T_CRITICAL_95[bound]
    return 1.96

# Plain names for the report. "crypto_perp" is what the code calls it;
# "futures" is what the user asked to compare.
CLASS_LABEL = {
    "crypto_perp": "crypto futures",
    "crypto_spot": "crypto spot",
    "equity": "stocks",
    "etf": "ETFs",
    "futures": "futures",
    "unknown": "unclassified",
}


def field(trade, name: str, default=None):
    """One accessor for both shapes a trade arrives in.

    The journal hands back `Trade` objects; a CSV read hands back dicts.
    """
    value = trade.get(name, default) if isinstance(trade, dict) \
        else getattr(trade, name, default)
    return default if value in (None, "") else value


def number(trade, name: str) -> float:
    try:
        return float(field(trade, name, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def class_of(trade) -> str:
    return str(field(trade, "asset_class", "unknown") or "unknown")


def _order(names) -> list[str]:
    known = [c for c in CLASS_ORDER if c in names]
    return known + sorted(n for n in names if n not in CLASS_ORDER)


# ── Per-class aggregates ─────────────────────────────────────

def summarize_class(rows: list) -> dict:
    """Everything one asset class did, with its own error bars."""
    if not rows:
        return {"trades": 0, "pnl": 0.0}

    pnl = sum(number(r, "pnl") for r in rows)
    fees = sum(number(r, "fees") for r in rows)
    funding = sum(number(r, "funding") for r in rows)
    r_multiples = [number(r, "r_multiple") for r in rows
                   if field(r, "r_multiple") is not None]
    wins = [r for r in rows if number(r, "pnl") > 0]
    losses = [r for r in rows if number(r, "pnl") < 0]

    gross_win = sum(number(r, "pnl") for r in wins)
    gross_loss = abs(sum(number(r, "pnl") for r in losses))

    expectancy = sum(r_multiples) / len(r_multiples) if r_multiples else 0.0
    se, t_stat, needed = _significance(r_multiples)
    critical = t_critical(len(r_multiples))

    return {
        "trades": len(rows),
        "pnl": round(pnl, 2),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(rows) * 100, 1),
        "expectancy_r": round(expectancy, 3),
        "se_r": round(se, 3),
        "t_stat": round(t_stat, 2),
        "t_critical": round(critical, 3),
        "significant": bool(r_multiples) and abs(t_stat) >= critical,
        "trades_for_significance": needed,
        "avg_win_r": round(sum(number(r, "r_multiple") for r in wins) / len(wins), 3)
        if wins else 0.0,
        "avg_loss_r": round(sum(number(r, "r_multiple") for r in losses) / len(losses), 3)
        if losses else 0.0,
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss else 0.0,
        "fees": round(fees, 2),
        "funding": round(funding, 2),
        "symbols": sorted({str(field(r, "symbol", "?")) for r in rows}),
        "exit_reasons": _counts(field(r, "exit_reason", "?") for r in rows),
    }


def _significance(r_multiples: list[float]) -> tuple[float, float, int]:
    """Standard error of the mean R, its t-stat, and the sample that would
    make it significant.

    Without this a class with four trades and a +0.7R average reads like
    an edge. The number of trades needed to clear the bar at the observed
    spread is the honest way to say how far off that conclusion is.

    A sample whose R values are all identical has zero spread and would
    divide by zero; it is reported as unmeasurable rather than as
    infinitely significant, because an identical run of four is a sign of
    too few trades, not of certainty.
    """
    n = len(r_multiples)
    if n < 2:
        return 0.0, 0.0, 0

    mean = sum(r_multiples) / n
    variance = sum((r - mean) ** 2 for r in r_multiples) / (n - 1)
    sd = math.sqrt(variance)
    if sd == 0:
        return 0.0, 0.0, 0

    se = sd / math.sqrt(n)
    t_stat = mean / se if se else 0.0
    # Solve n for |t| = critical at this mean and spread. The critical
    # value itself depends on n, so iterate once from the large-sample
    # figure — it converges immediately at any sample worth reporting.
    needed = 0
    if mean:
        needed = int(math.ceil((1.96 * sd / mean) ** 2))
        needed = int(math.ceil((t_critical(max(needed, 2)) * sd / mean) ** 2))
    return se, t_stat, needed


def by_asset_class(trades: list) -> dict:
    """Per-class summary over every trade given."""
    buckets: dict[str, list] = defaultdict(list)
    for trade in trades:
        buckets[class_of(trade)].append(trade)
    return {name: summarize_class(buckets[name]) for name in _order(buckets)}


def by_symbol(trades: list) -> dict:
    """Per-instrument summary — which coin, which stock."""
    buckets: dict[str, list] = defaultdict(list)
    for trade in trades:
        buckets[str(field(trade, "symbol", "?"))].append(trade)
    out = {}
    for symbol in sorted(buckets):
        rows = buckets[symbol]
        out[symbol] = summarize_class(rows) | {"asset_class": class_of(rows[0])}
    return out


# ── Per-day ──────────────────────────────────────────────────

def daily_pnl(trades: list) -> dict[str, dict[str, float]]:
    """Realised P&L per day, split by class.

    Keyed by the day the position *closed* — see the module docstring.
    """
    out: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for trade in trades:
        day = str(field(trade, "closed_on_day", "") or field(trade, "opened_on_day", ""))
        if not day:
            continue
        out[day][class_of(trade)] += number(trade, "pnl")
        out[day]["total"] += number(trade, "pnl")
    return {day: dict(classes) for day, classes in sorted(out.items())}


def daily_table(trades: list, days: list | None = None) -> list[dict]:
    """One row per trading day: equity, return, and P&L by class.

    `days` are the journal's own day records, which carry the equity curve
    including unrealised marks. Without them the table still works but
    reports realised P&L only.
    """
    realised = daily_pnl(trades)
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for trade in trades:
        day = str(field(trade, "closed_on_day", "") or "")
        if day:
            counts[day][class_of(trade)] += 1

    by_day = {str(d.get("day")): d for d in (days or [])}
    all_days = sorted(set(realised) | set(by_day))

    rows = []
    for day in all_days:
        record = by_day.get(day, {})
        rows.append({
            "day": day,
            "equity": _float(record.get("ending_equity")),
            "return_pct": _float(record.get("return_pct")),
            "realized_pnl": round(realised.get(day, {}).get("total", 0.0), 2),
            "by_class": {k: round(v, 2)
                         for k, v in realised.get(day, {}).items() if k != "total"},
            "trades_by_class": dict(counts.get(day, {})),
            "trades_closed": sum(counts.get(day, {}).values()),
            "halted_reason": record.get("halted_reason", "") or "",
        })
    return rows


def _float(value, default=0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _counts(values) -> dict:
    out: dict[str, int] = defaultdict(int)
    for value in values:
        out[str(value)] += 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


# ── The whole report ─────────────────────────────────────────

def period_report(trades: list, days: list | None = None,
                  starting_equity: float | None = None,
                  window_days: int | None = None) -> dict:
    """Cross-asset P&L over a window, per day and in aggregate.

    `window_days` keeps only the most recent N days of records — the
    "last three months" cut.
    """
    rows = daily_table(trades, days)
    kept_days = {r["day"] for r in rows}
    if window_days and len(rows) > window_days:
        rows = rows[-window_days:]
        kept_days = {r["day"] for r in rows}
        trades = [t for t in trades
                  if str(field(t, "closed_on_day", "")) in kept_days]

    equity_start = starting_equity
    if equity_start is None and days:
        first = next((d for d in days if str(d.get("day")) in kept_days), None)
        equity_start = _float((first or {}).get("starting_equity"), 0.0)
    equity_start = equity_start or 0.0
    equity_end = rows[-1]["equity"] if rows and rows[-1]["equity"] else equity_start

    classes = by_asset_class(trades)
    total_pnl = sum(c["pnl"] for c in classes.values())

    return {
        "days": len(rows),
        "period": {"from": rows[0]["day"], "to": rows[-1]["day"]} if rows else {},
        "starting_equity": round(equity_start, 2),
        "ending_equity": round(equity_end, 2),
        "total_return_pct": round((equity_end - equity_start) / equity_start * 100, 3)
        if equity_start else 0.0,
        "realized_pnl": round(total_pnl, 2),
        "trades": sum(c["trades"] for c in classes.values()),
        "by_asset_class": classes,
        "by_symbol": by_symbol(trades),
        "daily": rows,
        "best_day": max(rows, key=lambda r: r["realized_pnl"], default=None),
        "worst_day": min(rows, key=lambda r: r["realized_pnl"], default=None),
        "winning_days": sum(1 for r in rows if r["realized_pnl"] > 0),
        "losing_days": sum(1 for r in rows if r["realized_pnl"] < 0),
        "flat_days": sum(1 for r in rows if r["realized_pnl"] == 0),
    }


# ── Rendering ────────────────────────────────────────────────

def render(report: dict, logger, daily_rows: int = 0) -> None:
    """Print the cross-asset comparison the way it should be read."""
    period = report.get("period") or {}
    logger.info("═" * 78)
    logger.info("  CROSS-ASSET P&L — %d day(s) %s → %s", report["days"],
                period.get("from", "?"), period.get("to", "?"))
    logger.info("═" * 78)
    logger.info("  Equity      : $%.2f → $%.2f (%+.2f%%)",
                report["starting_equity"], report["ending_equity"],
                report["total_return_pct"])
    logger.info("  Days        : %d up / %d down / %d flat",
                report["winning_days"], report["losing_days"], report["flat_days"])

    classes = report.get("by_asset_class") or {}
    if not classes:
        logger.info("  No trades in this window.")
        logger.info("═" * 78)
        return

    logger.info("─" * 78)
    logger.info("  %-16s %7s %10s %8s %9s %8s %7s %6s",
                "asset class", "trades", "P&L $", "win%", "expect R",
                "± SE", "t", "real?")
    for name, stats in classes.items():
        logger.info("  %-16s %7d %10.2f %8.1f %+9.3f %8.3f %7.2f %6s",
                    CLASS_LABEL.get(name, name), stats["trades"], stats["pnl"],
                    stats.get("win_rate", 0.0), stats.get("expectancy_r", 0.0),
                    stats.get("se_r", 0.0), stats.get("t_stat", 0.0),
                    "yes" if stats.get("significant") else "no")

    # The honesty line. Without it a four-trade class with a fine average
    # reads like a finding.
    thin = [CLASS_LABEL.get(n, n) for n, s in classes.items()
            if not s.get("significant") and s["trades"]]
    if thin:
        logger.info("─" * 78)
        logger.info("  Not statistically distinguishable from zero: %s",
                    ", ".join(thin))
        for name, stats in classes.items():
            needed = stats.get("trades_for_significance") or 0
            if needed and not stats.get("significant"):
                logger.info("    %-16s would need ~%d trades at this spread "
                            "(has %d, needs |t| > %.2f)",
                            CLASS_LABEL.get(name, name), needed,
                            stats["trades"], stats.get("t_critical", 2.0))

    if daily_rows:
        logger.info("─" * 78)
        names = list(classes)
        header = "  %-12s %11s %9s %9s" % ("day", "equity", "ret%", "P&L $")
        header += "".join(" %>11s".replace(">", "") % CLASS_LABEL.get(n, n)[:11]
                          for n in names)
        logger.info(header)
        for row in report["daily"][-daily_rows:]:
            line = "  %-12s %11.2f %+9.2f %+9.2f" % (
                row["day"], row["equity"], row["return_pct"], row["realized_pnl"])
            for name in names:
                value = row["by_class"].get(name)
                line += " %11s" % (f"{value:+.2f}" if value else "·")
            logger.info(line)

    logger.info("═" * 78)
