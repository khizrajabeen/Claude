"""Replay several configurations over identical data and compare them.

The claim worth testing is not "the new stack is better" but "each layer we
added paid for itself". So the variants are cumulative, and each one adds
exactly one thing:

    single        one strategy, flat exposure — the old single-driver bot
    multi-equal   five strategies, equally weighted — diversification only
    multi-parity  ... weighted by inverse volatility with a correlation
                  haircut — risk budgeting on top
    full          ... plus portfolio volatility targeting

Every variant sees the same downloaded bars, the same costs and the same
random draw, so differences between them come from the configuration and
nothing else. Data is downloaded once and handed to each run.

This measures, it does not tune. Picking the best variant on this sample
and reporting its number as an expectation is precisely the overfitting the
purged-validation work exists to avoid — the point is the *shape* of the
differences, and whether they match what the research predicts.
"""

from __future__ import annotations

import copy
import logging

from bot.utils.backtester import DailyReplay

logger = logging.getLogger("trading_bot")


SELECTIVE = ["clenow", "turtle", "holygrail"]
LUX = ["supertrend", "smc", "nwenvelope", "lorentzian"]
EVERYTHING = SELECTIVE + LUX

_NO_VOL_TARGET = {"enabled": False}
_PARITY = {"weighting": "inverse_vol", "correlation_haircut": True}
_OFF = {"enabled": False}


def _solo(name: str, label: str) -> dict:
    """One strategy on its own, with the portfolio layers out of the way."""
    return {
        "label": label,
        "overrides": {"strategies": {"enabled": [name]},
                      "vol_target": _NO_VOL_TARGET, "autopilot": _OFF},
    }


def _group(names: list[str], label: str, autopilot: bool = False) -> dict:
    """A family, risk-weighted and volatility-targeted."""
    return {
        "label": label,
        "overrides": {"strategies": {"enabled": list(names)},
                      "portfolio": _PARITY, "vol_target": {"enabled": True},
                      "autopilot": {"enabled": True} if autopilot else _OFF},
    }


VARIANTS: dict[str, dict] = {
    # Each strategy alone, so a family's result can be attributed.
    "clenow": _solo("clenow", "Clenow trend only"),
    "turtle": _solo("turtle", "Turtle only (Dennis & Eckhardt)"),
    "holygrail": _solo("holygrail", "Holy Grail pullback only (Raschke)"),
    "supertrend": _solo("supertrend", "SuperTrend AI clustering only"),
    "smc": _solo("smc", "Smart Money Concepts only"),
    "nwenvelope": _solo("nwenvelope", "Nadaraya-Watson envelope only"),
    "lorentzian": _solo("lorentzian", "Lorentzian kNN only"),
    # Families.
    "selective": _group(SELECTIVE, "the three published systems, risk parity"),
    "lux": _group(LUX, "the four indicator-style signals, risk parity"),
    "everything": _group(EVERYTHING, "all seven, risk parity + vol target"),
    "autopilot": _group(EVERYTHING, "all seven, autopilot manages the roster",
                        autopilot=True),
}

# Every strategy on its own, for attribution.
SOLO = ["clenow", "turtle", "holygrail", "supertrend", "smc", "nwenvelope",
        "lorentzian"]
# The families worth testing out of sample.
CANDIDATES = ["selective", "lux", "everything", "autopilot"]
GROUPS = {"solo": SOLO, "candidates": CANDIDATES}


class VariantBench:
    """Runs each variant over one shared dataset."""

    def __init__(self, config: dict):
        self.config = config

    def run_split(self, days: int = 300, variants: list[str] | None = None,
                  split: float = 0.5) -> dict:
        """Run each variant on an early half and a later half separately.

        This is the only honest way to judge a roster that was *chosen* by
        looking at results. `selective` exists because clenow, turtle and
        holygrail led an earlier bench; repeating that bench would just
        confirm the choice that was made from it. Splitting the history
        answers the question that matters — does the pick survive on data
        it was not picked on?

        In-sample numbers here are reported for contrast and should be
        ignored when judging any variant that was selected.
        """
        base = DailyReplay(self.config)
        frames = base._download(days)
        if not frames:
            return {"error": "no data", "in_sample": {}, "out_of_sample": {}}

        early, late = _split_frames(frames, split)
        if not early or not late:
            return {"error": "not enough history to split", "in_sample": {},
                    "out_of_sample": {}}

        half_days = max(1, int(days * split)), max(1, int(days * (1 - split)))

        logger.info("═" * 78)
        logger.info("  OUT-OF-SAMPLE BENCH — %d days split %.0f/%.0f",
                    days, split * 100, (1 - split) * 100)
        logger.info("  A variant picked on one half proves nothing on that half.")
        logger.info("═" * 78)

        results = {}
        for label, subset, subset_days in (("in_sample", early, half_days[0]),
                                           ("out_of_sample", late, half_days[1])):
            logger.info("─" * 78)
            logger.info("  %s half", label.replace("_", " ").upper())
            logger.info("─" * 78)
            results[label] = self._run_variants(
                names=self._names(variants), frames=subset, days=subset_days,
                limits=base._limits(), tag=label, exchange=getattr(base, "_exchange", None),
            )

        self._log_split(results)
        return {"days": days, "split": split, **results}

    def _names(self, variants: list[str] | None) -> list[str]:
        if variants and len(variants) == 1 and variants[0] in GROUPS:
            return list(GROUPS[variants[0]])
        return variants or list(VARIANTS)

    def _run_variants(self, names: list[str], frames: dict, days: int,
                      limits: dict, tag: str = "", exchange=None) -> dict:
        """Replay every named variant over one set of frames."""
        results: dict[str, dict] = {}
        for position, name in enumerate(names, start=1):
            spec = VARIANTS[name]
            logger.info("─" * 78)
            logger.info("  [%d/%d] %s — %s", position, len(names), name, spec["label"])
            logger.info("─" * 78)
            variant_config = _merge(self.config, spec["overrides"])
            suffix = f"{tag}/{name}" if tag else name
            variant_config.setdefault("journal", {})["replay_dir"] = \
                f"{self.config.get('journal', {}).get('dir', 'state')}/bench/{suffix}"

            replay = DailyReplay(variant_config)
            replay._exchange = exchange
            try:
                report = replay.run(days=days, frames=frames)
            except Exception as e:
                logger.error("  Variant %s failed: %s", name, e, exc_info=True)
                results[name] = {"error": str(e)[:200], "label": spec["label"]}
                continue
            report["label"] = spec["label"]
            results[name] = report
        return results

    def _log_split(self, results: dict) -> None:
        """Side-by-side, with the out-of-sample column the only one that counts."""
        in_sample = results.get("in_sample", {})
        out_sample = results.get("out_of_sample", {})

        logger.info("═" * 78)
        logger.info("  IN-SAMPLE vs OUT-OF-SAMPLE")
        logger.info("═" * 78)
        logger.info("  %-16s %18s %24s", "variant", "in-sample", "OUT-OF-SAMPLE")
        logger.info("  %-16s %8s %9s %11s %7s %6s",
                    "", "return%", "trades", "return%", "trades", "t")
        logger.info("  " + "-" * 74)

        for name in in_sample:
            first = in_sample.get(name, {})
            second = out_sample.get(name, {})
            if "error" in first or "error" in second:
                logger.info("  %-16s  failed", name)
                continue
            a = first.get("trades", {})
            b = second.get("trades", {})
            logger.info(
                "  %-16s %8.2f %9d %11.2f %7d %6.2f",
                name, first.get("total_return_pct", 0.0), a.get("total_trades", 0),
                second.get("total_return_pct", 0.0), b.get("total_trades", 0),
                b.get("t_stat", 0.0),
            )

        logger.info("  " + "-" * 74)
        survivors = [
            name for name in out_sample
            if "error" not in out_sample[name]
            and out_sample[name].get("total_return_pct", 0) > 0
            and in_sample.get(name, {}).get("total_return_pct", 0) > 0
        ]
        if survivors:
            logger.info("  Positive in both halves: %s", ", ".join(survivors))
            logger.info("  That is consistency, not proof — check the t column.")
        else:
            logger.info(
                "  No variant is positive in both halves. Anything that looked "
                "good in one is not repeating in the other, which is what "
                "picking winners from a sample usually produces."
            )
        logger.info("═" * 78)

    def run(self, days: int = 90, variants: list[str] | None = None) -> dict:
        if variants and len(variants) == 1 and variants[0] in GROUPS:
            names = list(GROUPS[variants[0]])
        else:
            names = variants or list(VARIANTS)
        unknown = [n for n in names if n not in VARIANTS]
        if unknown:
            raise ValueError(f"Unknown variant(s): {unknown}. Known: {list(VARIANTS)}")

        logger.info("═" * 78)
        logger.info("  VARIANT BENCH — %d days, %d configurations, identical data",
                    days, len(names))
        logger.info("═" * 78)

        # Download once. Handing the same frames to every variant is what
        # makes the comparison about configuration rather than luck.
        base = DailyReplay(self.config)
        frames = base._download(days)
        if not frames:
            return {"error": "no data", "variants": {}}
        limits = base._limits()

        results: dict[str, dict] = {}
        for name in names:
            spec = VARIANTS[name]
            logger.info("─" * 78)
            logger.info("  Variant: %s — %s", name, spec["label"])
            logger.info("─" * 78)

            variant_config = _merge(self.config, spec["overrides"])
            # Each variant gets its own records directory.
            variant_config.setdefault("journal", {})["replay_dir"] = \
                f"{self.config.get('journal', {}).get('dir', 'state')}/bench/{name}"

            replay = DailyReplay(variant_config)
            replay._exchange = getattr(base, "_exchange", None)
            try:
                report = replay.run(days=days, frames=frames)
            except Exception as e:
                logger.error("  Variant %s failed: %s", name, e, exc_info=True)
                results[name] = {"error": str(e)[:200], "label": spec["label"]}
                continue
            report["label"] = spec["label"]
            results[name] = report

        self._log_table(results)
        return {"days": days, "variants": results}

    def _log_table(self, results: dict) -> None:
        logger.info("═" * 78)
        logger.info("  VARIANT COMPARISON")
        logger.info("═" * 78)
        logger.info("  %-14s %8s %8s %7s %9s %8s %7s %6s",
                    "variant", "return%", "maxDD%", "trades", "expect R",
                    "± SE", "t", "real?")
        logger.info("  " + "-" * 74)

        rows = []
        for name, report in results.items():
            if "error" in report:
                logger.info("  %-14s  failed: %s", name, report["error"][:50])
                continue
            trades = report.get("trades", {})
            row = {
                "variant": name,
                "return_pct": report.get("total_return_pct", 0.0),
                "max_dd_pct": trades.get("max_drawdown_pct", 0.0),
                "sharpe": trades.get("sharpe_daily", 0.0),
                "trades": trades.get("total_trades", 0),
                "expectancy_r": trades.get("avg_r", 0.0),
                "expectancy_se": trades.get("expectancy_se", 0.0),
                "t_stat": trades.get("t_stat", 0.0),
                "significant": trades.get("significant", False),
                "trades_for_significance": trades.get("trades_for_significance", 0),
                "profit_factor": trades.get("profit_factor", 0.0),
            }
            rows.append(row)
            logger.info(
                "  %-14s %8.2f %8.2f %7d %9.3f %8.3f %7.2f %6s",
                name, row["return_pct"], row["max_dd_pct"], row["trades"],
                row["expectancy_r"], row["expectancy_se"], row["t_stat"],
                "yes" if row["significant"] else "no",
            )

        logger.info("  " + "-" * 74)

        significant = [r for r in rows if r["significant"]]
        if significant:
            for row in significant:
                logger.info("  %s clears |t| >= 2 on %d trades.",
                            row["variant"], row["trades"])
        else:
            logger.info(
                "  Not one variant clears |t| >= 2. Every expectancy here is "
                "compatible with having no edge at all."
            )

        # The trap this table exists to expose.
        thin = [r for r in rows if 0 < r["trades"] < 30 and r["expectancy_r"] > 0]
        for row in thin:
            needed = row["trades_for_significance"]
            logger.info(
                "  %s looks best but traded %d times; at this variance it would "
                "need ~%s to prove the edge is real.",
                row["variant"], row["trades"],
                f"{needed} trades" if needed else "far more trades",
            )

        logger.info("═" * 78)
        if len(rows) < 2:
            return

        first, last = rows[0], rows[-1]
        logger.info(
            "  %s → %s: return %+.2f pts, drawdown %+.2f pts",
            first["variant"], last["variant"],
            last["return_pct"] - first["return_pct"],
            last["max_dd_pct"] - first["max_dd_pct"],
        )
        logger.info(
            "  One sample, one period. Read the direction of the differences, "
            "never the size — and read the t column before either."
        )
        logger.info("═" * 78)


def _split_frames(frames: dict, split: float) -> tuple[dict, dict]:
    """Cut every symbol's history at the same moment in time.

    Splitting each symbol at its own row count would put the halves on
    different calendars, so the two runs would not be comparable — one
    variant would be judged on a bull month and another on a crash.
    """
    all_indexes = [
        df.index
        for timeframes in frames.values()
        for df in timeframes.values()
        if df is not None and not df.empty
    ]
    if not all_indexes:
        return {}, {}

    # The window every symbol has in common, cut at the same instant.
    start = max(index[0] for index in all_indexes)
    end = min(index[-1] for index in all_indexes)
    if end <= start:
        return {}, {}
    cut = start + (end - start) * split

    early: dict = {}
    late: dict = {}
    for symbol, timeframes in frames.items():
        early_tf, late_tf = {}, {}
        for timeframe, df in timeframes.items():
            if df is None or df.empty:
                continue
            early_tf[timeframe] = df.loc[df.index <= cut]
            late_tf[timeframe] = df.loc[df.index > cut]
        if early_tf and late_tf and all(len(d) for d in late_tf.values()):
            early[symbol] = early_tf
            late[symbol] = late_tf
    return early, late


def _merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of the base config."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = copy.deepcopy(value)
    return merged
