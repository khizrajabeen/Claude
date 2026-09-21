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


VARIANTS: dict[str, dict] = {
    "single": {
        "label": "trend only, flat exposure",
        "overrides": {
            "strategies": {"enabled": ["trend"]},
            "vol_target": {"enabled": False},
        },
    },
    "multi-equal": {
        "label": "5 strategies, equal weights",
        "overrides": {
            "strategies": {"enabled": ["trend", "xsmom", "breakout", "reversion", "carry"]},
            "portfolio": {"weighting": "equal", "correlation_haircut": False},
            "vol_target": {"enabled": False},
        },
    },
    "multi-parity": {
        "label": "5 strategies, inverse-vol + correlation haircut",
        "overrides": {
            "strategies": {"enabled": ["trend", "xsmom", "breakout", "reversion", "carry"]},
            "portfolio": {"weighting": "inverse_vol", "correlation_haircut": True},
            "vol_target": {"enabled": False},
        },
    },
    "full": {
        "label": "risk parity + volatility targeting",
        "overrides": {
            "strategies": {"enabled": ["trend", "xsmom", "breakout", "reversion", "carry"]},
            "portfolio": {"weighting": "inverse_vol", "correlation_haircut": True},
            "vol_target": {"enabled": True},
        },
    },
}


class VariantBench:
    """Runs each variant over one shared dataset."""

    def __init__(self, config: dict):
        self.config = config

    def run(self, days: int = 90, variants: list[str] | None = None) -> dict:
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
        logger.info("  %-14s %8s %8s %8s %7s %9s %8s",
                    "variant", "return%", "maxDD%", "sharpe", "trades", "expect R", "PF")
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
                "expectancy_r": trades.get("expectancy_r", 0.0),
                "profit_factor": trades.get("profit_factor", 0.0),
            }
            rows.append(row)
            logger.info(
                "  %-14s %8.2f %8.2f %8.2f %7d %9.3f %8.2f",
                name, row["return_pct"], row["max_dd_pct"], row["sharpe"],
                row["trades"], row["expectancy_r"], row["profit_factor"],
            )

        logger.info("═" * 78)
        if len(rows) < 2:
            return

        first, last = rows[0], rows[-1]
        logger.info(
            "  %s → %s: return %+.2f pts, drawdown %+.2f pts, Sharpe %+.2f",
            first["variant"], last["variant"],
            last["return_pct"] - first["return_pct"],
            last["max_dd_pct"] - first["max_dd_pct"],
            last["sharpe"] - first["sharpe"],
        )
        logger.info(
            "  These are one sample over one period. Treat the direction of the "
            "differences as the signal, not their size."
        )
        logger.info("═" * 78)


def _merge(base: dict, overrides: dict) -> dict:
    """Deep-merge overrides into a copy of the base config."""
    merged = copy.deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = copy.deepcopy(value)
    return merged
