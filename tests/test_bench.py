"""The variant bench: does each layer we added pay for itself?

These check the experiment is fair — same data to every variant, isolated
records, one layer changed at a time — not that any particular variant
wins. Which one wins is what the measurement is for.
"""

from __future__ import annotations

import pytest

from bot.utils.bench import VARIANTS, VariantBench, _merge
from tests.conftest import make_ohlcv

START_SYMBOLS = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]


@pytest.fixture
def frames():
    from datetime import datetime, timezone
    start = datetime(2026, 2, 1, tzinfo=timezone.utc)
    out = {}
    for i, symbol in enumerate(START_SYMBOLS):
        out[symbol] = {
            "1h": make_ohlcv(bars=1400, start_price=100 * (i + 1),
                             drift=0.0020 - 0.0008 * i, vol=0.010,
                             seed=40 + i, start=start),
            "4h": make_ohlcv(bars=350, start_price=100 * (i + 1),
                             drift=0.0080 - 0.0032 * i, vol=0.020,
                             seed=40 + i, start=start, freq_hours=4),
        }
    return out


# ── Variant definitions ──────────────────────────────────────

def test_variants_isolate_one_change_each():
    """The point is attribution: each step adds exactly one thing."""
    single = VARIANTS["single"]["overrides"]
    equal = VARIANTS["multi-equal"]["overrides"]
    parity = VARIANTS["multi-parity"]["overrides"]
    full = VARIANTS["full"]["overrides"]

    assert single["strategies"]["enabled"] == ["trend"]
    assert len(equal["strategies"]["enabled"]) == 5

    # multi-equal -> multi-parity changes only the weighting.
    assert equal["strategies"] == parity["strategies"]
    assert equal["portfolio"]["weighting"] != parity["portfolio"]["weighting"]
    assert equal["vol_target"] == parity["vol_target"]

    # multi-parity -> full changes only volatility targeting.
    assert parity["portfolio"] == full["portfolio"]
    assert parity["vol_target"]["enabled"] is False
    assert full["vol_target"]["enabled"] is True


def test_merge_is_deep_and_non_destructive():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    merged = _merge(base, {"a": {"y": 9}, "c": 4})

    assert merged == {"a": {"x": 1, "y": 9}, "b": 3, "c": 4}
    assert base == {"a": {"x": 1, "y": 2}, "b": 3}, "the base must not be mutated"


def test_unknown_variant_is_rejected(config):
    with pytest.raises(ValueError, match="Unknown variant"):
        VariantBench(config).run(days=2, variants=["nonsense"])


# ── Running ──────────────────────────────────────────────────

def test_bench_runs_every_variant_over_the_same_data(config, frames, monkeypatch):
    config["data"]["symbols"] = list(frames)
    bench = VariantBench(config)

    monkeypatch.setattr("bot.utils.backtester.DailyReplay._download",
                        lambda self, days: frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._limits", lambda self: {})

    report = bench.run(days=6, variants=["single", "multi-equal", "full"])
    assert set(report["variants"]) == {"single", "multi-equal", "full"}

    for name, result in report["variants"].items():
        assert "error" not in result, f"{name} failed: {result.get('error')}"
        assert result["days"] == 6
        assert result["starting_equity"] == config["paper"]["initial_balance"]
        assert result["label"]


def test_each_variant_keeps_its_own_records(config, frames, monkeypatch):
    config["data"]["symbols"] = list(frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._download",
                        lambda self, days: frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._limits", lambda self: {})

    report = VariantBench(config).run(days=4, variants=["single", "full"])
    directories = {r["journal_dir"] for r in report["variants"].values()}
    assert len(directories) == 2, "variants must not share a journal"


def test_a_single_strategy_variant_really_runs_one(config, frames, monkeypatch):
    config["data"]["symbols"] = list(frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._download",
                        lambda self, days: frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._limits", lambda self: {})

    merged = _merge(config, VARIANTS["single"]["overrides"])
    from bot.strategies import build_strategies
    assert [s.name for s in build_strategies(merged)] == ["trend"]


def test_vol_targeting_changes_exposure_between_variants(config):
    """The 'full' variant must actually switch the targeter on."""
    from bot.portfolio.voltarget import VolatilityTargeter

    off = _merge(config, VARIANTS["multi-parity"]["overrides"])
    on = _merge(config, VARIANTS["full"]["overrides"])

    wild = [0.04, -0.05, 0.06, -0.04, 0.05, -0.06, 0.04, -0.05, 0.05, -0.04]
    assert VolatilityTargeter(off).decide(wild).scale == 1.0
    assert VolatilityTargeter(on).decide(wild).scale < 1.0
