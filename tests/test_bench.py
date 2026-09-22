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
            "1h": make_ohlcv(bars=1800, start_price=100 * (i + 1),
                             drift=0.0020 - 0.0008 * i, vol=0.010,
                             seed=40 + i, start=start),
            "4h": make_ohlcv(bars=450, start_price=100 * (i + 1),
                             drift=0.0080 - 0.0032 * i, vol=0.020,
                             seed=40 + i, start=start, freq_hours=4),
        }
    return out


# ── Variant definitions ──────────────────────────────────────

def test_every_strategy_has_a_solo_variant():
    """A family's result is only interpretable if each member can be run
    on its own."""
    from bot.strategies import REGISTRY
    from bot.utils.bench import SOLO

    assert set(SOLO) == set(REGISTRY)
    for name in SOLO:
        assert VARIANTS[name]["overrides"]["strategies"]["enabled"] == [name]


def test_solo_variants_strip_the_portfolio_layers():
    """Otherwise a single strategy's number is really a test of vol
    targeting."""
    for name in ("clenow", "smc", "lorentzian"):
        overrides = VARIANTS[name]["overrides"]
        assert overrides["vol_target"]["enabled"] is False
        assert overrides["autopilot"]["enabled"] is False


def test_families_differ_only_in_membership():
    selective = VARIANTS["selective"]["overrides"]
    lux = VARIANTS["lux"]["overrides"]
    everything = VARIANTS["everything"]["overrides"]

    assert selective["portfolio"] == lux["portfolio"] == everything["portfolio"]
    assert selective["vol_target"] == lux["vol_target"] == everything["vol_target"]
    assert set(everything["strategies"]["enabled"]) == \
        set(selective["strategies"]["enabled"]) | set(lux["strategies"]["enabled"])


def test_autopilot_variant_differs_only_by_the_autopilot():
    everything = VARIANTS["everything"]["overrides"]
    autopilot = VARIANTS["autopilot"]["overrides"]

    assert everything["strategies"] == autopilot["strategies"]
    assert everything["portfolio"] == autopilot["portfolio"]
    assert everything["autopilot"]["enabled"] is False
    assert autopilot["autopilot"]["enabled"] is True


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

    report = bench.run(days=6, variants=["clenow", "selective", "lux"])
    assert set(report["variants"]) == {"clenow", "selective", "lux"}

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

    report = VariantBench(config).run(days=4, variants=["clenow", "selective"])
    directories = {r["journal_dir"] for r in report["variants"].values()}
    assert len(directories) == 2, "variants must not share a journal"


def test_a_solo_variant_really_runs_one(config):
    from bot.strategies import build_strategies

    merged = _merge(config, VARIANTS["turtle"]["overrides"])
    assert [s.name for s in build_strategies(merged)] == ["turtle"]


def test_vol_targeting_changes_exposure_between_variants(config):
    """The 'full' variant must actually switch the targeter on."""
    from bot.portfolio.voltarget import VolatilityTargeter

    off = _merge(config, VARIANTS["clenow"]["overrides"])
    on = _merge(config, VARIANTS["selective"]["overrides"])

    wild = [0.04, -0.05, 0.06, -0.04, 0.05, -0.06, 0.04, -0.05, 0.05, -0.04]
    assert VolatilityTargeter(off).decide(wild).scale == 1.0
    assert VolatilityTargeter(on).decide(wild).scale < 1.0


# ── Out-of-sample splitting ──────────────────────────────────

def test_split_cuts_every_symbol_at_the_same_moment(frames):
    """Splitting each symbol at its own row count would put the halves on
    different calendars, so the two runs would not be comparable."""
    from bot.utils.bench import _split_frames

    early, late = _split_frames(frames, 0.5)
    assert set(early) == set(late) == set(frames)

    early_ends = {df.index[-1] for tf in early.values() for df in tf.values()}
    late_starts = {df.index[0] for tf in late.values() for df in tf.values()}
    # Every symbol's early half ends before every symbol's late half starts.
    assert max(early_ends) < min(late_starts)


def test_split_halves_do_not_overlap(frames):
    from bot.utils.bench import _split_frames

    early, late = _split_frames(frames, 0.5)
    for symbol in frames:
        for timeframe in frames[symbol]:
            a = set(early[symbol][timeframe].index)
            b = set(late[symbol][timeframe].index)
            assert not (a & b), "a bar appeared in both halves"


def test_split_respects_the_requested_fraction(frames):
    from bot.utils.bench import _split_frames

    early, late = _split_frames(frames, 0.75)
    a = len(early["BTC/USDT"]["1h"])
    b = len(late["BTC/USDT"]["1h"])
    assert a > b, "a 75/25 split should leave more history in the first half"


def test_oos_run_reports_both_halves(config, frames, monkeypatch):
    config["data"]["symbols"] = list(frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._download",
                        lambda self, days: frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._limits", lambda self: {})

    report = VariantBench(config).run_split(days=20, variants=["clenow", "selective"],
                                            split=0.5)
    assert set(report["in_sample"]) == {"clenow", "selective"}
    assert set(report["out_of_sample"]) == {"clenow", "selective"}
    for half in ("in_sample", "out_of_sample"):
        for name, result in report[half].items():
            assert "error" not in result, f"{half}/{name}: {result.get('error')}"


def test_oos_halves_keep_separate_records(config, frames, monkeypatch):
    """The two halves must not write over each other's journals."""
    config["data"]["symbols"] = list(frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._download",
                        lambda self, days: frames)
    monkeypatch.setattr("bot.utils.backtester.DailyReplay._limits", lambda self: {})

    report = VariantBench(config).run_split(days=16, variants=["clenow"], split=0.5)
    first = report["in_sample"]["clenow"]["journal_dir"]
    second = report["out_of_sample"]["clenow"]["journal_dir"]
    assert first != second


def test_the_selective_family_is_the_three_that_survived():
    from bot.strategies import SELECTIVE

    assert set(VARIANTS["selective"]["overrides"]["strategies"]["enabled"]) \
        == set(SELECTIVE) == {"clenow", "turtle", "holygrail"}


def test_variant_groups_resolve(config):
    from bot.utils.bench import GROUPS, VariantBench

    bench = VariantBench(config)
    for group, expected in GROUPS.items():
        assert bench._names([group]) == expected
    assert bench._names(["clenow", "selective"]) == ["clenow", "selective"]
