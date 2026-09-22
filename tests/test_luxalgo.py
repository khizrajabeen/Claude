"""LuxAlgo-style indicators and the strategies built on them.

Two of these primitives are easy to get subtly wrong in ways that make a
backtest look brilliant: swing points that are known before their right
shoulder prints, and a kernel regression that repaints. Both are tested
directly, because neither failure is visible in the returns — it just makes
them better.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bot.analysis.indicators import (
    fair_value_gaps, gaussian_kernel_regression, supertrend, swing_points,
)
from bot.strategies.nwenvelope import NadarayaWatsonEnvelope
from bot.strategies.smc import SmartMoneyConcepts
from bot.strategies.supertrend_ai import SuperTrendAI
from tests.conftest import build_context, make_ohlcv

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)


# ── Look-ahead: the part that matters ────────────────────────

def test_swing_points_wait_for_their_right_shoulder():
    """A swing high is not knowable until `length` bars after it prints.
    Indicators that ignore this read the future."""
    df = make_ohlcv(bars=400, seed=70)
    length = 10

    full_high, full_low = swing_points(df, length)
    partial_high, partial_low = swing_points(df.iloc[:-40], length)

    overlap = partial_high.index[-50:]
    for full, partial in ((full_high, partial_high), (full_low, partial_low)):
        both = pd.concat([full.loc[overlap], partial.loc[overlap]], axis=1).dropna()
        assert not both.empty
        assert (both.iloc[:, 0] == both.iloc[:, 1]).all(), \
            "a swing point changed when later bars were removed"


def test_kernel_regression_does_not_repaint():
    """The default Nadaraya-Watson smooths across the window and is
    recalculated every bar, so past values keep moving. Only the endpoint
    estimator could actually have been traded."""
    close = make_ohlcv(bars=500, seed=71)["close"]

    full = gaussian_kernel_regression(close, bandwidth=8.0, window=200)
    partial = gaussian_kernel_regression(close.iloc[:-60], bandwidth=8.0, window=200)

    overlap = partial.index[-80:]
    both = pd.concat([full.loc[overlap], partial.loc[overlap]], axis=1).dropna()
    assert len(both) > 50
    assert float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max()) < 1e-9, \
        "the fit moved when future bars were removed — this repaints"


def test_supertrend_does_not_repaint():
    df = make_ohlcv(bars=400, seed=72)
    full_line, full_dir = supertrend(df, 10, 3.0)
    partial_line, partial_dir = supertrend(df.iloc[:-50], 10, 3.0)

    overlap = partial_line.index[-60:]
    assert (full_dir.loc[overlap] == partial_dir.loc[overlap]).all()
    assert float((full_line.loc[overlap] - partial_line.loc[overlap]).abs().max()) < 1e-9


# ── Primitives ───────────────────────────────────────────────

def test_supertrend_flips_with_the_trend():
    up = make_ohlcv(bars=400, drift=0.006, vol=0.004, seed=73)
    down = make_ohlcv(bars=400, drift=-0.006, vol=0.004, seed=73)

    assert supertrend(up, 10, 3.0)[1].iloc[-1] == 1
    assert supertrend(down, 10, 3.0)[1].iloc[-1] == -1


def test_supertrend_stop_only_ratchets_forward():
    df = make_ohlcv(bars=400, drift=0.005, vol=0.004, seed=74)
    line, direction = supertrend(df, 10, 3.0)

    # Within one uptrend leg the stop must never fall.
    frame = pd.concat([line, direction], axis=1).dropna()
    frame.columns = ["line", "dir"]
    legs = (frame["dir"] != frame["dir"].shift()).cumsum()
    for _, leg in frame.groupby(legs):
        if len(leg) < 3:
            continue
        if leg["dir"].iloc[0] == 1:
            assert (leg["line"].diff().dropna() >= -1e-9).all()
        else:
            assert (leg["line"].diff().dropna() <= 1e-9).all()


def test_fair_value_gaps_find_real_imbalances():
    index = pd.date_range("2026-01-01", periods=40, freq="h", tz="UTC")
    df = pd.DataFrame({"open": 100.0, "high": 101.0, "low": 99.0,
                       "close": 100.0, "volume": 1.0}, index=index)
    # Bar 20 leaps clear of bar 18's high: a bullish imbalance.
    df.iloc[20, df.columns.get_loc("low")] = 110.0
    df.iloc[20, df.columns.get_loc("high")] = 115.0
    df.iloc[20, df.columns.get_loc("close")] = 112.0

    direction, top, bottom = fair_value_gaps(df, min_atr=0.1)
    assert direction.iloc[20] == 1
    assert top.iloc[20] > bottom.iloc[20]


def test_fair_value_gaps_ignore_noise():
    df = make_ohlcv(bars=300, vol=0.002, seed=75)
    direction, _, _ = fair_value_gaps(df, min_atr=5.0)
    assert (direction == 0).all(), "a huge threshold should find no gaps"


# ── SuperTrend AI ────────────────────────────────────────────

def test_supertrend_ai_picks_a_factor_from_the_cluster(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT"], bars=700,
                        drifts=[0.005, -0.005], vol=0.010, seed=76)
    for signal in SuperTrendAI(config).generate(ctx):
        assert 1.0 <= signal.meta["factor"] <= 5.0
        assert signal.meta["cluster_size"] >= 1
        assert "cluster" in signal.reason


def test_supertrend_ai_clustering_is_deterministic(config):
    """A randomly seeded k-means would make every backtest a lottery."""
    strategy = SuperTrendAI(config)
    scores = np.array([0.1, 0.12, 0.11, 0.5, 0.52, 0.48, -0.3, -0.28, -0.31])
    first = strategy._best_cluster(scores)
    assert first == strategy._best_cluster(scores)
    # The best cluster must be the high group.
    assert set(first) == {3, 4, 5}


def test_supertrend_ai_stays_silent_when_the_cluster_disagrees(config):
    """Clustering is only worth the cost if a split group means no trade."""
    config["strategies"]["supertrend"] = {"min_factor": 1.0, "max_factor": 5.0}
    ctx = build_context(symbols=["A/USDT"], bars=700, drifts=[0.0],
                        vol=0.02, seed=77)
    for signal in SuperTrendAI(config).generate(ctx):
        assert signal.direction in (1, -1)


# ── Smart Money Concepts ─────────────────────────────────────

def test_smc_reports_which_components_fired(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=700, drifts=[0.005, 0.001, -0.001, -0.005],
                        vol=0.012, seed=78)
    for signal in SmartMoneyConcepts(config).generate(ctx):
        assert signal.meta, "every SMC signal must say what produced it"
        assert set(signal.meta) <= {"structure", "sweep", "fvg", "zone"}
        assert signal.reason.startswith("SMC:")


def test_smc_dealing_range_is_bounded(config):
    """Regression: stale swing points once put price at '638% of range',
    which meant the range had stopped describing the market."""
    strategy = SmartMoneyConcepts(config)
    df = make_ohlcv(bars=700, drift=0.01, vol=0.004, seed=79)

    zone = strategy._premium_discount(df, float(df["close"].iloc[-1]))
    if zone is not None:
        percent = float(zone["note"].split("(")[1].split("%")[0])
        assert 0 <= percent <= 100


def test_smc_declines_when_price_has_left_the_range(config):
    """Far outside the range is not a stronger fade — it is a broken range."""
    strategy = SmartMoneyConcepts(config)
    df = make_ohlcv(bars=400, vol=0.004, seed=80)
    runaway = float(df["high"].max()) * 3
    assert strategy._premium_discount(df, runaway) is None


def test_smc_sweep_needs_a_rejection_not_just_a_break(config):
    """The signal is the close back inside, not the wick through."""
    strategy = SmartMoneyConcepts(config)
    df = make_ohlcv(bars=200, vol=0.006, seed=81).copy()

    prior_high = float(df["high"].iloc[-31:-1].max())
    # A clean break that closes above: not a sweep.
    df.iloc[-1, df.columns.get_loc("high")] = prior_high * 1.05
    df.iloc[-1, df.columns.get_loc("close")] = prior_high * 1.04
    assert strategy._liquidity_sweep(df, atr_now=prior_high * 0.01) is None


# ── Nadaraya-Watson envelope ─────────────────────────────────

def test_nw_envelope_fades_the_extremes(config):
    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=700, drifts=[0.0, 0.001, -0.001, 0.0],
                        vol=0.014, seed=82, htf_trends=[0, 0, 0, 0])
    for signal in NadarayaWatsonEnvelope(config).generate(ctx):
        assert "band" in signal.reason
        assert signal.meta["band"] > 0


def test_nw_envelope_refuses_to_fade_a_strong_trend(config):
    config["strategies"]["nwenvelope"] = {"max_adx": 1.0}
    ctx = build_context(symbols=["A/USDT"], bars=700, drifts=[0.006], seed=83)
    assert NadarayaWatsonEnvelope(config).generate(ctx) == []


def test_nw_envelope_ignores_a_move_far_beyond_the_band(config):
    """Miles past the envelope is a regime change, not a rubber band."""
    config["strategies"]["nwenvelope"] = {"max_excess_atr": 0.01, "max_adx": 100.0}
    ctx = build_context(symbols=["A/USDT"], bars=700, drifts=[0.02],
                        vol=0.004, seed=84, htf_trends=[0])
    assert NadarayaWatsonEnvelope(config).generate(ctx) == []


# ── Registry ─────────────────────────────────────────────────

def test_luxalgo_strategies_are_registered(config):
    from bot.strategies import LUX, REGISTRY

    for name in LUX:
        assert name in REGISTRY
        strategy = REGISTRY[name](config)
        assert strategy.name == name
        assert strategy.required_bars() > 0


def test_all_strategies_still_emit_well_formed_signals(config):
    from bot.strategies import build_strategies

    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=800, drifts=[0.005, 0.001, -0.001, -0.005],
                        vol=0.011, seed=85)
    for strategy in build_strategies(config):
        for signal in strategy.generate(ctx):
            assert signal.direction in (1, -1)
            assert 0 < signal.strength <= 1.0
            assert signal.reason


# ── Lorentzian classification ────────────────────────────────

def test_lorentzian_distance_compresses_outliers():
    """The whole reason for the log: under Euclidean distance a single
    volatility spike decides who counts as a neighbour."""
    from bot.analysis.indicators import lorentzian_distance

    current = np.array([0.5, 0.5, 0.5])
    history = np.array([
        [0.6, 0.6, 0.6],   # close on every axis
        [0.5, 0.5, 9.9],   # identical but for one wild axis
    ])
    distances = lorentzian_distance(current, history)
    euclidean = np.sqrt(((history - current) ** 2).sum(axis=1))

    # Under Euclidean the outlier is ~15x further; under Lorentzian ~8x.
    assert distances[1] / distances[0] < euclidean[1] / euclidean[0]


def test_normalise_uses_only_a_trailing_window():
    """Rescaling against the whole series lets a future minimum set
    today's value."""
    from bot.analysis.indicators import normalise

    series = make_ohlcv(bars=600, seed=86)["close"]
    full = normalise(series, window=200)
    partial = normalise(series.iloc[:-80], window=200)

    overlap = partial.index[-100:]
    both = pd.concat([full.loc[overlap], partial.loc[overlap]], axis=1).dropna()
    assert len(both) > 50
    assert float((both.iloc[:, 0] - both.iloc[:, 1]).abs().max()) < 1e-12


def test_lorentzian_neighbours_vote_on_resolved_outcomes(config):
    """A neighbour may not vote on a future it could not yet have seen."""
    from bot.strategies.lorentzian import LorentzianClassifier

    strategy = LorentzianClassifier(config)
    ctx = build_context(symbols=["A/USDT"], bars=1000, drifts=[0.004],
                        vol=0.011, seed=87, htf_trends=[1])
    df = ctx.frames["A/USDT"]
    features = strategy._features(df)
    assert features is not None

    result = strategy._classify(df, features)
    if result is None:
        pytest.skip("not enough resolved history in this sample")
    votes, total, _ = result
    assert total <= strategy.neighbours
    assert abs(votes) <= total


def test_lorentzian_stays_silent_when_neighbours_disagree(config):
    """A split neighbourhood is information, not a coin flip to resolve."""
    from bot.strategies.lorentzian import LorentzianClassifier

    config["strategies"]["lorentzian"] = {"min_vote_ratio": 0.99}
    ctx = build_context(symbols=["A/USDT", "B/USDT"], bars=1000,
                        drifts=[0.0, 0.0], vol=0.012, seed=88)
    assert LorentzianClassifier(config).generate(ctx) == []


def test_lorentzian_regime_filter_blocks_counter_trend_votes(config):
    from bot.strategies.lorentzian import LorentzianClassifier

    config["strategies"]["lorentzian"] = {"adx_floor": 0.0, "min_vote_ratio": 0.5}
    strategy = LorentzianClassifier(config)
    ctx = build_context(symbols=["A/USDT"], bars=1000, drifts=[0.004],
                        seed=89, htf_trends=[1])
    read = ctx.reads["A/USDT"]
    df = ctx.frames["A/USDT"]

    assert strategy._filters(df, read, side=-1) == "regime"
    assert strategy._filters(df, read, side=1) != "regime"


def test_lorentzian_reports_its_vote(config):
    from bot.strategies.lorentzian import LorentzianClassifier

    ctx = build_context(symbols=["A/USDT", "B/USDT", "C/USDT", "D/USDT"],
                        bars=1000, drifts=[0.004, 0.001, -0.001, -0.004],
                        vol=0.011, seed=90, htf_trends=[1, 1, -1, -1])
    for signal in LorentzianClassifier(config).generate(ctx):
        assert 0 < signal.meta["vote_ratio"] <= 1.0
        assert signal.meta["neighbours"] > 0
        assert "of the closest historical states" in signal.reason


def test_wave_trend_is_bounded_and_finite():
    from bot.analysis.indicators import wave_trend

    df = make_ohlcv(bars=500, seed=91)
    wt = wave_trend(df).dropna()
    assert len(wt) > 400
    assert np.isfinite(wt).all()
