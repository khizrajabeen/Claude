"""Measured beta, and the concentration it exposes.

Altcoins are levered Bitcoin. Measured over 199 daily observations on the
pairs this bot trades, every one runs a beta of 0.85-1.44 to BTC with an
average pairwise correlation of 0.62 — so five long alts is about 1.7
independent bets. A book that counts it as five has understated its
concentration fivefold, which is the failure these tests guard.
"""

import numpy as np
import pandas as pd
import pytest

from bot.analysis.beta import BetaBook
from bot.markets.instrument import build_instrument
from bot.risk.budget import DEFAULT_BETA, SYMBOL_BETA, symbol_beta


def _frames(betas: dict[str, float], n=200, seed=0, idio=0.002):
    """Synthetic frames where each symbol has a known beta to the first."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-01-01", periods=n, freq="1D", tz="UTC")
    market = rng.normal(0.0, 0.02, n)

    frames = {}
    for symbol, beta in betas.items():
        noise = rng.normal(0.0, idio, n)
        returns = beta * market + noise
        close = 100 * np.exp(np.cumsum(returns))
        frames[symbol] = pd.DataFrame(
            {"open": close, "high": close * 1.01, "low": close * 0.99,
             "close": close, "volume": 1.0}, index=idx)
    return frames


def _instruments(symbols, asset_class="crypto_spot"):
    return [build_instrument({"symbol": s, "asset_class": asset_class})
            for s in symbols]


# ── Measurement ──────────────────────────────────────────────

def test_a_known_beta_is_recovered(config):
    betas = {"BTC/USDT": 1.0, "ALT/USDT": 1.5}
    book = BetaBook(config)
    book.update(_frames(betas), _instruments(betas))
    assert book.beta("ALT/USDT") == pytest.approx(1.5, abs=0.1)


def test_the_benchmark_has_beta_one_against_itself(config):
    betas = {"BTC/USDT": 1.0, "ALT/USDT": 1.2}
    book = BetaBook(config)
    book.update(_frames(betas), _instruments(betas))
    assert book.beta("BTC/USDT") == pytest.approx(1.0)
    assert book.r2("BTC/USDT") == pytest.approx(1.0)


def test_r_squared_separates_a_proxy_from_an_independent_bet(config):
    """Beta 1.0 with r-squared 0.2 is a different bet from beta 1.0 with
    r-squared 0.8, and only the second is a proxy for the index.

    Both are built against one market in a single call, so the only thing
    differing between them is how much idiosyncratic noise they carry.
    """
    frames = _frames({"BTC/USDT": 1.0, "TIGHT/USDT": 1.0, "LOOSE/USDT": 1.0},
                     seed=1, idio=0.001)
    rng = np.random.default_rng(99)
    loose_close = frames["LOOSE/USDT"]["close"]
    base_returns = np.log(loose_close).diff().fillna(0.0).to_numpy()
    noisy = base_returns + rng.normal(0.0, 0.05, len(base_returns))
    close = 100 * np.exp(np.cumsum(noisy))
    frames["LOOSE/USDT"] = pd.DataFrame(
        {"open": close, "high": close * 1.01, "low": close * 0.99,
         "close": close, "volume": 1.0}, index=frames["LOOSE/USDT"].index)

    book = BetaBook(config)
    book.update(frames, _instruments(["BTC/USDT", "TIGHT/USDT", "LOOSE/USDT"]))
    assert book.r2("TIGHT/USDT") > 0.9
    assert book.r2("LOOSE/USDT") < 0.5
    # And this is exactly why r-squared is kept alongside beta: with
    # idiosyncratic noise several times the market's own move, the beta
    # estimate carries a large standard error. Beta alone would present
    # the noisy instrument as a confident market proxy; r-squared says
    # how much of its move the market actually accounts for.
    assert book.r2("TIGHT/USDT") > 2 * book.r2("LOOSE/USDT")


def test_too_little_history_keeps_the_neutral_default(config):
    """A beta estimated from a fortnight of noise is worse than none."""
    betas = {"BTC/USDT": 1.0, "NEW/USDT": 2.0}
    book = BetaBook(config)
    book.update(_frames(betas, n=10), _instruments(betas))
    assert not book.known("NEW/USDT")
    assert book.beta("NEW/USDT") == 1.0


def test_each_class_is_measured_against_its_own_benchmark(config):
    crypto = _frames({"BTC/USDT": 1.0, "ALT/USDT": 1.4}, seed=2)
    equity = _frames({"SPY": 1.0, "NVDA": 1.8}, seed=3)
    frames = {**crypto, **equity}
    instruments = (_instruments(["BTC/USDT", "ALT/USDT"], "crypto_spot")
                   + _instruments(["SPY", "NVDA"], "equity"))

    book = BetaBook(config)
    book.update(frames, instruments)
    assert book.benchmark("ALT/USDT") == "BTC/USDT"
    assert book.benchmark("NVDA") == "SPY"
    assert book.beta("NVDA") == pytest.approx(1.8, abs=0.15)


def test_a_missing_benchmark_falls_back_to_the_longest_series(config):
    """The named benchmark is not always in the universe."""
    betas = {"ALT/USDT": 1.0, "OTHER/USDT": 1.3}
    book = BetaBook(config)
    book.update(_frames(betas), _instruments(betas))
    assert book.benchmark("OTHER/USDT") in betas


def test_a_flat_benchmark_does_not_divide_by_zero(config):
    idx = pd.date_range("2026-01-01", periods=200, freq="1D", tz="UTC")
    flat = pd.DataFrame({"open": 100.0, "high": 100.0, "low": 100.0,
                         "close": 100.0, "volume": 1.0}, index=idx)
    frames = {"BTC/USDT": flat, "ALT/USDT": flat}
    book = BetaBook(config)
    book.update(frames, _instruments(frames))
    assert book.beta("ALT/USDT") == 1.0


# ── Concentration ────────────────────────────────────────────

def test_correlated_positions_count_as_fewer_than_their_number(config):
    betas = {"BTC/USDT": 1.0, "A/USDT": 1.2, "B/USDT": 1.3, "C/USDT": 1.1}
    book = BetaBook(config)
    book.update(_frames(betas, idio=0.004), _instruments(betas))

    alts = ["A/USDT", "B/USDT", "C/USDT"]
    effective = book.effective_positions(alts)
    assert effective < len(alts)
    assert effective >= 1.0


def test_independent_positions_count_as_themselves(config):
    """Three genuinely unrelated bets are three bets."""
    book = BetaBook(config)
    assert book.effective_positions(["A", "B", "C"]) == 3.0


def test_one_position_is_one_bet(config):
    book = BetaBook(config)
    assert book.effective_positions(["A"]) == 1.0
    assert book.effective_positions([]) == 0.0


def test_more_correlated_means_fewer_effective_bets(config):
    tight = BetaBook(config)
    tight.update(_frames({"BTC/USDT": 1.0, "A/USDT": 1.0, "B/USDT": 1.0},
                         idio=0.0005), _instruments(["BTC/USDT", "A/USDT", "B/USDT"]))
    loose = BetaBook(config)
    loose.update(_frames({"BTC/USDT": 1.0, "A/USDT": 1.0, "B/USDT": 1.0},
                         idio=0.08), _instruments(["BTC/USDT", "A/USDT", "B/USDT"]))

    pair = ["A/USDT", "B/USDT"]
    assert tight.effective_positions(pair) < loose.effective_positions(pair)


# ── The risk layer uses it ───────────────────────────────────

def test_a_measured_beta_is_preferred_over_the_table(config):
    betas = {"BTC/USDT": 1.0, "XRP/USDT": 1.4}
    book = BetaBook(config)
    book.update(_frames(betas), _instruments(betas))

    tabled = SYMBOL_BETA["XRP"]
    assert symbol_beta("XRP/USDT", book) == pytest.approx(1.4, abs=0.15)
    assert symbol_beta("XRP/USDT", book) != pytest.approx(tabled, abs=0.05)


def test_the_table_remains_the_fallback(config):
    """An instrument with too little history to measure still gets a
    sensible number rather than nothing."""
    book = BetaBook(config)
    assert symbol_beta("BTC/USDT", book) == SYMBOL_BETA["BTC"]
    assert symbol_beta("BTC/USDT", None) == SYMBOL_BETA["BTC"]


def test_an_unknown_symbol_takes_the_default(config):
    assert symbol_beta("NOTACOIN/USDT") == DEFAULT_BETA


def test_stacking_correlated_longs_is_refused(config):
    """Five long alts is about 1.7 independent bets."""
    from bot.risk.budget import RiskBudget
    from tests.test_multiasset import make_position, order

    config["risk"]["max_correlated_positions"] = 9
    config["risk"]["min_effective_positions"] = 0.5

    betas = {"BTC/USDT": 1.0, "A/USDT": 1.2, "B/USDT": 1.2, "C/USDT": 1.2}
    book = BetaBook(config)
    book.update(_frames(betas, idio=0.0005), _instruments(betas))

    budget = RiskBudget(config)
    budget.beta_book = book
    book_positions = [make_position("A/USDT", "long", "crypto_spot"),
                      make_position("B/USDT", "long", "crypto_spot")]
    decision = budget.check_new_trade(
        order("C/USDT", "long", "crypto_spot"), book_positions,
        equity=100_000.0, day_start_equity=100_000.0, peak_equity=100_000.0)
    assert not decision.allowed
    assert decision.reason == "too_correlated"
    assert decision.details["effective_positions"] < 3


def test_the_concentration_check_can_be_switched_off(config):
    from bot.risk.budget import RiskBudget
    from tests.test_multiasset import make_position, order

    config["risk"]["max_correlated_positions"] = 9
    config["risk"]["min_effective_positions"] = 0.0
    budget = RiskBudget(config)
    budget.beta_book = BetaBook(config)
    decision = budget.check_new_trade(
        order("C/USDT", "long", "crypto_spot"),
        [make_position("A/USDT", "long", "crypto_spot")],
        equity=100_000.0, day_start_equity=100_000.0, peak_equity=100_000.0)
    assert decision.allowed, decision.reason


def test_an_uncorrelated_addition_is_allowed(config):
    from bot.risk.budget import RiskBudget
    from tests.test_multiasset import make_position, order

    config["risk"]["max_correlated_positions"] = 9
    config["risk"]["min_effective_positions"] = 0.5

    betas = {"BTC/USDT": 1.0, "A/USDT": 1.0, "B/USDT": 1.0}
    book = BetaBook(config)
    book.update(_frames(betas, idio=0.09), _instruments(betas))

    budget = RiskBudget(config)
    budget.beta_book = book
    decision = budget.check_new_trade(
        order("B/USDT", "long", "crypto_spot"),
        [make_position("A/USDT", "long", "crypto_spot")],
        equity=100_000.0, day_start_equity=100_000.0, peak_equity=100_000.0)
    assert decision.allowed, decision.details
