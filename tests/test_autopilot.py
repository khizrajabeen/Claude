"""The autopilot: does the roster manage itself sensibly and safely?

The risk in any self-selecting scheme is chasing whatever worked last
month. These check the guards hold: a minimum sample before judging, a
threshold below zero rather than at it, a cap on how much can be benched
at once, and a probe allocation that keeps benching reversible.
"""

from __future__ import annotations

import pytest

from bot.portfolio.autopilot import REGIME_FIT, AutoPilot

NAMES = ["trend", "xsmom", "breakout", "reversion", "carry",
         "turtle", "clenow", "holygrail", "dualmom"]


def record(trades, avg_r):
    return {"trades": trades, "avg_r": avg_r}


def test_nothing_is_judged_on_a_small_sample(config):
    pilot = AutoPilot(config)
    stats = {name: record(5, -2.0) for name in NAMES}

    decision = pilot.decide(NAMES, stats)
    assert decision.benched == {}, "six trades is not evidence of anything"
    assert set(decision.active) == set(NAMES)


def test_a_persistently_losing_strategy_is_benched(config):
    pilot = AutoPilot(config)
    stats = {name: record(60, 0.10) for name in NAMES}
    stats["reversion"] = record(60, -0.45)

    decision = pilot.decide(NAMES, stats)
    assert "reversion" in decision.benched
    assert "trend" not in decision.benched
    assert "60 trades" in decision.benched["reversion"]


def test_a_slightly_negative_strategy_survives(config):
    """The threshold sits below zero on purpose — a small loss over a
    normal sample is indistinguishable from bad luck."""
    pilot = AutoPilot(config)
    stats = {name: record(60, -0.05) for name in NAMES}
    assert AutoPilot(config).decide(NAMES, stats).benched == {}


def test_benching_is_capped(config):
    """If most strategies are losing, the settings or the market are the
    problem; concentrating into the survivors makes it worse."""
    pilot = AutoPilot(config)
    stats = {name: record(60, -0.90) for name in NAMES}

    decision = pilot.decide(NAMES, stats)
    assert len(decision.benched) <= len(NAMES) // 2
    assert decision.notes, "the cap being hit must be reported"


def test_benched_strategies_keep_a_probe_allocation(config):
    """A strategy switched fully off produces no record and can never be
    switched back on."""
    pilot = AutoPilot(config)
    stats = {name: record(60, 0.10) for name in NAMES}
    stats["turtle"] = record(60, -0.50)

    decision = pilot.decide(NAMES, stats)
    weights = pilot.apply({name: 1 / len(NAMES) for name in NAMES}, decision)

    assert 0 < weights["turtle"] < weights["trend"] / 5
    assert sum(weights.values()) == pytest.approx(1.0)


def test_regime_tilt_favours_the_right_family(config):
    pilot = AutoPilot(config)
    equal = {name: 1 / len(NAMES) for name in NAMES}

    trending = pilot.apply(equal, pilot.decide(NAMES, {}, regime="trending_up"))
    ranging = pilot.apply(equal, pilot.decide(NAMES, {}, regime="ranging"))

    assert trending["trend"] > ranging["trend"]
    assert ranging["reversion"] > trending["reversion"]
    assert sum(trending.values()) == pytest.approx(1.0)


def test_the_regime_tilt_can_never_zero_a_strategy(config):
    """It is a tilt, not a switch — a mistaken regime call must not flatten
    the book."""
    for regime, table in REGIME_FIT.items():
        assert all(0.4 < value < 1.6 for value in table.values()), regime


def test_an_unknown_regime_is_neutral(config):
    pilot = AutoPilot(config)
    multipliers = pilot.regime_multipliers(NAMES, "no_such_regime")
    assert set(multipliers.values()) == {1.0}


def test_autopilot_can_be_switched_off(config):
    config["autopilot"] = {"enabled": False}
    pilot = AutoPilot(config)
    stats = {name: record(100, -1.0) for name in NAMES}

    decision = pilot.decide(NAMES, stats)
    assert decision.benched == {}
    assert decision.active == NAMES
    assert decision.notes


def test_weights_survive_every_strategy_being_benched(config):
    """Degenerate input must not produce NaN weights."""
    pilot = AutoPilot(config)
    decision = pilot.decide(NAMES, {})
    decision.benched = {name: "test" for name in NAMES}

    weights = pilot.apply({name: 1 / len(NAMES) for name in NAMES}, decision)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert all(w > 0 for w in weights.values())


def test_the_explanation_names_the_probes(config):
    pilot = AutoPilot(config)
    stats = {name: record(60, 0.1) for name in NAMES}
    stats["carry"] = record(60, -0.6)

    decision = pilot.decide(NAMES, stats)
    weights = pilot.apply({name: 1 / len(NAMES) for name in NAMES}, decision)
    assert "(probe)" in pilot.explain(weights, decision)


def test_journal_reports_per_strategy_records(config):
    """The autopilot's input has to come from the ledger, not from memory."""
    from datetime import datetime, timedelta, timezone

    from bot.daily.journal import Journal
    from bot.trading.models import Trade

    journal = Journal(config)
    now = datetime(2026, 8, 1, tzinfo=timezone.utc)
    for i in range(6):
        journal.append_trade(Trade(
            id=str(i), symbol="BTC/USDT", side="long", entry_price=100,
            exit_price=101, quantity=1, leverage=1, opened_at=now,
            closed_at=now + timedelta(hours=2),
            pnl=10.0 if i % 2 else -5.0, pnl_pct=1.0,
            r_multiple=2.0 if i % 2 else -1.0, fees=0.1, funding=0,
            slippage_cost=0, exit_reason="signal", risk_usd=5,
            strategy="turtle" if i < 3 else "reversion",
        ))

    stats = Journal(config).strategy_stats()
    assert set(stats) == {"turtle", "reversion"}
    assert stats["turtle"]["trades"] == 3
    assert "avg_r" in stats["turtle"] and "win_rate" in stats["turtle"]
