"""The cross-asset P&L report.

The point of this report is to answer "did the perps or the stocks make
the money?", so the tests are mostly about the ways it could answer that
question wrongly: crediting P&L to the wrong day, losing a class, or
presenting four trades with a fine average as though it were a finding.
"""

import pytest

from bot.utils.pnl import (
    by_asset_class,
    by_symbol,
    daily_pnl,
    daily_table,
    period_report,
    summarize_class,
)


def trade(symbol="BTC/USDT:USDT", asset_class="crypto_perp", pnl=10.0, r=0.5,
          closed="2026-09-21", opened=None, fees=1.0, funding=0.0,
          exit_reason="take_profit"):
    return {
        "symbol": symbol, "asset_class": asset_class, "pnl": pnl,
        "r_multiple": r, "closed_on_day": closed,
        "opened_on_day": opened or closed, "fees": fees, "funding": funding,
        "exit_reason": exit_reason,
    }


# ── Bucketing ────────────────────────────────────────────────

def test_classes_are_kept_apart():
    rows = [
        trade(asset_class="crypto_perp", pnl=100.0),
        trade(symbol="NVDA", asset_class="equity", pnl=-40.0),
        trade(symbol="SPY", asset_class="etf", pnl=25.0),
    ]
    out = by_asset_class(rows)
    assert set(out) == {"crypto_perp", "equity", "etf"}
    assert out["crypto_perp"]["pnl"] == pytest.approx(100.0)
    assert out["equity"]["pnl"] == pytest.approx(-40.0)
    assert out["etf"]["pnl"] == pytest.approx(25.0)


def test_classes_come_out_in_a_stable_order():
    """A report whose columns move between runs cannot be compared."""
    rows = [trade(symbol="SPY", asset_class="etf"),
            trade(asset_class="crypto_perp"),
            trade(symbol="NVDA", asset_class="equity")]
    assert list(by_asset_class(rows)) == ["crypto_perp", "equity", "etf"]
    assert list(by_asset_class(list(reversed(rows)))) == \
        ["crypto_perp", "equity", "etf"]


def test_an_unlabelled_trade_is_bucketed_not_dropped():
    out = by_asset_class([{"symbol": "X", "pnl": 5.0, "r_multiple": 0.1,
                           "closed_on_day": "2026-09-21"}])
    assert out["unknown"]["trades"] == 1
    assert out["unknown"]["pnl"] == pytest.approx(5.0)


def test_objects_work_as_well_as_dicts():
    """The journal hands back Trade objects; a CSV read hands back dicts."""
    class Row:
        symbol, asset_class = "NVDA", "equity"
        pnl, r_multiple, fees, funding = 25.0, 0.5, 1.0, 0.0
        closed_on_day, opened_on_day, exit_reason = "2026-09-21", "2026-09-21", "tp"

    out = by_asset_class([Row()])
    assert out["equity"]["pnl"] == pytest.approx(25.0)
    assert out["equity"]["expectancy_r"] == pytest.approx(0.5)


def test_by_symbol_keeps_each_instruments_class():
    rows = [trade(symbol="NVDA", asset_class="equity"),
            trade(symbol="BTC/USDT", asset_class="crypto_spot")]
    out = by_symbol(rows)
    assert out["NVDA"]["asset_class"] == "equity"
    assert out["BTC/USDT"]["asset_class"] == "crypto_spot"


# ── Significance ─────────────────────────────────────────────

def test_the_bar_is_the_critical_value_for_the_sample_in_hand():
    """At four trades the 95% threshold is |t| = 3.18, not 1.96.

    A flat 1.96 at every sample size calls a four-trade run significant,
    which is exactly the error this column exists to prevent.
    """
    from bot.utils.pnl import t_critical

    assert t_critical(4) == pytest.approx(3.182, abs=0.01)
    assert t_critical(2) > t_critical(10) > t_critical(200)
    assert t_critical(500) == pytest.approx(1.96, abs=0.01)


def test_a_thin_sample_faces_a_higher_bar_than_a_fat_one():
    tight = [0.9, 1.1, 0.8, 1.2]
    stats_small = summarize_class([trade(r=r) for r in tight])
    stats_large = summarize_class([trade(r=r) for r in tight * 8])

    assert stats_small["t_critical"] > stats_large["t_critical"]
    assert stats_large["significant"], "32 consistent trades is an edge"


def test_a_noisy_thin_sample_is_not_called_significant():
    """Four trades with a fine average but real spread is not a finding."""
    rows = [trade(r=r) for r in (2.0, -0.9, 1.6, -0.6)]
    stats = summarize_class(rows)
    assert stats["expectancy_r"] > 0.0
    assert not stats["significant"], "|t| is driven by n as well as the mean"


def test_a_consistent_edge_over_many_trades_is_called_significant():
    rows = [trade(r=r) for r in [0.6, 0.4, 0.7, 0.5, 0.6, 0.5, 0.4, 0.6,
                                 0.5, 0.7, 0.6, 0.5, 0.4, 0.6, 0.5, 0.6]]
    stats = summarize_class(rows)
    assert stats["significant"]
    assert stats["t_stat"] > 2.0


def test_the_sample_needed_for_significance_is_reported():
    rows = [trade(r=r) for r in (1.5, -0.9, 1.1, -1.0, 0.8)]
    stats = summarize_class(rows)
    assert not stats["significant"]
    assert stats["trades_for_significance"] > len(rows)


def test_a_single_trade_reports_no_error_bars_rather_than_dividing_by_zero():
    stats = summarize_class([trade(r=1.0)])
    assert stats["trades"] == 1
    assert stats["se_r"] == 0.0
    assert stats["t_stat"] == 0.0
    assert not stats["significant"]


def test_win_rate_and_profit_factor():
    rows = [trade(pnl=100.0, r=1.0), trade(pnl=100.0, r=1.0),
            trade(pnl=-50.0, r=-0.5)]
    stats = summarize_class(rows)
    assert stats["win_rate"] == pytest.approx(66.7, abs=0.1)
    assert stats["profit_factor"] == pytest.approx(4.0)


def test_an_unbeaten_class_reports_zero_profit_factor_not_infinity():
    stats = summarize_class([trade(pnl=10.0, r=1.0)])
    assert stats["profit_factor"] == 0.0


# ── Per day ──────────────────────────────────────────────────

def test_pnl_is_credited_to_the_day_the_position_closed():
    """A position opened Monday and closed Thursday is Thursday's P&L.

    That matches how realised P&L is booked everywhere else, and how an
    exchange statement reads.
    """
    rows = [trade(opened="2026-09-18", closed="2026-09-21", pnl=90.0)]
    daily = daily_pnl(rows)
    assert "2026-09-18" not in daily
    assert daily["2026-09-21"]["total"] == pytest.approx(90.0)


def test_a_day_splits_its_pnl_across_classes():
    rows = [
        trade(asset_class="crypto_perp", pnl=50.0, closed="2026-09-21"),
        trade(symbol="NVDA", asset_class="equity", pnl=-20.0, closed="2026-09-21"),
    ]
    daily = daily_pnl(rows)
    assert daily["2026-09-21"]["crypto_perp"] == pytest.approx(50.0)
    assert daily["2026-09-21"]["equity"] == pytest.approx(-20.0)
    assert daily["2026-09-21"]["total"] == pytest.approx(30.0)


def test_days_come_out_in_order():
    rows = [trade(closed="2026-09-21"), trade(closed="2026-09-19"),
            trade(closed="2026-09-20")]
    assert list(daily_pnl(rows)) == ["2026-09-19", "2026-09-20", "2026-09-21"]


def test_a_day_with_no_closes_still_appears_when_the_journal_has_it():
    """A flat day is a fact about the strategy, not an absence of data."""
    days = [{"day": "2026-09-20", "ending_equity": 10_000.0, "return_pct": 0.0,
             "starting_equity": 10_000.0},
            {"day": "2026-09-21", "ending_equity": 10_100.0, "return_pct": 1.0,
             "starting_equity": 10_000.0}]
    rows = daily_table([trade(closed="2026-09-21", pnl=100.0)], days)
    assert [r["day"] for r in rows] == ["2026-09-20", "2026-09-21"]
    assert rows[0]["realized_pnl"] == 0.0
    assert rows[0]["trades_closed"] == 0


def test_the_table_carries_the_equity_curve_from_the_journal():
    days = [{"day": "2026-09-21", "ending_equity": 10_100.0, "return_pct": 1.0,
             "starting_equity": 10_000.0}]
    rows = daily_table([trade(closed="2026-09-21", pnl=100.0)], days)
    assert rows[0]["equity"] == pytest.approx(10_100.0)
    assert rows[0]["return_pct"] == pytest.approx(1.0)


def test_the_table_works_without_journal_days():
    rows = daily_table([trade(closed="2026-09-21", pnl=100.0)])
    assert rows[0]["realized_pnl"] == pytest.approx(100.0)
    assert rows[0]["equity"] == 0.0


# ── The window ───────────────────────────────────────────────

def _three_days():
    days, trades = [], []
    for i, day in enumerate(["2026-09-19", "2026-09-20", "2026-09-21"]):
        days.append({"day": day, "starting_equity": 10_000.0 + i * 100,
                     "ending_equity": 10_100.0 + i * 100, "return_pct": 1.0})
        trades.append(trade(closed=day, pnl=100.0))
    return trades, days


def test_the_window_keeps_only_the_most_recent_days():
    trades, days = _three_days()
    report = period_report(trades, days, window_days=2)
    assert report["days"] == 2
    assert report["period"] == {"from": "2026-09-20", "to": "2026-09-21"}


def test_trades_outside_the_window_are_excluded_from_the_totals():
    """Otherwise the per-class table describes a longer period than the
    header claims."""
    trades, days = _three_days()
    report = period_report(trades, days, window_days=2)
    assert report["trades"] == 2
    assert report["realized_pnl"] == pytest.approx(200.0)


def test_a_window_longer_than_the_record_keeps_everything():
    trades, days = _three_days()
    report = period_report(trades, days, window_days=500)
    assert report["days"] == 3
    assert report["trades"] == 3


def test_up_down_and_flat_days_are_counted():
    days = [{"day": "2026-09-19", "ending_equity": 10_000.0, "return_pct": 0.0,
             "starting_equity": 10_000.0},
            {"day": "2026-09-20", "ending_equity": 10_100.0, "return_pct": 1.0,
             "starting_equity": 10_000.0},
            {"day": "2026-09-21", "ending_equity": 9_900.0, "return_pct": -2.0,
             "starting_equity": 10_100.0}]
    trades = [trade(closed="2026-09-20", pnl=100.0),
              trade(closed="2026-09-21", pnl=-200.0)]
    report = period_report(trades, days)
    assert (report["winning_days"], report["losing_days"], report["flat_days"]) \
        == (1, 1, 1)


def test_best_and_worst_days_are_identified():
    trades = [trade(closed="2026-09-20", pnl=100.0),
              trade(closed="2026-09-21", pnl=-200.0)]
    report = period_report(trades)
    assert report["best_day"]["day"] == "2026-09-20"
    assert report["worst_day"]["day"] == "2026-09-21"


def test_an_empty_record_reports_nothing_rather_than_raising():
    report = period_report([], [])
    assert report["days"] == 0
    assert report["by_asset_class"] == {}
    assert report["best_day"] is None


def test_the_report_names_every_class_that_traded():
    trades = [trade(asset_class="crypto_perp"),
              trade(symbol="BTC/USDT", asset_class="crypto_spot"),
              trade(symbol="NVDA", asset_class="equity"),
              trade(symbol="SPY", asset_class="etf")]
    report = period_report(trades)
    assert set(report["by_asset_class"]) == {
        "crypto_perp", "crypto_spot", "equity", "etf"
    }


def test_render_does_not_raise_on_a_real_shaped_report():
    import logging
    trades, days = _three_days()
    trades.append(trade(symbol="NVDA", asset_class="equity", pnl=-30.0,
                        closed="2026-09-21", r=-0.8))
    from bot.utils.pnl import render
    render(period_report(trades, days), logging.getLogger("test"), daily_rows=5)


def test_render_survives_an_empty_report():
    import logging
    from bot.utils.pnl import render
    render(period_report([], []), logging.getLogger("test"), daily_rows=5)
