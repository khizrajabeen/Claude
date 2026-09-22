"""Per-class trading profiles: stop width, leverage and the edge bar.

A perpetual is the only leg that can use leverage, and leverage is what
makes a wide stop affordable — the dollar risk is unchanged but the margin
falls, so the account can carry a position whose stop sits out of the
noise and still have cash for the rest of the book. Over a 90-day replay
65% of all exits were stop-outs and an equity entry was refused for want
of $1,300 while the crypto book sat on the cash.

The danger of leverage is that it converts a normal loss into a total one
if the stop sits outside the liquidation price, so that is what most of
these tests are about.
"""

import pytest

from bot.markets.instrument import TRADING_PROFILE, trading_profile
from bot.risk.budget import RiskBudget


def _order(config, asset_class, atr_pct=0.01, equity=10_000.0, risk_pct=0.75,
           price=100.0):
    return RiskBudget(config).size_order(
        symbol="X/USDT", side="long", entry_price=price, atr=price * atr_pct,
        equity=equity, risk_pct=risk_pct, asset_class=asset_class,
    )


# ── The profiles themselves ──────────────────────────────────

def test_a_perp_takes_a_wider_stop_than_spot():
    assert (TRADING_PROFILE["crypto_perp"]["atr_stop_mult"]
            > TRADING_PROFILE["crypto_spot"]["atr_stop_mult"])


def test_only_derivatives_may_use_leverage():
    """Spot is cash and no borrow is modelled for it or for equities."""
    assert TRADING_PROFILE["crypto_perp"]["max_leverage"] > 1.0
    for cash_class in ("crypto_spot", "equity", "etf"):
        assert TRADING_PROFILE[cash_class]["max_leverage"] == 1.0


def test_a_perp_must_clear_a_higher_edge_bar():
    assert (TRADING_PROFILE["crypto_perp"]["min_edge"]
            > TRADING_PROFILE["crypto_spot"]["min_edge"])


def test_config_can_override_a_profile(config):
    config["profiles"] = {"crypto_perp": {"max_leverage": 1.0}}
    assert trading_profile("crypto_perp", config)["max_leverage"] == 1.0
    # Untouched keys keep their defaults.
    assert trading_profile("crypto_perp", config)["atr_stop_mult"] == \
        TRADING_PROFILE["crypto_perp"]["atr_stop_mult"]


def test_an_unknown_class_has_an_empty_profile():
    assert trading_profile("something_new") == {}


# ── Sizing ───────────────────────────────────────────────────

def test_a_wider_stop_buys_a_smaller_position(config):
    """Same dollar risk, further stop, therefore fewer units."""
    perp = _order(config, "crypto_perp")
    spot = _order(config, "crypto_spot")
    assert perp.r_distance > spot.r_distance
    assert perp.quantity < spot.quantity


def test_leverage_cuts_the_margin_not_the_risk(config):
    """The whole point: the stop is where it is and a stop-out costs the
    same dollars either way."""
    perp = _order(config, "crypto_perp")
    assert perp.leverage > 1.0
    margin = perp.notional / perp.leverage
    assert margin < perp.notional
    assert perp.quantity * perp.r_distance == pytest.approx(perp.risk_usd)


def test_a_cash_instrument_is_never_levered(config):
    for cash_class in ("crypto_spot", "equity", "etf"):
        assert _order(config, cash_class).leverage == 1.0, cash_class


def test_the_stop_stays_well_inside_the_liquidation_price(config):
    """Leverage that puts liquidation nearer than the stop converts a
    normal loss into a total one."""
    for atr_pct in (0.005, 0.01, 0.02, 0.05, 0.10):
        order = _order(config, "crypto_perp", atr_pct=atr_pct)
        if order.quantity <= 0:
            continue
        stop_fraction = order.r_distance / order.entry_price
        liquidation_fraction = 1.0 / order.leverage - 0.005
        assert liquidation_fraction > stop_fraction, (
            f"at {atr_pct:.1%} ATR the stop sits outside liquidation")


def test_a_violent_market_reduces_the_leverage(config):
    """A wide stop on a wild instrument cannot be carried at full
    leverage without crossing the liquidation price."""
    calm = _order(config, "crypto_perp", atr_pct=0.01)
    wild = _order(config, "crypto_perp", atr_pct=0.06)
    assert wild.leverage < calm.leverage
    assert "liquidation_buffer" in wild.caps_applied


def test_reducing_leverage_never_goes_below_one(config):
    order = _order(config, "crypto_perp", atr_pct=0.25)
    assert order.leverage >= 1.0


def test_the_dollar_risk_is_the_same_across_classes(config):
    """Different stops and leverage, one risk budget."""
    risks = []
    for asset_class in ("crypto_perp", "crypto_spot"):
        order = _order(config, asset_class, atr_pct=0.03)
        risks.append(order.quantity * order.r_distance)
    assert risks[0] == pytest.approx(risks[1], rel=0.02)


# ── The edge bar ─────────────────────────────────────────────

def _planner(config):
    from bot.daily.plan import DayPlanner
    from bot.portfolio.allocator import StrategyAllocator
    return DayPlanner(config, RiskBudget(config), [], StrategyAllocator(config))


def _read(asset_class):
    from bot.daily.briefing import SymbolRead
    return SymbolRead(symbol="X", price=1.0, atr=1.0, atr_pct=1.0, adx=1.0,
                      regime="ranging", asset_class=asset_class)


def test_a_perp_demands_more_conviction_than_spot(config):
    planner = _planner(config)
    assert planner._min_edge(_read("crypto_perp")) > \
        planner._min_edge(_read("crypto_spot"))


def test_a_class_without_a_profile_falls_back_to_the_global_bar(config):
    planner = _planner(config)
    assert planner._min_edge(_read("something_new")) == planner.min_edge
