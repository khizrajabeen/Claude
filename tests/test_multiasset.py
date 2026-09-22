"""Data routing, cross-asset risk grouping and per-class attribution.

The theme is that four asset classes in one book must not be treated as
one factor. A stock and a perp are independent bets; counting them as
correlated denies the book the diversification that motivated holding
both, and netting their betas adds numbers measured against different
market factors.
"""

from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from bot.data.router import DataRouter
from bot.markets.instrument import build_instrument
from bot.risk.budget import (
    SizedOrder,
    RiskBudget,
    correlation_group,
    symbol_beta,
)
from bot.trading.models import Position
from bot.utils.backtester import attribute_by_class


def utc(y, m, d, hh=0, mm=0):
    return datetime(y, m, d, hh, mm, tzinfo=timezone.utc)


PERP = build_instrument({"symbol": "BTC/USDT:USDT", "asset_class": "crypto_perp"})
SPOT = build_instrument({"symbol": "BTC/USDT", "asset_class": "crypto_spot"})
STOCK = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
ETF = build_instrument({"symbol": "SPY", "asset_class": "etf"})


# ── Router ───────────────────────────────────────────────────

class StubProvider:
    def __init__(self, classes, frame=None, fail=False):
        self.classes = set(classes)
        self.frame = frame
        self.fail = fail
        self.asked = []

    def handles(self, instrument):
        return instrument.asset_class.value in self.classes

    def bars(self, instrument, timeframe, limit):
        self.asked.append((instrument.symbol, timeframe, limit))
        if self.fail:
            raise RuntimeError("feed down")
        return self.frame if self.frame is not None else pd.DataFrame()

    def price(self, instrument):
        if self.fail:
            raise RuntimeError("feed down")
        return 100.0

    def quote_volume(self, instrument):
        return 1_000_000.0

    def funding_rate(self, instrument):
        return 0.0001


def frame(n=10):
    idx = pd.date_range("2026-09-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0},
        index=idx,
    )


def test_each_instrument_reaches_the_provider_for_its_class(config):
    crypto = StubProvider(["crypto_spot", "crypto_perp"], frame())
    equity = StubProvider(["equity", "etf"], frame())
    router = DataRouter(config, providers=[crypto, equity])

    router.bars(PERP, "1h", 10)
    router.bars(STOCK, "1d", 10)

    assert [a[0] for a in crypto.asked] == ["BTC/USDT:USDT"]
    assert [a[0] for a in equity.asked] == ["NVDA"]


def test_one_dead_feed_does_not_take_down_the_day(config):
    """A provider that raises must yield an empty frame, not an exception.

    The briefing marks the symbol untradable and moves on; letting the
    error escape would abandon every other instrument that day.
    """
    crypto = StubProvider(["crypto_spot", "crypto_perp"], fail=True)
    router = DataRouter(config, providers=[crypto])

    result = router.bars(PERP, "1h", 10)
    assert isinstance(result, pd.DataFrame) and result.empty


def test_an_unhandled_class_returns_empty_rather_than_raising(config):
    router = DataRouter(config, providers=[StubProvider(["crypto_spot"])])
    assert router.bars(STOCK, "1d", 10).empty
    assert router.price(STOCK) is None


def test_prices_skips_what_it_cannot_get(config):
    crypto = StubProvider(["crypto_spot", "crypto_perp"], frame())
    router = DataRouter(config, providers=[crypto])
    prices = router.prices([PERP, STOCK])
    assert "BTC/USDT:USDT" in prices
    assert "NVDA" not in prices


def test_missing_optional_reads_degrade_to_none(config):
    """A provider need not implement the order book; OHLCV feeds have none."""
    router = DataRouter(config, providers=[StubProvider(["crypto_perp"], frame())])
    assert router.order_book(PERP) is None
    assert router.spread_bps(PERP) is None
    assert router.market_limits(PERP)["min_qty"] == 0.0


# ── Correlation grouping ─────────────────────────────────────

def test_spot_and_perp_of_one_coin_share_a_risk_factor():
    """Long BTC spot and long the BTC perp is one bet held twice."""
    assert correlation_group("crypto_spot") == correlation_group("crypto_perp")


def test_stocks_and_etfs_share_a_risk_factor():
    assert correlation_group("equity") == correlation_group("etf")


def test_crypto_and_equities_are_different_factors():
    assert correlation_group("crypto_perp") != correlation_group("equity")


def test_beta_is_quoted_against_each_groups_own_factor():
    assert symbol_beta("BTC/USDT") == pytest.approx(1.0)
    assert symbol_beta("SPY") == pytest.approx(1.0)
    assert symbol_beta("COIN") > symbol_beta("MSFT")


def make_position(symbol, side, asset_class, price=100.0, qty=1.0):
    return Position(
        id=f"p-{symbol}", symbol=symbol, side=side, quantity=qty,
        entry_price=price, leverage=1.0, opened_at=utc(2026, 9, 21),
        stop_price=price * 0.98, take_profit=price * 1.04,
        risk_usd=price * qty * 0.02, atr_at_entry=1.0,
        opened_on_day="2026-09-21", asset_class=asset_class,
    )


def order(symbol, side, asset_class, price=100.0, qty=1.0):
    return SizedOrder(
        symbol=symbol, side=side, quantity=qty, entry_price=price,
        stop_price=price * 0.98, take_profit=price * 1.04,
        risk_usd=price * qty * 0.02, notional=price * qty,
        leverage=1.0, atr=1.0, r_distance=price * 0.02,
        asset_class=asset_class,
    )


def check(config, the_order, positions):
    return RiskBudget(config).check_new_trade(
        the_order, positions, equity=100_000.0, day_start_equity=100_000.0,
        peak_equity=100_000.0,
    )


def test_crowding_is_capped_within_a_factor(config):
    config["risk"]["max_correlated_positions"] = 2
    book = [
        make_position("BTC/USDT", "long", "crypto_spot"),
        make_position("ETH/USDT", "long", "crypto_spot"),
    ]
    decision = check(config, order("SOL/USDT", "long", "crypto_spot"), book)
    assert not decision.allowed
    assert decision.reason == "max_correlated_positions"


def test_a_stock_is_not_crowded_out_by_crypto(config):
    """Three long coins and a long stock are two factors, not four bets.

    Counting every same-side position in the book was the old behaviour,
    and it would have blocked the equity leg the moment crypto filled up —
    so the cross-asset comparison would have been crypto versus nothing.
    """
    config["risk"]["max_correlated_positions"] = 2
    book = [
        make_position("BTC/USDT", "long", "crypto_spot"),
        make_position("ETH/USDT:USDT", "long", "crypto_perp"),
    ]
    decision = check(config, order("NVDA", "long", "equity"), book)
    assert decision.allowed, decision.reason
    assert decision.details["group"] == correlation_group("equity")
    assert decision.details["same_side_in_group"] == 0


def test_the_perp_leg_counts_against_the_spot_leg(config):
    config["risk"]["max_correlated_positions"] = 2
    book = [
        make_position("BTC/USDT", "long", "crypto_spot"),
        make_position("BTC/USDT:USDT", "long", "crypto_perp"),
    ]
    decision = check(config, order("SOL/USDT", "long", "crypto_spot"), book)
    assert not decision.allowed
    assert decision.reason == "max_correlated_positions"


def test_net_beta_is_measured_per_factor(config):
    """A crypto book at its beta limit must not veto an equity trade."""
    config["risk"]["max_net_beta"] = 0.5
    config["risk"]["max_correlated_positions"] = 9
    heavy = [make_position("BTC/USDT", "long", "crypto_spot", price=100.0, qty=400.0)]

    budget = RiskBudget(config)
    crypto_beta = budget.net_beta_exposure(heavy, 100_000.0, group="crypto")
    equity_beta = budget.net_beta_exposure(heavy, 100_000.0, group="us_equity")
    assert crypto_beta > 0.3
    assert equity_beta == 0.0

    assert not check(config, order("ETH/USDT", "long", "crypto_spot",
                                   price=100.0, qty=400.0), heavy).allowed
    assert check(config, order("MSFT", "long", "equity",
                               price=100.0, qty=10.0), heavy).allowed


def test_a_record_written_before_asset_classes_groups_as_crypto():
    """Old journals have no asset_class; a pair symbol still means crypto."""
    assert correlation_group("", "BTC/USDT") == correlation_group("crypto_spot")
    assert correlation_group(None, "NVDA") == correlation_group("equity")


# ── Per-class attribution ────────────────────────────────────

def trade_row(symbol, asset_class, pnl, r):
    return {"symbol": symbol, "asset_class": asset_class, "pnl": pnl,
            "r_multiple": r, "fees": 1.0, "funding": 0.0}


def test_attribution_splits_pnl_by_class():
    rows = [
        trade_row("BTC/USDT:USDT", "crypto_perp", 120.0, 1.2),
        trade_row("ETH/USDT:USDT", "crypto_perp", -40.0, -0.9),
        trade_row("NVDA", "equity", 60.0, 0.8),
    ]
    out = attribute_by_class(rows)
    assert set(out) == {"crypto_perp", "equity"}
    assert out["crypto_perp"]["trades"] == 2
    assert out["crypto_perp"]["pnl"] == pytest.approx(80.0)
    assert out["equity"]["pnl"] == pytest.approx(60.0)
    assert out["crypto_perp"]["win_rate"] == pytest.approx(50.0)
    assert out["equity"]["expectancy_r"] == pytest.approx(0.8)


def test_attribution_reads_objects_as_well_as_dicts():
    """The journal hands back Trade objects; a JSON report hands back dicts.
    Attribution is asked for from both paths."""
    class Row:
        symbol, asset_class, pnl, r_multiple, fees, funding = \
            "NVDA", "equity", 25.0, 0.5, 1.0, 0.0

    out = attribute_by_class([Row()])
    assert out["equity"]["pnl"] == pytest.approx(25.0)
    assert out["equity"]["expectancy_r"] == pytest.approx(0.5)


def test_attribution_buckets_unlabelled_trades_rather_than_dropping_them():
    out = attribute_by_class([trade_row("BTC/USDT", None, 10.0, 0.4)])
    assert out["unknown"]["trades"] == 1


def test_attribution_of_nothing_is_empty_not_an_error():
    assert attribute_by_class([]) == {}
