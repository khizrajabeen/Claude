"""The Alpaca connector.

It sits ahead of the daily Nasdaq feed and declines when no keys are set,
so it is an upgrade rather than a new requirement. The failures worth
guarding are the quiet ones: keys pointed at the wrong endpoint, a deep
request silently returning one page, and live trading reached by
accident.
"""

import json
from datetime import datetime, timezone

import pandas as pd
import pytest

from bot.data.alpaca import LIVE_URL, PAPER_URL, AlpacaProvider
from bot.markets.instrument import build_instrument

STOCK = build_instrument({"symbol": "NVDA", "asset_class": "equity"})
ETF = build_instrument({"symbol": "SPY", "asset_class": "etf"})
COIN = build_instrument({"symbol": "BTC/USDT", "asset_class": "crypto_spot"})


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    """Serves canned payloads and records what was asked for."""

    def __init__(self, pages=None, payload=None, status=200):
        self.pages = list(pages or [])
        self.payload = payload or {}
        self.status = status
        self.calls = []
        self.headers = {}

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        if self.pages:
            return FakeResponse(self.pages.pop(0), self.status)
        return FakeResponse(self.payload, self.status)


def bars_payload(n, start_day=1, token=None):
    rows = [{
        "t": f"2026-09-{start_day + i:02d}T00:00:00Z",
        "o": 100 + i, "h": 101 + i, "l": 99 + i, "c": 100.5 + i, "v": 1000 + i,
    } for i in range(n)]
    out = {"bars": rows}
    if token:
        out["next_page_token"] = token
    return out


def provider(config, session=None, **alpaca):
    config["alpaca"] = {"key_id": "PKTEST", "secret_key": "SECRET", **alpaca}
    return AlpacaProvider(config, session=session)


# ── Configuration ────────────────────────────────────────────

def test_it_declines_when_no_keys_are_set(config, monkeypatch):
    for var in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    config["alpaca"] = {}
    p = AlpacaProvider(config)
    assert not p.configured
    assert not p.handles(STOCK), "must fall through to the daily feed"


def test_it_handles_equities_once_configured(config):
    p = provider(config)
    assert p.handles(STOCK)
    assert p.handles(ETF)


def test_it_never_claims_crypto(config):
    assert not provider(config).handles(COIN)


def test_paper_is_the_default(config):
    """The difference between endpoints is one string and the consequence
    of getting it wrong is real money."""
    assert provider(config).paper
    assert provider(config).trading_url == PAPER_URL


def test_live_is_opt_in(config):
    assert provider(config, paper=False).trading_url == LIVE_URL


def test_the_environment_overrides_the_config(config, monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY_ID", "PKENV")
    monkeypatch.setenv("ALPACA_API_SECRET_KEY", "ENVSECRET")
    monkeypatch.setenv("ALPACA_PAPER", "false")
    p = provider(config)
    assert p.key == "PKENV" and p.secret == "ENVSECRET"
    assert not p.paper


def test_an_unconfigured_request_explains_itself(config, monkeypatch):
    from bot.data.base import ProviderError
    for var in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    config["alpaca"] = {}
    with pytest.raises(ProviderError, match="ALPACA_API_KEY_ID"):
        AlpacaProvider(config, session=FakeSession()).bars(STOCK, "1d", 10)


def test_a_403_names_the_likely_cause(config):
    from bot.data.base import ProviderError
    session = FakeSession(payload={}, status=403)
    with pytest.raises(ProviderError, match="paper"):
        provider(config, session=session).bars(STOCK, "1d", 10)


# ── Bars ─────────────────────────────────────────────────────

def test_bars_come_back_as_an_ohlcv_frame(config):
    session = FakeSession(payload=bars_payload(5))
    frame = provider(config, session=session).bars(STOCK, "1d", 5)
    assert list(frame.columns) == ["open", "high", "low", "close", "volume"]
    assert len(frame) == 5
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert frame.index.tz is not None


def test_bars_are_sorted_oldest_first(config):
    session = FakeSession(payload=bars_payload(6))
    frame = provider(config, session=session).bars(STOCK, "1d", 6)
    assert frame.index.is_monotonic_increasing


def test_a_deep_request_follows_the_pages(config):
    """Alpaca caps a page at 10,000 bars; stopping at the first would
    quietly return a fraction of the history asked for."""
    session = FakeSession(pages=[
        bars_payload(3, start_day=1, token="next"),
        bars_payload(3, start_day=4),
    ])
    frame = provider(config, session=session).bars(STOCK, "1d", 6)
    assert len(frame) == 6
    assert len(session.calls) == 2
    assert session.calls[1][1]["page_token"] == "next"


def test_paging_stops_when_the_venue_runs_out(config):
    session = FakeSession(pages=[bars_payload(2)])
    frame = provider(config, session=session).bars(STOCK, "1d", 500)
    assert len(frame) == 2


def test_no_bars_yields_an_empty_frame_not_an_error(config):
    session = FakeSession(payload={"bars": []})
    assert provider(config, session=session).bars(STOCK, "1d", 10).empty


def test_the_timeframe_is_translated_to_the_venues_name(config):
    session = FakeSession(payload=bars_payload(2))
    provider(config, session=session).bars(STOCK, "15m", 2)
    assert session.calls[0][1]["timeframe"] == "15Min"


def test_an_unknown_timeframe_falls_back_to_daily(config):
    session = FakeSession(payload=bars_payload(2))
    provider(config, session=session).bars(STOCK, "3mo", 2)
    assert session.calls[0][1]["timeframe"] == "1Day"


def test_bars_are_split_adjusted(config):
    """Unadjusted prices put a false gap in every series that ever
    split."""
    session = FakeSession(payload=bars_payload(2))
    provider(config, session=session).bars(STOCK, "1d", 2)
    assert session.calls[0][1]["adjustment"] == "split"


# ── Quotes and prices ────────────────────────────────────────

def test_the_latest_trade_is_the_price(config):
    session = FakeSession(payload={"trade": {"p": 227.38}})
    assert provider(config, session=session).price(STOCK) == pytest.approx(227.38)


def test_a_failed_price_is_none_not_an_exception(config):
    """One dead symbol must not take down the day."""
    session = FakeSession(payload={}, status=500)
    assert provider(config, session=session).price(STOCK) is None


def test_the_spread_is_computed_from_the_quote(config):
    session = FakeSession(payload={"quote": {"bp": 99.95, "ap": 100.05}})
    spread = provider(config, session=session).spread_bps(STOCK)
    assert spread == pytest.approx(10.0, abs=0.1)


def test_a_one_sided_quote_yields_no_spread(config):
    session = FakeSession(payload={"quote": {"bp": 99.95}})
    assert provider(config, session=session).spread_bps(STOCK) is None


def test_equities_have_no_funding_leg(config):
    assert provider(config).funding_rate(STOCK) is None


def test_orders_are_whole_shares(config):
    """Fractional positions cannot be shorted and the sizing does not
    model that."""
    limits = provider(config).market_limits(STOCK)
    assert limits["qty_step"] == 1.0
    assert limits["min_qty"] == 1.0


# ── Account ──────────────────────────────────────────────────

def test_the_account_reports_equity_and_which_endpoint_it_came_from(config):
    session = FakeSession(payload={
        "equity": "10500.25", "cash": "4000.00", "buying_power": "8000",
        "currency": "USD", "trading_blocked": False,
    })
    account = provider(config, session=session).account()
    assert account["equity"] == pytest.approx(10500.25)
    assert account["paper"] is True
    assert account["blocked"] is False


def test_positions_are_normalised_to_the_bots_shape(config):
    session = FakeSession(payload=[{
        "symbol": "NVDA", "qty": "-5", "avg_entry_price": "220.00",
        "market_value": "-1100", "unrealized_pl": "25.5",
    }])
    rows = provider(config, session=session).positions()
    assert rows[0]["side"] == "short"
    assert rows[0]["quantity"] == pytest.approx(-5.0)


def test_the_venue_is_asked_whether_the_market_is_open(config):
    """A holiday table goes stale; the venue does not."""
    session = FakeSession(payload={"is_open": True, "next_close": "..."})
    assert provider(config, session=session).clock()["is_open"] is True


# ── Routing ──────────────────────────────────────────────────

def test_the_router_prefers_alpaca_but_keeps_the_fallback(config):
    from bot.data import DataRouter

    names = [type(p).__name__ for p in DataRouter(config).providers]
    assert names.index("AlpacaProvider") < names.index("EquityProvider")


def test_the_router_falls_through_when_alpaca_is_unconfigured(config, monkeypatch):
    from bot.data import DataRouter

    for var in ("ALPACA_API_KEY_ID", "ALPACA_API_SECRET_KEY"):
        monkeypatch.delenv(var, raising=False)
    config["alpaca"] = {}
    chosen = DataRouter(config).provider_for(STOCK)
    assert type(chosen).__name__ == "EquityProvider"
