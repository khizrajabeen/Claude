"""Execution against a real account.

Everything before this class was simulation, so the failures worth
guarding here are the ones that only appear when an order leaves the
machine: a symbol spelled two ways by the same venue, a fee taken in
kind so the quantity bought is not the quantity held, and a 403 that
means "refused" being reported as "bad keys".

All of it runs against a fake session. A test that places a real order
is not a test.
"""

import json

import pytest

from bot.risk.budget import SizedOrder
from bot.trading.alpaca_broker import AlpacaBroker, OrderRejected, LIVE_URL, PAPER_URL


class FakeResponse:
    def __init__(self, payload, status=200, text=""):
        self._payload = payload
        self.status_code = status
        self.text = text or json.dumps(payload)
        self.content = b"x"

    def json(self):
        return self._payload


class Seq:
    """Successive replies for one route.

    A bare list is a perfectly good JSON payload — `/positions` returns
    one — so a sequence of replies needs its own wrapper rather than
    overloading `list`, which silently turned an empty position list
    into an IndexError.
    """

    def __init__(self, *replies):
        self.replies = list(replies)

    def next(self):
        return self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]


class FakeSession:
    """Routes canned payloads by (method, path) and records every call."""

    def __init__(self, routes=None):
        self.routes = routes or {}
        self.calls = []
        self.headers = {}

    def request(self, method, url, **kw):
        path = url.split("/v2", 1)[1]
        self.calls.append((method, path, kw.get("json")))
        # Longest prefix wins, so /orders/{id} beats /orders.
        best = None
        for (m, prefix), payload in self.routes.items():
            if m == method and path.startswith(prefix):
                if best is None or len(prefix) > len(best[0]):
                    best = (prefix, payload)
        if best is None:
            return FakeResponse({}, 404, "no route")
        payload = best[1]
        if isinstance(payload, Seq):
            payload = payload.next()
        if isinstance(payload, FakeResponse):
            return payload
        return FakeResponse(payload)

    def get(self, url, **kw):
        return self.request("GET", url, **kw)


ACCOUNT = {"equity": "100000", "cash": "99000", "buying_power": "396000",
           "account_number": "PA30PX4UK01N", "status": "ACTIVE",
           "currency": "USD", "last_equity": "99500",
           "trading_blocked": False, "shorting_enabled": True}

ASSETS = [{"symbol": "BTC/USD", "tradable": True},
          {"symbol": "ETH/USD", "tradable": True},
          {"symbol": "AAPL", "tradable": True},
          {"symbol": "DELISTED/USD", "tradable": False}]


def broker(routes=None, config=None):
    routes = {("GET", "/account"): ACCOUNT,
              ("GET", "/assets"): ASSETS,
              ("GET", "/positions"): [],
              ("GET", "/orders"): [],
              **(routes or {})}
    b = AlpacaBroker(config or {}, session=FakeSession(routes))
    b.key, b.secret = "PKTEST", "secret"
    return b


# ── Symbols ──────────────────────────────────────────────────

@pytest.mark.parametrize("ours, theirs", [
    ("BTC/USDT", "BTC/USD"),
    ("ETH/USDC", "ETH/USD"),
    ("SOL/USDT:USDT", "SOL/USD"),      # a perp's spot leg
    ("AAPL", "AAPL"),
    ("BTC/USD", "BTC/USD"),
])
def test_venue_symbol_maps_our_spelling_to_theirs(ours, theirs):
    assert AlpacaBroker.venue_symbol(ours) == theirs


def test_positions_and_orders_spell_the_same_asset_differently():
    """Orders take BTC/USD; the positions endpoint returns BTCUSD.

    Compared as strings, every crypto position reads as both missing
    from the ledger and unexpected at the venue.
    """
    assert AlpacaBroker._bare("BTC/USD") == AlpacaBroker._bare("BTCUSD")


def test_untradable_symbols_are_refused_before_an_order_is_sent():
    b = broker()
    assert b.can_trade("BTC/USDT")
    assert not b.can_trade("NEAR/USDT")       # real: not listed on Alpaca
    order = SizedOrder(symbol="NEAR/USDT", side="long", quantity=1.0,
                       entry_price=5.0, stop_price=4.5, take_profit=6.0,
                       risk_usd=10, notional=5.0, leverage=1.0, atr=0.1,
                       r_distance=0.5, asset_class="crypto_spot")
    with pytest.raises(OrderRejected, match="not.*tradable"):
        b.open(order)
    assert not any(m == "POST" for m, _, _ in b.session().calls), \
        "an untradable symbol must not reach the venue"


# ── Endpoints ────────────────────────────────────────────────

def test_paper_is_the_default_and_live_is_opt_in(monkeypatch):
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    assert AlpacaBroker({}).base == PAPER_URL
    monkeypatch.setenv("ALPACA_PAPER", "false")
    assert AlpacaBroker({}).base == LIVE_URL
    monkeypatch.setenv("ALPACA_PAPER", "true")
    assert AlpacaBroker({}).base == PAPER_URL


def test_a_403_on_an_order_is_a_refusal_not_an_auth_failure():
    """The same status means two different things by route.

    Reporting a refused order as bad credentials sends the operator off
    to rotate keys that were never the problem.
    """
    b = broker({("POST", "/orders"): FakeResponse(
        {}, 403, "insufficient balance for BTC/USD")})
    with pytest.raises(OrderRejected) as e:
        b._submit("BTC/USDT", "sell", 1.0)
    assert "refused" in str(e.value)
    assert "keys" not in str(e.value)


def test_a_403_on_a_read_still_points_at_the_keys():
    b = broker({("GET", "/account"): FakeResponse({}, 403, "forbidden")})
    with pytest.raises(OrderRejected, match="keys"):
        b.account()


# ── Fills ────────────────────────────────────────────────────

def filled(qty, price, oid="o1"):
    return {"id": oid, "status": "filled",
            "filled_qty": str(qty), "filled_avg_price": str(price)}


def test_the_fee_is_taken_in_kind_so_the_ledger_books_what_arrived():
    """Alpaca's crypto commission comes out of the coin, not the cash.

    A filled order for 0.003003 BTC leaves 0.00299549 in the account.
    Booking the bought quantity leaves the ledger trying to sell more
    than it owns, which the venue rejects outright.
    """
    b = broker({
        ("POST", "/orders"): filled(0.003003, 83272.29),
        ("GET", "/orders/"): filled(0.003003, 83272.29),
        ("GET", "/positions"): Seq([], [{"symbol": "BTCUSD", "qty": "0.00299549",
                                         "qty_available": "0.00299549"}]),
    })
    order = SizedOrder(symbol="BTC/USDT", side="long", quantity=0.003003,
                       entry_price=83000.0, stop_price=80000.0, take_profit=88000.0,
                       risk_usd=9.0, notional=250.0, leverage=1.0,
                       atr=800.0, r_distance=3000.0, asset_class="crypto_spot")
    position = b.open(order, day="2026-09-24", asset_class="crypto_spot")

    assert position.quantity == pytest.approx(0.00299549), \
        "the ledger must hold what the account holds, not what was ordered"
    assert position.entry_price == pytest.approx(83272.29), \
        "the venue's fill, not the price the signal expected"
    assert position.entry_fee > 0, "the in-kind commission still has a dollar value"
    assert position.tags["alpaca_order_id"] == "o1"


def test_an_unfilled_order_is_cancelled_rather_than_left_working():
    """An order that fills after the bot gave up is an unmanaged position."""
    b = broker({("POST", "/orders"): {"id": "o9", "status": "new"},
                ("GET", "/orders/"): {"id": "o9", "status": "new"}})
    b.fill_timeout = 0.01
    with pytest.raises(OrderRejected, match="no fill"):
        b._submit("BTC/USDT", "buy", 1.0)
    assert ("DELETE", "/orders/o9", None) in b.session().calls


@pytest.mark.parametrize("status", ["canceled", "rejected", "expired"])
def test_a_dead_order_raises_instead_of_spinning(status):
    b = broker({("POST", "/orders"): {"id": "o2", "status": status},
                ("GET", "/orders/"): {"id": "o2", "status": status}})
    with pytest.raises(OrderRejected, match=status):
        b._submit("BTC/USDT", "buy", 1.0)


def test_crypto_orders_are_gtc_and_equities_are_day():
    """An equity `gtc` survives the session and fills tomorrow, at a price
    the signal never saw. Crypto cannot use `day` at all."""
    b = broker({("POST", "/orders"): filled(1, 100),
                ("GET", "/orders/"): filled(1, 100)})
    b._submit("BTC/USDT", "buy", 1.0)
    b._submit("AAPL", "buy", 1.0)
    posts = [body for m, p, body in b.session().calls if m == "POST"]
    assert posts[0]["time_in_force"] == "gtc"
    assert posts[0]["symbol"] == "BTC/USD"
    assert posts[1]["time_in_force"] == "day"


# ── Account truth ────────────────────────────────────────────

def test_equity_comes_from_the_account_not_the_local_ledger():
    """Sizing off a simulated balance while trading a real one is how a
    bot quietly takes twice the risk it believes it is taking."""
    b = broker()
    b.cash = 1.0                       # a ledger that has drifted badly
    assert b.equity({}) == pytest.approx(100_000.0)


def test_equity_falls_back_to_the_ledger_when_the_venue_cannot_be_reached():
    b = broker({("GET", "/account"): FakeResponse({}, 500, "boom")})
    assert b.equity({}) > 0            # did not raise


def test_reconcile_reports_drift_rather_than_papering_over_it():
    b = broker({("GET", "/positions"): [{"symbol": "BTCUSD", "qty": "0.5"}]})
    notes = b.reconcile()
    assert any("BTCUSD" in n for n in notes)
    assert "0.5" in " ".join(notes)


def test_reconcile_is_quiet_when_the_ledger_matches():
    from datetime import datetime, timezone
    from bot.trading.models import Position
    b = broker({("GET", "/positions"): [{"symbol": "BTCUSD", "qty": "0.5"}]})
    b.positions = [Position(id=1, symbol="BTC/USDT", side="long", entry_price=80000.0,
                            quantity=0.5, leverage=1.0,
                            opened_at=datetime.now(timezone.utc),
                            stop_price=78000.0, take_profit=84000.0,
                            initial_stop=78000.0, risk_usd=10.0, margin=100.0,
                            entry_fee=0.0)]
    assert b.reconcile() == []


def test_the_published_snapshot_masks_the_account_number():
    """The snapshot is published to a public page."""
    b = broker()
    snap = b.snapshot()
    assert "PA30PX4UK01N" not in json.dumps(snap)
    assert snap["account_number"].endswith("K01N")
    assert snap["mode"] == "paper"
