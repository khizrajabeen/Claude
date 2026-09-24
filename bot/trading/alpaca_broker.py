"""Execution against a real Alpaca account.

Everything before this module was simulation. `PaperBroker` models fills,
fees, margin and funding; it never told a venue anything. This subclass
keeps all of that bookkeeping — the risk engine, the position ledger, the
trade journal — and replaces exactly one thing: where the fill price
comes from.

The seam is `_fill_price`. `PaperBroker.open` and `.close` both call it to
turn an intended price into an achieved one. Here the order is submitted
first, the venue's own average fill is stashed, and `_fill_price` hands
that back. Nothing else in the lifecycle has to know that the trade was
real, which is why this is a subclass rather than a parallel
implementation that would drift from the original within a month.

Alpaca is one account for both asset classes: US equities and ETFs, and
73 crypto pairs, all fractionable. That is unusual and worth using — it
means one balance, one margin pool and one reconciliation point instead
of two.

Two honesty rules are enforced here rather than left to the caller:

  * A symbol Alpaca cannot trade is refused at submission, not silently
    booked. NEAR and SUI are in the bot's analysis universe and are not
    on Alpaca; a position the venue never took is worse than a skipped
    trade, because it holds risk budget that does not exist.
  * Equity, not the local cash ledger, is read from the account. When
    simulation and reality disagree about the balance, reality wins.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import replace
from datetime import datetime, timezone

from bot.trading.broker import PaperBroker, InsufficientFunds
from bot.trading.models import Position, Trade

logger = logging.getLogger(__name__)

PAPER_URL = "https://paper-api.alpaca.markets/v2"
LIVE_URL = "https://api.alpaca.markets/v2"

# Alpaca quotes crypto against USD; the bot's universe is written against
# USDT because that is what the crypto venues list. The two are not the
# same asset, but for sizing and signalling at this horizon the
# difference is inside the spread.
_QUOTE_SWAP = {"USDT": "USD", "USDC": "USD"}


class OrderRejected(Exception):
    """The venue refused the order, or it did not fill in time."""


class AlpacaBroker(PaperBroker):
    """PaperBroker's ledger, Alpaca's fills."""

    def __init__(self, config: dict, cash: float | None = None,
                 positions: list[Position] | None = None,
                 trade_counter: int = 0, session=None):
        super().__init__(config, cash=cash, positions=positions,
                         trade_counter=trade_counter)
        alpaca = (config.get("alpaca") or {})
        self.key = os.environ.get("ALPACA_API_KEY_ID") or alpaca.get("key_id", "")
        self.secret = (os.environ.get("ALPACA_API_SECRET_KEY")
                       or alpaca.get("secret_key", ""))
        paper_env = os.environ.get("ALPACA_PAPER")
        self.paper = (paper_env.lower() != "false" if paper_env is not None
                      else bool(alpaca.get("paper", True)))
        self.base = PAPER_URL if self.paper else LIVE_URL
        self.timeout = float(alpaca.get("timeout_seconds", 20))
        self.fill_timeout = float(alpaca.get("fill_timeout_seconds", 45))
        self._session = session
        self._pending_fill: tuple[float, float] | None = None
        self._tradable: set[str] | None = None

        # Alpaca charges no commission on US equities. Crypto is a real
        # fee and the simulated default would understate it.
        paper_cfg = config.get("paper", {})
        self.equity_fee_bps = float(alpaca.get("equity_fee_bps", 0.0))
        self.crypto_fee_bps = float(
            alpaca.get("crypto_fee_bps", paper_cfg.get("fee_taker_bps", 25.0)))

    # ── Plumbing ──────────────────────────────────────────────

    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)

    def session(self):
        if self._session is None:
            import requests
            s = requests.Session()
            s.headers.update({
                "APCA-API-KEY-ID": self.key,
                "APCA-API-SECRET-KEY": self.secret,
                "accept": "application/json",
            })
            bundle = os.environ.get("REQUESTS_CA_BUNDLE") or "/root/.ccr/ca-bundle.crt"
            if os.path.exists(bundle):
                s.verify = bundle
            self._session = s
        return self._session

    def _request(self, method: str, path: str, **kw):
        if not self.configured:
            raise OrderRejected(
                "Alpaca keys are not set — export ALPACA_API_KEY_ID and "
                "ALPACA_API_SECRET_KEY before trading"
            )
        r = self.session().request(method, f"{self.base}{path}",
                                   timeout=self.timeout, **kw)
        if r.status_code == 403:
            # 403 means two different things here. On a read it is bad
            # credentials; on an order it is the venue refusing — not
            # enough buying power, or not enough of the asset to sell.
            # Reporting the first when it is the second sends you to
            # rotate perfectly good keys.
            if method == "POST" and path.startswith("/orders"):
                raise OrderRejected(f"order refused: {r.text[:300]}")
            raise OrderRejected(
                f"Alpaca refused {method} {path} (403) — check the keys match "
                f"the {'paper' if self.paper else 'live'} endpoint"
            )
        if r.status_code >= 400:
            raise OrderRejected(f"{method} {path} -> {r.status_code}: {r.text[:300]}")
        return r.json() if r.content else {}

    # ── Symbols ───────────────────────────────────────────────

    @staticmethod
    def venue_symbol(symbol: str) -> str:
        """The bot's symbol as Alpaca spells it.

        BTC/USDT -> BTC/USD, SOL/USDT:USDT -> SOL/USD, AAPL -> AAPL.
        The `:USDT` suffix marks a perpetual, which Alpaca does not list;
        the caller decides whether to route the spot leg instead, but the
        spelling is handled here so every call site agrees.
        """
        base = symbol.split(":")[0]
        if "/" not in base:
            return base
        left, right = base.split("/", 1)
        return f"{left}/{_QUOTE_SWAP.get(right, right)}"

    @staticmethod
    def _bare(symbol: str) -> str:
        """Alpaca's two spellings of one instrument, reduced to one.

        Orders take `BTC/USD`; the positions endpoint returns `BTCUSD`.
        Comparing the two as strings reports every crypto position as
        both missing and unexpected.
        """
        return symbol.replace("/", "").upper()

    def position_qty(self, symbol: str) -> float:
        """How much of this the account actually holds, right now.

        Not what the ledger believes. Alpaca takes its crypto fee in
        kind, so a filled order for 0.003003 BTC leaves 0.00299549 in
        the account, and selling the amount that was bought is an
        oversell the venue rejects.
        """
        want = self._bare(self.venue_symbol(symbol))
        try:
            for p in self.venue_positions():
                if self._bare(p["symbol"]) == want:
                    return float(p.get("qty_available") or p["qty"])
        except Exception as e:
            logger.warning("Could not read the position for %s: %s", symbol, e)
        return 0.0

    def tradable(self) -> set[str]:
        """Symbols this account can actually trade, fetched once."""
        if self._tradable is None:
            out: set[str] = set()
            for klass in ("us_equity", "crypto"):
                try:
                    for a in self._request(
                            "GET", f"/assets?asset_class={klass}&status=active"):
                        if a.get("tradable"):
                            out.add(a["symbol"])
                except Exception as e:      # a listing failure is not fatal
                    logger.warning("Could not list %s assets: %s", klass, e)
            self._tradable = out
        return self._tradable

    def can_trade(self, symbol: str) -> bool:
        known = self.tradable()
        return not known or self.venue_symbol(symbol) in known

    # ── Account truth ─────────────────────────────────────────

    def account(self) -> dict:
        return self._request("GET", "/account")

    def equity(self, prices: dict[str, float]) -> float:
        """Account equity as the venue reports it.

        When the local ledger and the account disagree, the account is
        right. Sizing off a simulated balance while trading a real one is
        how a bot quietly takes twice the risk it thinks it is taking.
        """
        try:
            return float(self.account()["equity"])
        except Exception as e:
            logger.warning("Falling back to the local ledger for equity: %s", e)
            return super().equity(prices)

    def free_cash(self) -> float:
        try:
            a = self.account()
            return min(float(a["cash"]), float(a["buying_power"]))
        except Exception as e:
            logger.warning("Falling back to the local ledger for cash: %s", e)
            return super().free_cash()

    def venue_positions(self) -> list[dict]:
        return self._request("GET", "/positions")

    def open_orders(self) -> list[dict]:
        return self._request("GET", "/orders?status=open&limit=100")

    def activities(self, limit: int = 50) -> list[dict]:
        try:
            return self._request("GET", f"/account/activities?page_size={limit}")
        except Exception as e:
            logger.debug("activities unavailable: %s", e)
            return []

    def clock(self) -> dict:
        return self._request("GET", "/clock")

    # ── Execution ─────────────────────────────────────────────

    def _submit(self, symbol: str, side: str, quantity: float) -> dict:
        """Place a market order and wait for it to fill.

        Market, not limit: the strategies size by ATR and act on the
        close of a bar, so certainty of execution is worth more than a
        few basis points of price improvement. A limit order that does
        not fill leaves the risk model believing in a position that does
        not exist, which is the failure this whole module exists to stop.
        """
        venue = self.venue_symbol(symbol)
        crypto = "/" in venue
        payload = {
            "symbol": venue,
            "qty": str(quantity),
            "side": side,
            "type": "market",
            # Crypto trades around the clock and cannot use `day`; an
            # equity `gtc` would survive the session and fill tomorrow at
            # a price the signal never saw.
            "time_in_force": "gtc" if crypto else "day",
        }
        order = self._request("POST", "/orders", json=payload)
        return self._await_fill(order["id"], venue)

    def _await_fill(self, order_id: str, venue: str) -> dict:
        deadline = time.monotonic() + self.fill_timeout
        delay = 0.4
        while True:
            order = self._request("GET", f"/orders/{order_id}")
            status = order.get("status")
            if status == "filled":
                return order
            if status in ("canceled", "expired", "rejected", "suspended"):
                raise OrderRejected(
                    f"{venue}: order {status}"
                    + (f" — {order.get('reason')}" if order.get("reason") else ""))
            if time.monotonic() > deadline:
                # Do not leave it working: an order that fills after the
                # bot has given up is an unmanaged position.
                try:
                    self._request("DELETE", f"/orders/{order_id}")
                except Exception:
                    pass
                raise OrderRejected(
                    f"{venue}: no fill within {self.fill_timeout:.0f}s "
                    f"(status {status}) — order cancelled")
            time.sleep(delay)
            delay = min(delay * 1.6, 3.0)

    def _fill_price(self, price, side, notional, entering,
                    adv_notional=None, spread_bps=None, daily_vol_bps=None):
        """Return the venue's own fill instead of a modelled one."""
        if self._pending_fill is not None:
            fill, slip = self._pending_fill
            return fill, slip
        return super()._fill_price(price, side, notional, entering,
                                   adv_notional, spread_bps, daily_vol_bps)

    def open(self, order, now=None, day="", **kw) -> Position:
        if not self.can_trade(order.symbol):
            raise OrderRejected(
                f"{order.symbol} ({self.venue_symbol(order.symbol)}) is not "
                f"tradable on this Alpaca account")

        side = "buy" if order.side == "long" else "sell"
        before = self.position_qty(order.symbol)
        filled = self._submit(order.symbol, side, order.quantity)
        avg = float(filled["filled_avg_price"])
        asked = float(filled.get("filled_qty") or order.quantity)

        # What the account actually gained, which is not what was bought.
        # Alpaca takes its crypto commission in kind, so a filled order
        # for 0.003003 BTC leaves 0.00299549 in the account. Booking the
        # bought quantity and ALSO charging a cash fee counts the same
        # commission twice and leaves the ledger unable to sell what it
        # thinks it owns.
        after = self.position_qty(order.symbol)
        net = abs(after - before) or asked
        fee_value = max(0.0, (asked - net)) * avg
        self.taker_bps = (fee_value / (net * avg) * 10_000) if net and avg else 0.0

        self._pending_fill = (avg, abs(avg - order.entry_price)
                              / order.entry_price * 10_000)
        try:
            if abs(net - order.quantity) > 1e-12:
                order = replace(order, quantity=net)
            position = super().open(order, now=now, day=day, **kw)
        finally:
            self._pending_fill = None

        position.tags = {**(position.tags or {}),
                         "alpaca_order_id": filled["id"],
                         "alpaca_symbol": self.venue_symbol(order.symbol)}
        logger.info("FILLED %s %.8f %s @ %.6f (fee $%.4f, order %s)",
                    side, net, order.symbol, avg, fee_value, filled["id"])
        return position

    def close(self, position: Position, price: float, reason: str = "signal",
              now=None, day="", **kw) -> Trade:
        side = "sell" if position.side == "long" else "buy"

        # Sell what is there, not what the ledger remembers. The gap is
        # the in-kind fee, and asking for the larger number is an
        # oversell the venue refuses outright.
        available = self.position_qty(position.symbol)
        quantity = position.quantity
        if side == "sell" and available > 0:
            quantity = min(quantity, available)
        if quantity <= 0:
            raise OrderRejected(
                f"{position.symbol}: nothing available to close "
                f"(ledger {position.quantity:g}, account {available:g})")

        filled = self._submit(position.symbol, side, quantity)
        avg = float(filled["filled_avg_price"])
        self.taker_bps = 0.0      # the exit fee is inside the proceeds
        self._pending_fill = (avg, abs(avg - price) / price * 10_000 if price else 0.0)
        try:
            if abs(quantity - position.quantity) > 1e-12:
                position.quantity = quantity
            trade = super().close(position, price, reason=reason,
                                  now=now, day=day, **kw)
        finally:
            self._pending_fill = None
        logger.info("CLOSED %.8f %s @ %.6f (%s, order %s)",
                    quantity, position.symbol, avg, reason, filled["id"])
        return trade

    # ── Reconciliation ────────────────────────────────────────

    def reconcile(self) -> list[str]:
        """Compare the local ledger with the account, and report drift.

        Called at the start of a session. It does not silently rewrite
        the ledger: a mismatch means something happened outside the bot
        — a manual trade, a partial fill, a corporate action — and the
        operator should see it rather than have it papered over.
        """
        notes: list[str] = []
        try:
            venue = {self._bare(p["symbol"]): float(p["qty"])
                     for p in self.venue_positions()}
        except Exception as e:
            return [f"could not read positions from Alpaca: {e}"]

        local: dict[str, float] = {}
        for p in self.positions:
            key = self._bare(self.venue_symbol(p.symbol))
            local[key] = local.get(key, 0.0) + p.quantity * p.direction

        for sym in sorted(set(venue) | set(local)):
            there, here = venue.get(sym, 0.0), local.get(sym, 0.0)
            if abs(there - here) > max(1e-6, abs(here) * 1e-4):
                notes.append(f"{sym}: account {there:g}, ledger {here:g}")
        return notes

    def snapshot(self) -> dict:
        """Everything the dashboard needs about the live account."""
        acct = self.account()
        try:
            market = self.clock()
        except Exception:
            market = {}
        return {
            "as_of": datetime.now(timezone.utc).isoformat(),
            "mode": "paper" if self.paper else "live",
            # Masked. This snapshot is published to a public page, and
            # an account number is an identifier worth not handing out
            # even when it cannot be used on its own. The last four are
            # enough to tell two accounts apart.
            "account_number": "••••" + (acct.get("account_number") or "")[-4:],
            "currency": acct.get("currency", "USD"),
            "equity": float(acct.get("equity", 0)),
            "last_equity": float(acct.get("last_equity", 0)),
            "cash": float(acct.get("cash", 0)),
            "buying_power": float(acct.get("buying_power", 0)),
            "status": acct.get("status", ""),
            "trading_blocked": bool(acct.get("trading_blocked")),
            "shorting_enabled": bool(acct.get("shorting_enabled")),
            "market_open": bool(market.get("is_open")),
            "next_open": market.get("next_open", ""),
            "next_close": market.get("next_close", ""),
            "positions": [
                {
                    "symbol": p["symbol"],
                    "qty": float(p["qty"]),
                    "side": p["side"],
                    "avg_entry_price": float(p["avg_entry_price"]),
                    "current_price": float(p.get("current_price") or 0),
                    "market_value": float(p.get("market_value") or 0),
                    "unrealized_pl": float(p.get("unrealized_pl") or 0),
                    "unrealized_plpc": float(p.get("unrealized_plpc") or 0) * 100,
                    "asset_class": p.get("asset_class", ""),
                }
                for p in self.venue_positions()
            ],
            "orders": [
                {
                    "symbol": o["symbol"], "side": o["side"], "qty": o.get("qty"),
                    "type": o.get("order_type") or o.get("type"),
                    "status": o.get("status"),
                    "submitted_at": o.get("submitted_at", ""),
                    "limit_price": o.get("limit_price"),
                }
                for o in self.open_orders()
            ],
        }
