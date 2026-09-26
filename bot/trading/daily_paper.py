"""Bounded daily strategy cycle; orders can ONLY reach Alpaca paper.

No simulated clock, no inherited simulated broker fills. Venue positions
and cumulative order fills are authoritative. Persist intent before POST;
recover by stable client_order_id after timeouts. Never adopt old holdings.
"""
import argparse
import hashlib
import json
import os
from decimal import Decimal, ROUND_DOWN
from pathlib import Path

import pandas as pd
import requests

from bot.strategies.daily_candidates import VERSION, completed_bars, exposure
from bot.utils.daily_research import SYMBOLS, fetch, frames_from

PAPER_URL = "https://paper-api.alpaca.markets/v2"
PREFIX = "mdaily1"
TERMINAL = {"filled", "canceled", "expired", "rejected"}


def atomic_write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True))
    os.replace(tmp, path)


def record(directory, kind, payload):
    directory.mkdir(parents=True, exist_ok=True)
    item = dict(at=pd.Timestamp.now(tz="UTC").isoformat(), kind=kind, **payload)
    # Each cycle retains its own append-only event file, including failures.
    with (directory / "events.jsonl").open("a") as f:
        f.write(json.dumps(item, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


class PaperAPI:
    def __init__(self, session=None):
        if os.getenv("ALPACA_PAPER", "true").lower() != "true":
            raise ValueError("This runner only permits ALPACA_PAPER=true")
        key, secret = os.getenv("ALPACA_API_KEY_ID"), os.getenv("ALPACA_API_SECRET_KEY")
        if not key or not secret:
            raise ValueError("Alpaca paper credentials are unavailable")
        self.session = session or requests.Session()
        self.session.headers.update({"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})

    def request(self, method, path, **kwargs):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid paper API path")
        r = self.session.request(method, PAPER_URL + path, timeout=20, **kwargs)
        if r.status_code == 404 and method == "GET" and path == "/orders:by_client_order_id":
            return None
        # Never include headers, response bodies or credentials in errors.
        if r.status_code >= 400:
            raise RuntimeError(f"Alpaca paper {method} {path} returned HTTP {r.status_code}")
        return r.json()

    def quote(self, symbol):
        r = self.session.get("https://data.alpaca.markets/v1beta3/crypto/us/latest/quotes",
                             params={"symbols": symbol}, timeout=20)
        r.raise_for_status()
        return r.json()["quotes"][symbol]


def bare(symbol):
    return symbol.replace("/", "")


def order_id(candidate, day, symbol, side):
    digest = hashlib.sha256(f"{VERSION}|{candidate}|{day}|{symbol}|{side}".encode()).hexdigest()[:24]
    return f"{PREFIX}-{digest}"


def run_cycle(api, frames, config, state_path, events, now, submit=False):
    now = pd.Timestamp(now)
    candidate = config["candidate"]
    capital = float(config["capital_usd"])
    allocation = float(config["allocation_per_asset"])
    if not 0 < capital <= 1000 or not 0 < allocation <= 0.20:
        raise ValueError("Paper sleeve is limited to $1000 and 20% per entry")
    signals = {}
    for s in SYMBOLS:
        df = completed_bars(frames[s], now)
        if len(df) < 200 or df.index[-1] != now.normalize() - pd.Timedelta(days=1):
            raise ValueError(f"Missing completed daily history for {s}")
        signals[s] = int(exposure(df, candidate).iloc[-1])
    record(events, "decision", dict(candidate=candidate, version=VERSION, signals=signals,
                                     data_through=str(df.index[-1]), submit=submit))
    account = api.request("GET", "/account")
    if account.get("trading_blocked") or account.get("account_blocked") or account.get("status") != "ACTIVE":
        raise ValueError("Paper account is not active for trading")
    fingerprint = hashlib.sha256(str(account["id"]).encode()).hexdigest()
    signature = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        account=fingerprint, config=signature, owned=[], pending={}, intent_day={}, cash=capital, accounted={})
    if state["account"] != fingerprint or state["config"] != signature:
        raise ValueError("Account/config changed; preserve and review the existing sleeve")
    # Recover all previously submitted/ambiguous intents before making a new decision.
    for cid, intent in list(state["pending"].items()):
        order = api.request("GET", "/orders:by_client_order_id", params={"client_order_id": cid})
        if order is None:
            if state.get("intent_day", {}).get(cid) != str(now.date()):
                record(events, "expired_unsubmitted_intent", dict(client_order_id=cid))
                del state["pending"][cid]
                atomic_write(state_path, state)
                continue
            if not submit:
                return dict(status="unresolved_intent", signals=signals)
            # Same ID, never a fresh one: handles crash between intent and POST.
            order = api.request("POST", "/orders", json=intent)
        qty = float(order.get("filled_qty") or 0)
        price = float(order.get("filled_avg_price") or 0)
        notional = qty * price
        previous = state["accounted"].get(cid, 0.0)
        delta = notional - previous
        # Conservative fee reserve; actual venue fills and positions are logged.
        fee = float(config["fee_bps"]) / 10000
        # Buy fees are charged in received crypto, already reflected in
        # venue quantity; do not subtract the same fee from cash again.
        state["cash"] += delta * (1-fee if intent["side"] == "sell" else -1)
        state["accounted"][cid] = notional
        record(events, "order_observed", dict(client_order_id=cid, status=order["status"],
            filled_qty=qty, filled_avg_price=price, side=intent["side"], symbol=intent["symbol"]))
        if order["status"] in TERMINAL:
            del state["pending"][cid]
        atomic_write(state_path, state)
    if state["pending"]:
        return dict(status="pending_orders", signals=signals)
    positions = api.request("GET", "/positions")
    allowed = {bare(s) for s in state["owned"]}
    for p in positions:
        if bare(p["symbol"]) not in allowed or float(p["qty"]) < 0:
            raise ValueError("Unmanaged holdings exist; this experiment will not adopt or liquidate them")
    open_orders = api.request("GET", "/orders", params={"status": "open"})
    if open_orders:
        raise ValueError("Unresolved venue orders exist; refusing overlapping orders")
    held = {bare(p["symbol"]): p for p in positions}
    sleeve_equity = state["cash"] + sum(float(p["market_value"]) for p in positions)
    record(events, "sleeve", dict(estimated_equity=sleeve_equity,
                                  positions=[dict(symbol=p["symbol"],qty=p["qty"],
                                  market_value=p["market_value"]) for p in positions]))
    actions = []
    for s in SYMBOLS:
        position = held.get(bare(s))
        qty = float(position["qty"]) if position else 0.0
        side = "sell" if qty > 0 and not signals[s] else "buy" if qty == 0 and signals[s] else None
        if side is None:
            continue
        cid = order_id(candidate, str(now.date()), s, side)
        existing = api.request("GET", "/orders:by_client_order_id", params={"client_order_id": cid})
        if existing is not None:
            continue
        quote = api.quote(s)
        bid, ask = float(quote["bp"]), float(quote["ap"])
        age = (now-pd.Timestamp(quote["t"])).total_seconds()
        if min(bid,ask) <= 0 or ask < bid or not 0 <= age <= 300:
            raise ValueError("Invalid or stale paper quote")
        spread = (ask-bid)/((ask+bid)/2)*10000
        if spread > float(config["max_spread_bps"]):
            record(events,"skipped_spread",dict(symbol=s,spread_bps=spread))
            continue
        body = dict(symbol=s,side=side,type="market",time_in_force="gtc",client_order_id=cid)
        if side == "buy":
            budget = min(capital,sleeve_equity)*allocation
            # Never borrow or spend proceeds from an unconfirmed sale.
            budget = min(budget, state["cash"],
                         float(account["cash"])*0.95)
            if budget < 10:
                continue
            body["notional"] = f"{budget:.2f}"
        else:
            available = min(qty,float(position.get("qty_available",qty)))
            if available <= 0:
                continue
            body["qty"] = str(Decimal(str(available)).quantize(Decimal("0.000000001"), rounding=ROUND_DOWN))
            if Decimal(body["qty"]) <= 0:
                continue
        actions.append(body)
        record(events,"order_plan",dict(order=body,bid=bid,ask=ask))
        if submit:
            state["pending"][cid] = body
            state.setdefault("intent_day", {})[cid] = str(now.date())
            if s not in state["owned"]:
                state["owned"].append(s)
            atomic_write(state_path,state)  # durable BEFORE any order leaves
            result = api.request("POST","/orders",json=body)
            record(events,"order_ack",dict(client_order_id=cid,status=result["status"]))
            # One submission per cycle. Next cycle reconciles its actual fill
            # before allocating again; this also handles partial executions.
            break
    return dict(status="submitted" if submit and actions else "planned" if actions else "no_action",
                signals=signals,actions=actions)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--config",default="research/paper_daily.json")
    ap.add_argument("--submit",action="store_true")
    ap.add_argument("--state",default="paper-journal/daily-v1/state.json")
    args=ap.parse_args()
    from dotenv import load_dotenv
    load_dotenv()
    api=PaperAPI()
    now=pd.Timestamp.now(tz="UTC")
    raw=fetch("2021-01-01T00:00:00Z",now.normalize().isoformat())
    frames=frames_from(raw)
    config=json.loads(Path(args.config).read_text())
    state_path=Path(args.state)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (state_path.parent / "cycle.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    events=state_path.parent/"runs"/now.strftime("%Y%m%dT%H%M%S%fZ")
    atomic_write(events/"manifest.json",dict(config=config,version=VERSION,
        git_commit=os.getenv("GITHUB_SHA","local"),
        data_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest()))
    atomic_write(events/"bars.json",raw)
    try:
        result=run_cycle(api,frames,config,state_path,events,now,args.submit)
        atomic_write(events/"result.json",result)
        print(json.dumps(result))
    except Exception as exc:
        record(events,"error",dict(error_type=type(exc).__name__))
        raise


if __name__=="__main__":
    main()
