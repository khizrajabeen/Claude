"""Fixed candidate comparison; no optimization and no same-close fills.

Run: python -m bot.utils.daily_research --out research/daily-v1
Uses Alpaca daily data, caches raw response and records its SHA256.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bot.strategies.daily_candidates import CANDIDATES, VERSION, exposure, validate_bars

URL = "https://data.alpaca.markets/v1beta3/crypto/us/bars"
SYMBOLS = ("BTC/USD", "ETH/USD")


def fetch(start, end):
    headers = {}
    if os.environ.get("ALPACA_API_KEY_ID") and os.environ.get("ALPACA_API_SECRET_KEY"):
        headers = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
                   "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    params = dict(symbols=",".join(SYMBOLS), timeframe="1Day", start=start, end=end, limit=10000)
    bars = {s: [] for s in SYMBOLS}
    tokens = set()
    while True:
        r = requests.get(URL, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        result = r.json()
        for s in SYMBOLS:
            bars[s].extend(result.get("bars", {}).get(s, []))
        token = result.get("next_page_token")
        if not token:
            break
        if token in tokens:
            raise ValueError("Repeated data page")
        tokens.add(token)
        params["page_token"] = token
    return {"bars": bars, "start": start, "end": end, "source": URL}


def frames_from(raw):
    frames = {}
    for symbol in SYMBOLS:
        df = pd.DataFrame(raw["bars"][symbol])
        if df.empty:
            raise ValueError(f"No history for {symbol}")
        df.index = pd.to_datetime(df.pop("t"), utc=True)
        df = df.rename(columns=dict(o="open", h="high", l="low", c="close", v="volume"))
        df = df[["open", "high", "low", "close", "volume"]].sort_index()
        validate_bars(df)
        frames[symbol] = df
    if not frames[SYMBOLS[0]].index.equals(frames[SYMBOLS[1]].index):
        raise ValueError("The two markets have different timestamps")
    return frames


def backtest(frames, candidate, start, end, fee_bps=25, slip_bps=10,
             initial=10000, allocation=0.20):
    """Independent flat-start segments; hold at most one position per asset.

    Exposure at yesterday's close executes at today's open. Each entry
    spends allocation*current equity including fee. Uninvested cash earns
    zero; no shorting, leverage, intrabar stops, adds or partial exits.
    Terminal holdings are liquidated with costs for comparable cash P&L.
    """
    index = frames[SYMBOLS[0]].index
    dates = index[(index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))]
    if len(dates) < 2:
        raise ValueError("Insufficient evaluation dates")
    desired = {s: (exposure(df, candidate).shift(1).fillna(0) if candidate != "buy_hold"
                   else pd.Series(1, index=df.index)) for s, df in frames.items()}
    cash = float(initial)
    holdings = {}
    trades, curve = [], []
    fees = 0.0
    slip = slip_bps / 10000
    fee = fee_bps / 10000
    exposure_days = 0.0

    def sell(symbol, when, price, reason):
        nonlocal cash, fees
        position = holdings.pop(symbol)
        fill = price * (1 - slip)
        commission = position["qty"] * fill * fee
        proceeds = position["qty"] * fill - commission
        cash += proceeds
        fees += commission
        trades.append(dict(symbol=symbol, entry_time=position["date"], exit_time=str(when),
            entry_fill=position["fill"], exit_fill=fill, quantity=position["qty"],
            pnl=proceeds-position["spent"], fees=position["fee"]+commission, reason=reason))

    for dt in dates:
        opening_equity = cash + sum(p["qty"]*frames[s].at[dt,"open"] for s,p in holdings.items())
        for s in list(holdings):
            if not desired[s].at[dt]:
                sell(s, dt, frames[s].at[dt,"open"], "signal")
        for s in SYMBOLS:
            if desired[s].at[dt] and s not in holdings:
                budget = min(cash, allocation*opening_equity)
                fill = frames[s].at[dt,"open"] * (1+slip)
                qty = budget * (1-fee) / fill
                commission = budget*fee
                cash -= budget
                fees += commission
                holdings[s] = dict(qty=qty, spent=budget, fill=fill, fee=commission, date=str(dt))
        market_value = sum(p["qty"]*frames[s].at[dt,"close"] for s,p in holdings.items())
        equity = cash+market_value
        exposure_days += market_value/equity
        curve.append(dict(date=str(dt),equity=equity))
    for s in list(holdings):
        sell(s, dates[-1], frames[s].at[dates[-1],"close"], "end_of_sample")
    curve[-1]["equity"] = cash
    equity = np.r_[initial, [v["equity"] for v in curve]]
    dd = equity / np.maximum.accumulate(equity) - 1
    returns = np.diff(equity)/equity[:-1]
    pnls = [t["pnl"] for t in trades]
    if not np.isclose(sum(pnls),cash-initial,atol=1e-7):
        raise AssertionError("Round-trip ledger does not match cash")
    return dict(candidate=candidate,start=str(dates[0]),end=str(dates[-1]),days=len(dates),
        return_pct=(cash/initial-1)*100,max_drawdown_pct=-float(dd.min())*100,
        trades=len(trades),win_rate=sum(p>0 for p in pnls)/len(pnls) if pnls else 0,
        avg_trade_pnl=float(np.mean(pnls)) if pnls else 0,fees=fees,
        mean_exposure_pct=exposure_days/len(dates)*100,
        sharpe=float(returns.mean()/returns.std()*np.sqrt(365)) if returns.std() else 0,
        fee_bps=fee_bps,slippage_bps=slip_bps,ledger=trades,equity=curve)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out", default="research/daily-v1")
    ap.add_argument("--raw")
    ap.add_argument("--end", help="Exclusive UTC end; required for a legacy raw file without a cutoff")
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    if args.raw:
        raw=json.loads(Path(args.raw).read_text())
        end=args.end or raw.get("evaluation_end") or raw.get("end")
        if not end:
            raise ValueError("Saved data needs its original --end cutoff")
    else:
        end=args.end or pd.Timestamp.now(tz="UTC").normalize().isoformat()
        raw=fetch("2021-01-01T00:00:00Z",end)
    cutoff=pd.Timestamp(end)
    if cutoff.tz is None:
        raise ValueError("End cutoff must be timezone-aware")
    end=cutoff.isoformat()
    raw["evaluation_end"]=end
    raw["bars"]={s:[b for b in raw["bars"][s]
                    if pd.Timestamp(b["t"])+pd.Timedelta(days=1)<=cutoff] for s in SYMBOLS}
    encoded=json.dumps(raw,sort_keys=True).encode()
    (out/"bars.json").write_bytes(encoded)
    frames=frames_from(raw)
    # A declared clock cutoff also protects against APIs returning extra bars.
    frames={s:df.loc[df.index+pd.Timedelta(days=1)<=pd.Timestamp(end)] for s,df in frames.items()}
    windows={"development":("2022-01-01T00:00:00Z","2024-01-01T00:00:00Z"),
             "holdout":("2024-01-01T00:00:00Z","2026-01-01T00:00:00Z"),
             "recent":("2026-01-01T00:00:00Z",end)}
    results=[]
    for window,(start,stop) in windows.items():
        for candidate in (*CANDIDATES,"buy_hold"):
            for cost in (10,25):
                r=backtest(frames,candidate,start,stop,slip_bps=cost)
                r["window"]=window
                stem=f"{window}-{candidate}-{cost}bps"
                (out/f"{stem}.json").write_text(json.dumps(r,indent=2))
                results.append({k:v for k,v in r.items() if k not in ("ledger","equity")})
    report=dict(version=VERSION,source=URL,data_sha256=hashlib.sha256(encoded).hexdigest(),
        assumptions="20% equity per new entry per asset; 25bps fee each side; next-open execution; cash earns zero; long-only",
        limitations="Daily fills omit intraday path, order queues and venue latency. Both assets selected today; no statistical edge claim. No parameter search.",
        cash_benchmark_return_pct=0,results=results)
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=="__main__":
    main()
