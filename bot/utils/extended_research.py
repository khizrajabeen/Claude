"""Predeclared challenger research. Does not submit orders or tune parameters.

Crypto: baseline, faster momentum, breakout, and trend-filtered reversion.
Equity ETFs: matching long-only and long/short momentum experiments.
Every result includes trading costs; shorts additionally pay modeled carry.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests


CRYPTO = ("BTC/USD", "ETH/USD", "SOL/USD", "LTC/USD", "LINK/USD")
ETFS = ("SPY", "QQQ")
RULES = ("momentum_90_200", "momentum_20_100", "breakout_55_20", "reversion_2_200")
WINDOWS = {
    "development": ("2022-01-01", "2024-01-01"),
    "reused_validation": ("2024-01-01", "2026-01-01"),
    "recent": ("2026-01-01", "2026-09-26"),
}


def signals(df, rule, shorts=False):
    close = df.close
    if rule.startswith("momentum_"):
        _, lookback, trend = rule.split("_")
        momentum = close / close.shift(int(lookback)) - 1
        average = close.rolling(int(trend)).mean()
        result = ((momentum > 0) & (close > average)).astype(int)
        if shorts:
            result -= ((momentum < 0) & (close < average)).astype(int)
        return result
    if rule == "buy_hold":
        return pd.Series(1, index=df.index)
    if rule == "breakout_55_20":
        high = df.high.shift(1).rolling(55).max()
        low = df.low.shift(1).rolling(20).min()
        held = 0
        out = []
        for price, upper, lower in zip(close, high, low):
            if held and price < lower:
                held = 0
            elif not held and price > upper:
                held = 1
            out.append(held)
        return pd.Series(out, index=df.index)
    if rule != "reversion_2_200" or shorts:
        raise ValueError("Unsupported research rule")
    # Simple two-period RSI, not Wilder smoothing. Explicitly fixed rule.
    move = close.diff()
    up = move.clip(lower=0).rolling(2).mean()
    down = (-move.clip(upper=0)).rolling(2).mean()
    rsi = 100 * up / (up + down)
    trend = close.rolling(200).mean()
    fast = close.rolling(5).mean()
    held = 0
    age = 0
    out = []
    for price, strength, slow, quick in zip(close, rsi, trend, fast):
        if held:
            age += 1
            if price > quick or price < slow or age >= 5:
                held = 0
        elif price > slow and strength < 10:
            held, age = 1, 0
        out.append(held)
    return pd.Series(out, index=df.index)


def download(symbols, equity=False):
    headers = {}
    if os.getenv("ALPACA_API_KEY_ID") and os.getenv("ALPACA_API_SECRET_KEY"):
        headers = {"APCA-API-KEY-ID": os.environ["ALPACA_API_KEY_ID"],
                   "APCA-API-SECRET-KEY": os.environ["ALPACA_API_SECRET_KEY"]}
    if equity and not headers:
        raise ValueError("Equity research needs Alpaca data credentials")
    endpoint = ("https://data.alpaca.markets/v2/stocks/bars" if equity else
                "https://data.alpaca.markets/v1beta3/crypto/us/bars")
    params = dict(symbols=",".join(symbols), timeframe="1Day", limit=10000,
                  start="2021-01-01T00:00:00Z", end="2026-09-26T00:00:00Z")
    if equity:
        params.update(feed="iex", adjustment="split")
    raw = {s: [] for s in symbols}
    tokens = set()
    while True:
        response = requests.get(endpoint, headers=headers, params=params, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError(f"Data request returned HTTP {response.status_code}")
        data = response.json()
        for s in symbols:
            raw[s].extend(data.get("bars", {}).get(s, []))
        token = data.get("next_page_token")
        if not token:
            break
        if token in tokens:
            raise ValueError("Repeated pagination token")
        tokens.add(token)
        params["page_token"] = token
    cutoff = pd.Timestamp("2026-09-26T00:00:00Z")
    raw = {s: [b for b in bars if pd.Timestamp(b["t"])+pd.Timedelta(days=1) <= cutoff]
           for s, bars in raw.items()}
    return dict(source=endpoint, evaluation_end=str(cutoff), equity=equity, bars=raw)


def frames(raw, excluded=None):
    result = {}
    for s, rows in raw["bars"].items():
        df = pd.DataFrame(rows)
        if len(df) < 200:
            raise ValueError(f"Insufficient history: {s}")
        df.index = pd.to_datetime(df.pop("t"), utc=True).dt.normalize()
        df = df.rename(columns=dict(o="open",h="high",l="low",c="close",v="volume"))
        df = df[["open","high","low","close","volume"]].sort_index()
        if not df.index.is_unique or not np.isfinite(df.to_numpy()).all():
            raise ValueError(f"Invalid history: {s}")
        if (df[["open","high","low","close"]] <= 0).any().any():
            raise ValueError(f"Non-positive price: {s}")
        if not raw["equity"] and not (df.index.to_series().diff().dropna()==pd.Timedelta(days=1)).all():
            if excluded is None:
                raise ValueError(f"Missing crypto daily bars: {s}")
            gaps = df.index.to_series().diff().dropna()
            excluded[s] = f"Missing daily history; largest gap {gaps.max()}"
            continue
        result[s] = df
    return result


def simulate(data, rule, start, end, fee_bps=25, slip_bps=10,
             shorts=False, carry_pct=4, allocation=0.4):
    common = next(iter(data.values())).index
    for df in data.values():
        common = common.intersection(df.index)
    dates = common[(common >= pd.Timestamp(start, tz="UTC")) &
                   (common < pd.Timestamp(end, tz="UTC"))]
    if len(dates) < 20:
        raise ValueError("Insufficient common evaluation history")
    # Each original source series computes its own indicators; positions only
    # trade on common days. Never fabricate bars by forward filling.
    target = {s: signals(df,rule,shorts).shift(1).fillna(0) for s,df in data.items()}
    initial = cash = 10000.0
    held = {}
    trades, curve = [], []
    fee, slip = fee_bps/10000, slip_bps/10000
    total_fees = carry = turnover = exposure_sum = 0.0
    previous = dates[0]

    def close(s, dt, price, reason):
        nonlocal cash, total_fees, turnover
        p = held.pop(s)
        fill = price*(1-p["side"]*slip)
        commission = p["qty"]*fill*fee
        cash_delta = p["side"]*p["qty"]*fill-commission
        cash += cash_delta
        total_fees += commission
        turnover += p["qty"]*fill
        pnl = p["opening_cash_delta"]+cash_delta-p["carry"]
        trades.append(dict(symbol=s,side=p["side"],entry_time=p["date"],exit_time=str(dt),
                           pnl=pnl,fees=p["fee"]+commission,carry=p["carry"],reason=reason))

    for dt in dates:
        days = (dt-previous).days
        for s,p in held.items():
            if p["side"] < 0:
                charge = p["qty"]*data[s].at[previous,"close"]*carry_pct/100*days/365
                cash -= charge;carry += charge;p["carry"] += charge
        opening_equity = cash+sum(p["side"]*p["qty"]*data[s].at[dt,"open"] for s,p in held.items())
        for s in list(held):
            if int(target[s].at[dt]) != held[s]["side"]:
                close(s,dt,data[s].at[dt,"open"],"signal")
        for s,df in data.items():
            side = int(target[s].at[dt])
            if side and s not in held:
                budget = max(0, opening_equity*allocation/len(data))
                fill = df.at[dt,"open"]*(1+side*slip)
                qty = budget/(fill*(1+fee))
                commission = qty*fill*fee
                delta = -side*qty*fill-commission
                cash += delta;total_fees += commission;turnover += qty*fill
                held[s] = dict(qty=qty,side=side,opening_cash_delta=delta,fee=commission,
                               carry=0.0,date=str(dt))
        equity = cash+sum(p["side"]*p["qty"]*data[s].at[dt,"close"] for s,p in held.items())
        exposure_sum += sum(p["qty"]*data[s].at[dt,"close"] for s,p in held.items())/equity
        curve.append(dict(date=str(dt),equity=equity))
        previous=dt
    for s in list(held):
        close(s,dates[-1],data[s].at[dates[-1],"close"],"end_of_sample")
    curve[-1]["equity"] = cash
    if not np.isclose(sum(t["pnl"] for t in trades),cash-initial,atol=1e-6):
        raise AssertionError("Trade ledger does not reconcile with equity")
    eq = np.r_[initial,[v["equity"] for v in curve]]
    daily = pd.Series(np.diff(eq)/eq[:-1],index=dates)
    monthly = pd.Series(eq[1:],index=dates).resample("ME").last()
    monthly_returns = monthly.pct_change()
    monthly_returns.iloc[0] = monthly.iloc[0]/initial-1
    return dict(rule=rule,shorts=shorts,symbols=list(data),start=str(dates[0]),end=str(dates[-1]),
        days=len(dates),return_pct=(cash/initial-1)*100,
        max_drawdown_pct=-float((eq/np.maximum.accumulate(eq)-1).min())*100,
        round_trips=len(trades),positive_day_pct=float((daily>0).mean())*100,
        worst_month_pct=float(monthly_returns.min())*100 if len(monthly_returns) else None,
        fees=total_fees,carry=carry,turnover=turnover,mean_gross_exposure_pct=exposure_sum/len(dates)*100,
        costs=dict(fee_bps=fee_bps,slippage_bps=slip_bps,short_carry_pct=carry_pct),
        ledger=trades,equity=curve)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--out",default="research/extended")
    ap.add_argument("--equities",action="store_true")
    ap.add_argument("--cached",action="store_true")
    args=ap.parse_args()
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    datasets = [("crypto",CRYPTO,False)] + ([("etfs",ETFS,True)] if args.equities else [])
    all_frames={};hashes={};excluded={}
    for name,symbols,equity in datasets:
        path=out/f"{name}-bars.json"
        raw=json.loads(path.read_text()) if args.cached else download(symbols,equity)
        text=json.dumps(raw,sort_keys=True);path.write_text(text)
        hashes[name]=hashlib.sha256(text.encode()).hexdigest()
        all_frames[name]=frames(raw,excluded)
    results=[]
    arms=[("crypto_2",{s:all_frames["crypto"][s] for s in CRYPTO[:2]},r,False)
          for r in (*RULES,"buy_hold")]
    arms += [(f"crypto_{len(all_frames['crypto'])}",all_frames["crypto"],r,False) for r in (*RULES,"buy_hold")]
    if "etfs" in all_frames:
        arms += [("etfs",all_frames["etfs"],r,shorts)
                 for r in ("momentum_90_200","momentum_20_100") for shorts in (False,True)]
        arms += [("etfs",all_frames["etfs"],"buy_hold",False)]
    for universe,data,rule,shorts in arms:
        for window,(start,end) in WINDOWS.items():
            for stressed in (False,True):
                equity=universe=="etfs"
                r=simulate(data,rule,start,end,fee_bps=0 if equity else 25,
                    slip_bps=(15 if stressed else 5) if equity else (25 if stressed else 10),
                    shorts=shorts,carry_pct=11 if stressed else 4)
                stem=f"{universe}-{rule}-{'ls' if shorts else 'long'}-{window}-{'stress' if stressed else 'base'}"
                (out/f"{stem}.json").write_text(json.dumps(r,indent=2))
                results.append(dict(universe=universe,window=window,stressed=stressed,
                    **{k:v for k,v in r.items() if k not in ("ledger","equity")}))
    report=dict(data_sha256=hashes,excluded_markets=excluded,results=results,
        limitations=["Later historical periods were already inspected in prior work; they are reused validation, not pristine holdouts.",
          "No strategy is guaranteed to profit daily. Cash benchmark is zero; uninvested cash earns zero.",
          "Universe chosen today; common-history comparison is not survivorship-free.",
          "40% of equity allocated across new entries; weights drift; all entries use next daily open.",
          "ETF returns exclude cash dividends; shorts pay modeled 3% borrow + 1% dividend drag annually (10% + 1% under stress). These are scenarios, not measured borrow quotes.",
          "IEX data may differ from consolidated prices; historical short availability is not reconstructed.",
          "Crypto shorting is unsupported by Alpaca and is not submitted or simulated as an executable Alpaca strategy.",
          "Multiple comparisons are exploratory; the frozen paper baseline is not automatically replaced."])
    (out/"summary.json").write_text(json.dumps(report,indent=2))
    lines=["# Extended strategy research", "", "Excluded markets: "+json.dumps(excluded), "", *["- "+x for x in report["limitations"]], "",
           "| Universe | Rule | Side | Window | Costs | Return | Drawdown | Trades |",
           "|---|---|---|---|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['universe']} | {r['rule']} | {'long/short' if r['shorts'] else 'long'} | {r['window']} | {'stress' if r['stressed'] else 'base'} | {r['return_pct']:+.2f}% | {r['max_drawdown_pct']:.2f}% | {r['round_trips']} |")
    (out/"RESULTS.md").write_text("\n".join(lines)+"\n")
    print(json.dumps(dict(comparisons=len(results),data_sha256=hashes)))


if __name__=="__main__":
    main()
