import json

import numpy as np
import pandas as pd
import pytest

from bot.strategies.daily_candidates import CANDIDATES, completed_bars, exposure
from bot.trading.daily_paper import PaperAPI, run_cycle
from bot.utils.daily_research import backtest, SYMBOLS


def frame(n=260):
    close = np.linspace(100,200,n)
    return pd.DataFrame(dict(open=close,high=close+0.1,low=close-0.1,
                            close=close,volume=np.ones(n)*1000),
        index=pd.date_range("2024-01-01",periods=n,freq="D",tz="UTC"))


@pytest.mark.parametrize("candidate",CANDIDATES)
def test_prefix_causality_and_no_signal_during_warmup(candidate):
    df=frame()
    full=exposure(df,candidate)
    assert not full.iloc[:55 if candidate.startswith("breakout") else 199].any()
    for cut in (200,215,240,259):
        assert exposure(df.iloc[:cut],candidate).iloc[-1] == full.iloc[cut-1]
    corrupted=df.copy()
    corrupted.loc[corrupted.index[220]:,["open","high","low","close"]] *= 2
    pd.testing.assert_series_equal(exposure(df,candidate).iloc[:220],
                                   exposure(corrupted,candidate).iloc[:220])


def test_current_daily_candle_is_not_visible():
    df=frame()
    assert len(completed_bars(df,df.index[-1])) == len(df)-1


def test_research_executes_after_signal_and_reconciles_cash():
    df=frame()
    frames={s:df.copy() for s in SYMBOLS}
    r=backtest(frames,"momentum_90_200",str(df.index[190]),
               str(df.index[-1]+pd.Timedelta(days=1)))
    assert pd.Timestamp(r["ledger"][0]["entry_time"]) == df.index[200]
    assert sum(t["pnl"] for t in r["ledger"]) == pytest.approx(r["return_pct"]*100)
    assert all(t["fees"]>0 for t in r["ledger"])
    costly=backtest(frames,"momentum_90_200",str(df.index[190]),
                   str(df.index[-1]+pd.Timedelta(days=1)),slip_bps=100)
    assert costly["return_pct"] < r["return_pct"]


class API:
    def __init__(self,now):
        self.now=now;self.orders={};self.positions=[];self.posts=0
        self.timeout=False

    def request(self,method,path,**kwargs):
        if path=="/account":return dict(id="paper-test",status="ACTIVE",cash="10000")
        if path=="/positions":return self.positions
        if path=="/orders:by_client_order_id":
            return self.orders.get(kwargs["params"]["client_order_id"])
        if method=="GET" and path=="/orders":return []
        if method=="POST":
            self.posts+=1
            body=kwargs["json"];cid=body["client_order_id"]
            value=float(body["notional"]) if body["side"]=="buy" else float(body["qty"])*200
            order=dict(status="filled",filled_qty=str(value/200),filled_avg_price="200")
            self.orders[cid]=order
            if body["side"]=="buy":
                self.positions.append(dict(symbol=body["symbol"],qty=str(value/200),market_value=str(value)))
            else:
                self.positions=[p for p in self.positions if p["symbol"]!=body["symbol"]]
            if self.timeout:raise TimeoutError("simulated lost acknowledgement")
            return order
        raise AssertionError((method,path))

    def quote(self,symbol):return dict(bp=199.9,ap=200.1,t=str(self.now))


CFG=dict(candidate="momentum_90_200",capital_usd=1000,allocation_per_asset=0.2,
         fee_bps=25,max_spread_bps=50)


def setup():
    df=frame();now=df.index[-1]+pd.Timedelta(days=1,minutes=15)
    return {s:df.copy() for s in SYMBOLS},now,API(now)


def test_live_endpoint_setting_is_rejected_before_network(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER","false")
    with pytest.raises(ValueError,match="only permits"):PaperAPI()


def test_plan_does_not_submit_and_submit_is_idempotent(tmp_path):
    frames,now,api=setup();state=tmp_path/"state.json"
    run_cycle(api,frames,CFG,state,tmp_path,now,submit=False)
    assert api.posts==0
    for _ in range(5):run_cycle(api,frames,CFG,state,tmp_path,now,submit=True)
    assert api.posts==2 # exactly one buy per asset, even on repeated invocations
    assert max(float(p["market_value"]) for p in api.positions)<=200
    assert json.loads(state.read_text())["pending"]=={}


def test_timeout_after_venue_acceptance_recovers_without_duplicate(tmp_path):
    frames,now,api=setup();state=tmp_path/"state.json";api.timeout=True
    with pytest.raises(TimeoutError):run_cycle(api,frames,CFG,state,tmp_path,now,True)
    assert json.loads(state.read_text())["pending"]
    api.timeout=False
    run_cycle(api,frames,CFG,state,tmp_path,now,True)
    assert api.posts==2 # recovered BTC, submitted ETH only


def test_unmanaged_positions_are_not_adopted(tmp_path):
    frames,now,api=setup()
    api.positions=[dict(symbol="BTCUSD",qty="1",market_value="200")]
    with pytest.raises(ValueError,match="Unmanaged"):
        run_cycle(api,frames,CFG,tmp_path/"state.json",tmp_path,now,True)
    assert api.posts==0


def test_partial_order_prevents_new_exposure(tmp_path):
    frames,now,api=setup();state=tmp_path/"state.json"
    run_cycle(api,frames,CFG,state,tmp_path,now,True)
    next(iter(api.orders.values()))["status"]="partially_filled"
    result=run_cycle(api,frames,CFG,state,tmp_path,now,True)
    assert result["status"]=="pending_orders"
    assert api.posts==1


def test_missing_or_stale_data_blocks_orders(tmp_path):
    frames,now,api=setup()
    for s in frames:frames[s]=frames[s].iloc[:-1]
    with pytest.raises(ValueError,match="Missing completed"):
        run_cycle(api,frames,CFG,tmp_path/"state.json",tmp_path,now,True)
    assert api.posts==0


def test_exit_signal_sells_held_quantity_and_recovers_cash(tmp_path):
    frames,now,api=setup();state=tmp_path/"state.json"
    for _ in range(3):run_cycle(api,frames,CFG,state,tmp_path,now,True)
    assert api.posts==2
    for s,df in frames.items():
        df.loc[df.index[-1],["open","high","low","close"]]=[90,91,89,90]
    # Distinct side yields a distinct ID; no short or synthetic position.
    for _ in range(3):run_cycle(api,frames,CFG,state,tmp_path,now,True)
    assert api.posts==4
    assert api.positions==[]
    assert json.loads(state.read_text())["cash"]==pytest.approx(999.0)


def test_crash_before_post_recovers_same_intent(tmp_path):
    frames,now,api=setup();state=tmp_path/"state.json"
    original=api.request
    def fail_post(method,path,**kwargs):
        if method=="POST":raise TimeoutError("before sending")
        return original(method,path,**kwargs)
    api.request=fail_post
    with pytest.raises(TimeoutError):run_cycle(api,frames,CFG,state,tmp_path,now,True)
    assert api.posts==0
    api.request=original
    run_cycle(api,frames,CFG,state,tmp_path,now,True)
    assert api.posts==2
