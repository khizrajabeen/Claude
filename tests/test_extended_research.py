import numpy as np
import pandas as pd
import pytest

from bot.utils.extended_research import RULES, signals, simulate


def frame(declining=False):
    prices = np.linspace(200,100,320) if declining else np.linspace(100,200,320)
    return pd.DataFrame(dict(open=prices,high=prices+1,low=prices-1,
        close=prices,volume=np.full(320,1000)),
        index=pd.date_range("2023-01-01",periods=320,tz="UTC",freq="D"))


@pytest.mark.parametrize("rule",RULES)
def test_challengers_are_prefix_causal(rule):
    df=frame()
    expected=signals(df,rule)
    for cut in (210,250,300):
        pd.testing.assert_series_equal(signals(df.iloc[:cut],rule),expected.iloc[:cut])


def test_short_momentum_can_profit_from_decline_after_costs():
    df=frame(True)
    start,end=str(df.index[200].date()),str(df.index[-1].date())
    result=simulate({"SPY":df},"momentum_90_200",start,end,fee_bps=0,slip_bps=5,shorts=True)
    assert result["return_pct"]>0
    assert result["carry"]>0
    assert result["ledger"][0]["side"]==-1
    assert sum(t["pnl"] for t in result["ledger"])==pytest.approx(result["return_pct"]*100)
    costly=simulate({"SPY":df},"momentum_90_200",start,end,fee_bps=0,slip_bps=15,
                    shorts=True,carry_pct=11)
    assert costly["return_pct"]<result["return_pct"]
    long_only=simulate({"SPY":df},"momentum_90_200",start,end,shorts=False)
    assert long_only["return_pct"]==0


def test_short_is_never_filled_on_signal_bar():
    df=frame(True)
    result=simulate({"SPY":df},"momentum_90_200",str(df.index[190].date()),
                    str(df.index[-1].date()),shorts=True)
    assert pd.Timestamp(result["ledger"][0]["entry_time"])==df.index[200]


def test_flat_market_buy_hold_loses_only_costs():
    df=frame()
    df[["open","close"]]=100
    df["high"]=101;df["low"]=99
    result=simulate({"X":df},"buy_hold",str(df.index[200].date()),str(df.index[-1].date()))
    assert result["return_pct"]<0
    assert result["carry"]==0
    assert result["round_trips"]==1
