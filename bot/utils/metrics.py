"""Performance metrics for evaluating trading strategies.

Implements the metrics that institutional traders and quant funds
actually care about — not just total return.
"""

import numpy as np
import pandas as pd


def sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 365) -> float:
    """Annualized Sharpe ratio."""
    excess = returns - risk_free_rate / periods
    if excess.std() == 0:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(periods))


def sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods: int = 365) -> float:
    """Annualized Sortino ratio (penalizes downside volatility only)."""
    excess = returns - risk_free_rate / periods
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float("inf") if excess.mean() > 0 else 0.0
    return float(excess.mean() / downside.std() * np.sqrt(periods))


def calmar_ratio(returns: pd.Series, periods: int = 365) -> float:
    """Calmar ratio = annualized return / max drawdown."""
    ann_return = returns.mean() * periods
    mdd = max_drawdown(returns)
    if mdd == 0:
        return float("inf") if ann_return > 0 else 0.0
    return float(ann_return / mdd)


def max_drawdown(returns: pd.Series) -> float:
    """Maximum drawdown from peak to trough."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(abs(dd.min())) if len(dd) > 0 else 0.0


def max_drawdown_duration(returns: pd.Series) -> int:
    """Max number of periods spent in drawdown."""
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    in_dd = cum < peak

    max_dur = 0
    current = 0
    for is_dd in in_dd:
        if is_dd:
            current += 1
            max_dur = max(max_dur, current)
        else:
            current = 0
    return max_dur


def profit_factor(returns: pd.Series) -> float:
    """Gross profits / gross losses."""
    wins = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return float("inf") if wins > 0 else 0.0
    return float(wins / losses)


def win_rate(returns: pd.Series) -> float:
    """Percentage of profitable trades/periods."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).sum() / len(returns) * 100)


def expectancy(returns: pd.Series) -> float:
    """Expected value per trade = win_rate * avg_win - loss_rate * avg_loss."""
    wins = returns[returns > 0]
    losses = returns[returns < 0]
    if len(returns) == 0:
        return 0.0

    wr = len(wins) / len(returns)
    lr = len(losses) / len(returns)
    avg_w = wins.mean() if len(wins) > 0 else 0
    avg_l = abs(losses.mean()) if len(losses) > 0 else 0

    return float(wr * avg_w - lr * avg_l)


def risk_of_ruin(win_rate_pct: float, payoff_ratio: float, risk_per_trade_pct: float) -> float:
    """Estimate probability of losing entire account.

    Args:
        win_rate_pct: Win rate as percentage (e.g. 55.0)
        payoff_ratio: Average win / average loss
        risk_per_trade_pct: Risk per trade as percentage
    """
    wr = win_rate_pct / 100
    lr = 1 - wr
    if wr == 0 or payoff_ratio == 0:
        return 1.0

    edge = wr * payoff_ratio - lr
    if edge <= 0:
        return 1.0

    units = 100 / risk_per_trade_pct  # How many trades to ruin
    ruin = ((lr / (wr * payoff_ratio)) ** units) if wr * payoff_ratio > lr else 1.0
    return float(min(1.0, ruin))


def compute_all_metrics(returns: pd.Series) -> dict:
    """Compute all performance metrics."""
    return {
        "total_return_pct": float((1 + returns).prod() - 1) * 100,
        "sharpe_ratio": sharpe_ratio(returns),
        "sortino_ratio": sortino_ratio(returns),
        "calmar_ratio": calmar_ratio(returns),
        "max_drawdown_pct": max_drawdown(returns) * 100,
        "max_drawdown_duration": max_drawdown_duration(returns),
        "profit_factor": profit_factor(returns),
        "win_rate_pct": win_rate(returns),
        "expectancy": expectancy(returns),
        "total_trades": len(returns),
        "avg_return_pct": float(returns.mean()) * 100,
        "std_return_pct": float(returns.std()) * 100,
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurt()),
    }
