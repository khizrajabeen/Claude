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


def summarize_trades(trades: list, starting_equity: float | None = None) -> dict:
    """Roll a list of Trade records into the numbers worth reporting.

    Results are expressed in R (profit per unit of risk taken) as well as
    dollars. R is what makes trades comparable when position size varies
    with volatility, which it does here by design.
    """
    if not trades:
        return {"total_trades": 0}

    pnls = [t.pnl for t in trades]
    r_multiples = [t.r_multiple for t in trades]
    wins = [t for t in trades if t.pnl > 0]
    losses = [t for t in trades if t.pnl <= 0]

    gross_profit = sum(t.pnl for t in wins)
    gross_loss = abs(sum(t.pnl for t in losses))

    avg_win_r = float(np.mean([t.r_multiple for t in wins])) if wins else 0.0
    avg_loss_r = abs(float(np.mean([t.r_multiple for t in losses]))) if losses else 0.0
    wr = len(wins) / len(trades)

    # Drawdown must be measured against account equity. Measuring it
    # against cumulative PnL divides by a number that starts near zero and
    # produces nonsense like "431% drawdown".
    base = float(starting_equity) if starting_equity else 0.0
    if base <= 0:
        base = max(abs(float(np.sum(pnls))), 1.0)
    equity = base + np.cumsum(pnls)
    peak = np.maximum.accumulate(np.concatenate([[base], equity]))[1:]
    max_dd = float(np.max((peak - equity) / peak) * 100) if len(equity) else 0.0

    # Group into calendar days before annualising, so the Sharpe reflects
    # daily variability rather than pretending each trade is a day.
    by_day: dict[str, float] = {}
    for t in trades:
        by_day[t.closed_on_day or t.closed_at.date().isoformat()] = \
            by_day.get(t.closed_on_day or t.closed_at.date().isoformat(), 0.0) + t.pnl
    daily = pd.Series(list(by_day.values()))
    sharpe_daily = float(daily.mean() / daily.std() * np.sqrt(365)) \
        if len(daily) > 1 and daily.std() > 0 else 0.0

    reasons: dict[str, int] = {}
    for t in trades:
        reasons[t.exit_reason] = reasons.get(t.exit_reason, 0) + 1

    # How much of this is luck?
    #
    # A Sharpe of 13 on 13 trades is not a good strategy, it is a small
    # sample. The standard error of mean R falls with the square root of
    # the trade count, so the t-statistic is the only honest way to compare
    # a selective system against a high-frequency one — without it, the
    # strategy that traded least always looks best.
    r_array = np.asarray(r_multiples, dtype=float)
    r_std = float(r_array.std(ddof=1)) if len(r_array) > 1 else 0.0
    standard_error = r_std / np.sqrt(len(r_array)) if r_std > 0 else 0.0
    t_stat = float(np.mean(r_array) / standard_error) if standard_error > 0 else 0.0

    # Trades needed for the observed edge to reach a t of 2, if it is real.
    if abs(float(np.mean(r_array))) > 1e-9 and r_std > 0:
        trades_for_significance = int(np.ceil((2 * r_std / float(np.mean(r_array))) ** 2))
    else:
        trades_for_significance = 0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(wr * 100, 2),
        "total_pnl": round(float(sum(pnls)), 2),
        "avg_r": round(float(np.mean(r_multiples)), 4),
        "expectancy_r": round(wr * avg_win_r - (1 - wr) * avg_loss_r, 4),
        "avg_win_r": round(avg_win_r, 4),
        "avg_loss_r": round(avg_loss_r, 4),
        "best_r": round(float(np.max(r_multiples)), 4),
        "worst_r": round(float(np.min(r_multiples)), 4),
        "profit_factor": round(gross_profit / gross_loss, 3) if gross_loss > 0 else 0.0,
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "total_fees": round(float(sum(t.fees for t in trades)), 2),
        "total_funding": round(float(sum(t.funding for t in trades)), 4),
        "max_drawdown_pct": round(max_dd, 3),
        "sharpe_daily": round(sharpe_daily, 3),
        "avg_holding_minutes": round(
            float(np.mean([t.holding_minutes for t in trades])), 1
        ),
        "avg_mae_r": round(float(np.mean([t.mae_r for t in trades])), 4),
        "avg_mfe_r": round(float(np.mean([t.mfe_r for t in trades])), 4),
        "exit_reasons": reasons,
        "trading_days": len(by_day),
        "r_std": round(r_std, 4),
        "expectancy_se": round(standard_error, 4),
        "t_stat": round(t_stat, 2),
        # |t| >= 2 is the usual bar. Below it, the result is compatible
        # with having no edge at all.
        "significant": bool(abs(t_stat) >= 2.0),
        "trades_for_significance": trades_for_significance,
    }
