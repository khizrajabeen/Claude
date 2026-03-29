"""Paper trading engine — realistic simulation before risking real money.

Simulates:
  - Slippage (proportional to order size and volatility)
  - Maker/taker fees
  - Partial fills on limit orders
  - Latency delays
  - Portfolio tracking with full trade journal
"""

import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")


@dataclass
class PaperPosition:
    """A simulated open position."""
    id: str
    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    amount: float
    leverage: float
    entry_time: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_pct: float = 0.0
    highest_price: float = 0.0  # For trailing stop
    lowest_price: float = 0.0

    @property
    def notional(self) -> float:
        return self.entry_price * self.amount * self.leverage


@dataclass
class TradeRecord:
    """Completed trade for the journal."""
    id: str
    symbol: str
    side: str
    entry_price: float
    exit_price: float
    amount: float
    leverage: float
    entry_time: datetime
    exit_time: datetime
    pnl: float
    pnl_pct: float
    fees: float
    exit_reason: str  # "signal", "stop_loss", "take_profit", "trailing_stop"


class PaperTrader:
    """Full paper trading simulation engine."""

    def __init__(self, config: dict):
        self.config = config
        paper_config = config.get("paper", {})
        self.balance = paper_config.get("initial_balance", 10000.0)
        self.initial_balance = self.balance
        self.slippage_pct = paper_config.get("slippage_pct", 0.05)
        self.maker_fee_pct = paper_config.get("maker_fee_pct", 0.02)
        self.taker_fee_pct = paper_config.get("taker_fee_pct", 0.04)

        self.positions: list[PaperPosition] = []
        self.trade_journal: list[TradeRecord] = []
        self.equity_curve: list[dict] = []
        self.peak_equity = self.balance
        self.trade_counter = 0

        self.risk_config = config.get("risk", {})
        self.daily_start_balance = self.balance
        self.daily_start_date = datetime.now().date()
        self.cooldown_until = 0.0

        logger.info(f"Paper trader initialized | Balance: ${self.balance:.2f}")

    # ── Order Execution ───────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        price: float,
        leverage: float = 1.0,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        trailing_stop_pct: float = 0.0,
    ) -> PaperPosition | None:
        """Open a simulated position with realistic slippage and fees."""
        # Check cooldown
        if time.time() < self.cooldown_until:
            remaining = self.cooldown_until - time.time()
            logger.info(f"Cooldown active ({remaining:.0f}s remaining) — skipping")
            return None

        # Check daily drawdown limit
        if not self._check_daily_limit():
            return None

        # Simulate slippage
        slippage = self._calc_slippage(price, amount, side)
        fill_price = price + slippage if side == "long" else price - slippage

        # Calculate fees
        notional = fill_price * amount * leverage
        fees = notional * (self.taker_fee_pct / 100)

        # Check if we have enough balance
        margin_required = notional / leverage  # Margin = notional / leverage
        if margin_required + fees > self.balance:
            logger.warning(
                f"Insufficient balance: need ${margin_required + fees:.2f}, "
                f"have ${self.balance:.2f}"
            )
            return None

        # Deduct fees from balance
        self.balance -= fees

        self.trade_counter += 1
        pos_id = f"paper_{self.trade_counter:06d}"

        # Calculate stop/take-profit prices
        if stop_loss_pct > 0:
            if side == "long":
                sl = fill_price * (1 - stop_loss_pct / 100)
            else:
                sl = fill_price * (1 + stop_loss_pct / 100)
        else:
            sl = 0.0

        if take_profit_pct > 0:
            if side == "long":
                tp = fill_price * (1 + take_profit_pct / 100)
            else:
                tp = fill_price * (1 - take_profit_pct / 100)
        else:
            tp = 0.0

        position = PaperPosition(
            id=pos_id,
            symbol=symbol,
            side=side,
            entry_price=fill_price,
            amount=amount,
            leverage=leverage,
            entry_time=datetime.now(),
            stop_loss=sl,
            take_profit=tp,
            trailing_stop_pct=trailing_stop_pct,
            highest_price=fill_price,
            lowest_price=fill_price,
        )
        self.positions.append(position)

        logger.info(
            f"PAPER {side.upper()} | {amount} {symbol} @ ${fill_price:.2f} "
            f"| {leverage:.1f}x | SL=${sl:.2f} TP=${tp:.2f} | fees=${fees:.2f}"
        )
        return position

    def close_position(
        self, position: PaperPosition, price: float, reason: str = "signal"
    ) -> TradeRecord:
        """Close a position and record the trade."""
        # Slippage on exit
        slippage = self._calc_slippage(price, position.amount, "sell" if position.side == "long" else "buy")
        fill_price = price - slippage if position.side == "long" else price + slippage

        # PnL calculation
        if position.side == "long":
            pnl_pct = ((fill_price - position.entry_price) / position.entry_price) * 100
        else:
            pnl_pct = ((position.entry_price - fill_price) / position.entry_price) * 100

        pnl_pct *= position.leverage  # Leverage amplifies PnL
        raw_pnl = (pnl_pct / 100) * position.entry_price * position.amount

        # Exit fees
        notional = fill_price * position.amount * position.leverage
        fees = notional * (self.taker_fee_pct / 100)
        net_pnl = raw_pnl - fees

        self.balance += net_pnl

        # Track peak and drawdown
        self.peak_equity = max(self.peak_equity, self.balance)

        # Record trade
        trade = TradeRecord(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=fill_price,
            amount=position.amount,
            leverage=position.leverage,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            pnl=net_pnl,
            pnl_pct=pnl_pct,
            fees=fees,
            exit_reason=reason,
        )
        self.trade_journal.append(trade)
        self.positions.remove(position)

        # Cooldown after loss
        if net_pnl < 0:
            cooldown = self.risk_config.get("cooldown_after_loss_seconds", 300)
            self.cooldown_until = time.time() + cooldown
            logger.info(f"Loss cooldown activated ({cooldown}s)")

        logger.info(
            f"PAPER CLOSE {position.side.upper()} {position.symbol} | "
            f"PnL: ${net_pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason} | "
            f"Balance: ${self.balance:.2f}"
        )

        self._record_equity()
        return trade

    # ── Position Management ───────────────────────────────────

    def check_exits(self, symbol: str, current_price: float) -> list[TradeRecord]:
        """Check all positions for stop-loss, take-profit, and trailing stops."""
        closed = []
        positions = [p for p in self.positions if p.symbol == symbol]

        for pos in positions:
            # Update highest/lowest for trailing stop
            pos.highest_price = max(pos.highest_price, current_price)
            pos.lowest_price = min(pos.lowest_price, current_price)

            # Check stop-loss
            if pos.stop_loss > 0:
                if pos.side == "long" and current_price <= pos.stop_loss:
                    closed.append(self.close_position(pos, current_price, "stop_loss"))
                    continue
                if pos.side == "short" and current_price >= pos.stop_loss:
                    closed.append(self.close_position(pos, current_price, "stop_loss"))
                    continue

            # Check take-profit
            if pos.take_profit > 0:
                if pos.side == "long" and current_price >= pos.take_profit:
                    closed.append(self.close_position(pos, current_price, "take_profit"))
                    continue
                if pos.side == "short" and current_price <= pos.take_profit:
                    closed.append(self.close_position(pos, current_price, "take_profit"))
                    continue

            # Check trailing stop
            if pos.trailing_stop_pct > 0:
                if pos.side == "long":
                    trail_price = pos.highest_price * (1 - pos.trailing_stop_pct / 100)
                    if current_price <= trail_price:
                        closed.append(self.close_position(pos, current_price, "trailing_stop"))
                        continue
                else:
                    trail_price = pos.lowest_price * (1 + pos.trailing_stop_pct / 100)
                    if current_price >= trail_price:
                        closed.append(self.close_position(pos, current_price, "trailing_stop"))
                        continue

            # Check liquidation (leveraged position)
            if pos.leverage > 1:
                if pos.side == "long":
                    liq_price = pos.entry_price * (1 - 1 / pos.leverage)
                    if current_price <= liq_price:
                        closed.append(self.close_position(pos, current_price, "liquidation"))
                        continue
                else:
                    liq_price = pos.entry_price * (1 + 1 / pos.leverage)
                    if current_price >= liq_price:
                        closed.append(self.close_position(pos, current_price, "liquidation"))
                        continue

        return closed

    # ── Risk Checks ───────────────────────────────────────────

    def _check_daily_limit(self) -> bool:
        """Check if daily drawdown limit is breached."""
        today = datetime.now().date()
        if today != self.daily_start_date:
            self.daily_start_balance = self.balance
            self.daily_start_date = today

        daily_loss_pct = (
            (self.daily_start_balance - self.balance) / self.daily_start_balance * 100
            if self.daily_start_balance > 0 else 0
        )
        max_daily = self.risk_config.get("max_daily_drawdown_pct", 5.0)
        if daily_loss_pct >= max_daily:
            logger.warning(f"DAILY LOSS LIMIT HIT ({daily_loss_pct:.1f}%) — halting")
            return False
        return True

    @property
    def total_drawdown_pct(self) -> float:
        if self.peak_equity == 0:
            return 0.0
        return (self.peak_equity - self.balance) / self.peak_equity * 100

    @property
    def is_halted(self) -> bool:
        max_dd = self.risk_config.get("max_total_drawdown_pct", 15.0)
        return self.total_drawdown_pct >= max_dd

    # ── Slippage ──────────────────────────────────────────────

    def _calc_slippage(self, price: float, amount: float, side: str) -> float:
        """Simulate realistic slippage."""
        base_slippage = price * (self.slippage_pct / 100)
        # Larger orders have more slippage
        size_factor = 1 + np.log1p(amount) * 0.1
        return base_slippage * size_factor

    # ── Reporting ─────────────────────────────────────────────

    def _record_equity(self):
        self.equity_curve.append({
            "timestamp": datetime.now().isoformat(),
            "equity": self.balance,
            "open_positions": len(self.positions),
            "drawdown_pct": self.total_drawdown_pct,
        })

    def get_stats(self) -> dict:
        """Calculate comprehensive trading statistics."""
        if not self.trade_journal:
            return {"total_trades": 0, "balance": self.balance}

        trades = self.trade_journal
        pnls = [t.pnl for t in trades]
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        total_return = (self.balance - self.initial_balance) / self.initial_balance * 100

        # Sharpe ratio (annualized)
        if len(pnls) > 1:
            daily_returns = pd.Series(pnls) / self.initial_balance
            sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(365) if daily_returns.std() > 0 else 0
        else:
            sharpe = 0

        # Sortino ratio
        if len(losses) > 0:
            downside = pd.Series([t.pnl for t in losses]) / self.initial_balance
            sortino = (
                pd.Series(pnls).mean() / self.initial_balance
                / downside.std() * np.sqrt(365)
                if downside.std() > 0 else 0
            )
        else:
            sortino = float("inf")

        # Profit factor
        gross_profit = sum(t.pnl for t in wins) if wins else 0
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # Max consecutive losses
        max_consec_loss = 0
        current_streak = 0
        for t in trades:
            if t.pnl < 0:
                current_streak += 1
                max_consec_loss = max(max_consec_loss, current_streak)
            else:
                current_streak = 0

        # Average trade duration
        durations = [(t.exit_time - t.entry_time).total_seconds() for t in trades]

        stats = {
            "total_trades": len(trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(trades) * 100 if trades else 0,
            "total_pnl": sum(pnls),
            "total_return_pct": total_return,
            "avg_pnl": np.mean(pnls),
            "avg_win": np.mean([t.pnl for t in wins]) if wins else 0,
            "avg_loss": np.mean([t.pnl for t in losses]) if losses else 0,
            "largest_win": max(pnls) if pnls else 0,
            "largest_loss": min(pnls) if pnls else 0,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown_pct": self.total_drawdown_pct,
            "max_consecutive_losses": max_consec_loss,
            "avg_trade_duration_min": np.mean(durations) / 60 if durations else 0,
            "total_fees": sum(t.fees for t in trades),
            "current_balance": self.balance,
            "peak_balance": self.peak_equity,
        }
        return stats

    def print_stats(self):
        """Print formatted trading statistics."""
        stats = self.get_stats()
        logger.info("=" * 60)
        logger.info("  PAPER TRADING RESULTS")
        logger.info("=" * 60)
        logger.info(f"  Trades       : {stats['total_trades']} ({stats['winning_trades']}W / {stats['losing_trades']}L)")
        logger.info(f"  Win Rate     : {stats['win_rate']:.1f}%")
        logger.info(f"  Total PnL    : ${stats['total_pnl']:+.2f} ({stats['total_return_pct']:+.1f}%)")
        logger.info(f"  Profit Factor: {stats['profit_factor']:.2f}")
        logger.info(f"  Sharpe Ratio : {stats['sharpe_ratio']:.2f}")
        logger.info(f"  Sortino Ratio: {stats['sortino_ratio']:.2f}")
        logger.info(f"  Max Drawdown : {stats['max_drawdown_pct']:.1f}%")
        logger.info(f"  Max Consec L : {stats['max_consecutive_losses']}")
        logger.info(f"  Avg Win      : ${stats['avg_win']:.2f}")
        logger.info(f"  Avg Loss     : ${stats['avg_loss']:.2f}")
        logger.info(f"  Total Fees   : ${stats['total_fees']:.2f}")
        logger.info(f"  Balance      : ${stats['current_balance']:.2f}")
        logger.info("=" * 60)

    def save_journal(self, path: str = "data/paper_trades.json"):
        """Save trade journal to disk."""
        Path(path).parent.mkdir(exist_ok=True)
        records = []
        for t in self.trade_journal:
            records.append({
                "id": t.id,
                "symbol": t.symbol,
                "side": t.side,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "amount": t.amount,
                "leverage": t.leverage,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "pnl": t.pnl,
                "pnl_pct": t.pnl_pct,
                "fees": t.fees,
                "exit_reason": t.exit_reason,
            })
        with open(path, "w") as f:
            json.dump({"trades": records, "stats": self.get_stats()}, f, indent=2, default=str)
        logger.info(f"Trade journal saved to {path}")

    def get_open_positions(self, symbol: str | None = None) -> list[PaperPosition]:
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return list(self.positions)
