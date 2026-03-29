"""Risk management module.

Enforces stop-loss, take-profit, position sizing limits,
and daily loss circuit breakers.
"""

import logging
from dataclasses import dataclass, field
from datetime import date

logger = logging.getLogger("trading_bot")


@dataclass
class Position:
    """Represents an open trading position."""
    symbol: str
    side: str           # "long" or "short"
    entry_price: float
    amount: float
    order_id: str = ""

    @property
    def notional(self) -> float:
        return self.entry_price * self.amount


@dataclass
class RiskManager:
    """Manages risk limits and tracks positions."""

    stop_loss_pct: float
    take_profit_pct: float
    max_daily_loss_pct: float
    max_position_size_pct: float
    max_open_trades: int
    positions: list = field(default_factory=list)
    daily_pnl: float = 0.0
    daily_pnl_date: date = field(default_factory=date.today)

    @classmethod
    def from_config(cls, config: dict) -> "RiskManager":
        rm = config["risk_management"]
        return cls(
            stop_loss_pct=rm["stop_loss_pct"],
            take_profit_pct=rm["take_profit_pct"],
            max_daily_loss_pct=rm["max_daily_loss_pct"],
            max_position_size_pct=rm["max_position_size_pct"],
            max_open_trades=config["trading"].get("max_open_trades", 3),
        )

    def _reset_daily_pnl_if_needed(self):
        today = date.today()
        if self.daily_pnl_date != today:
            logger.info(f"New trading day — resetting daily PnL (was {self.daily_pnl:.2f})")
            self.daily_pnl = 0.0
            self.daily_pnl_date = today

    def can_open_trade(self, portfolio_value: float) -> bool:
        """Check if a new trade is allowed under current risk limits."""
        self._reset_daily_pnl_if_needed()

        if len(self.positions) >= self.max_open_trades:
            logger.warning("Max open trades reached — skipping signal")
            return False

        loss_limit = portfolio_value * (self.max_daily_loss_pct / 100)
        if self.daily_pnl < 0 and abs(self.daily_pnl) >= loss_limit:
            logger.warning("Daily loss limit hit — halting trading for today")
            return False

        return True

    def check_position_size(self, amount: float, price: float, portfolio_value: float) -> bool:
        """Check if a proposed position is within size limits."""
        notional = amount * price
        max_notional = portfolio_value * (self.max_position_size_pct / 100)
        if notional > max_notional:
            logger.warning(
                f"Position size {notional:.2f} exceeds max {max_notional:.2f} — rejected"
            )
            return False
        return True

    def check_stop_loss(self, position: Position, current_price: float) -> bool:
        """Return True if stop-loss is triggered."""
        if position.side == "long":
            loss_pct = ((position.entry_price - current_price) / position.entry_price) * 100
        else:
            loss_pct = ((current_price - position.entry_price) / position.entry_price) * 100

        if loss_pct >= self.stop_loss_pct:
            logger.warning(
                f"STOP-LOSS triggered for {position.symbol} "
                f"(loss: {loss_pct:.2f}%, limit: {self.stop_loss_pct}%)"
            )
            return True
        return False

    def check_take_profit(self, position: Position, current_price: float) -> bool:
        """Return True if take-profit is triggered."""
        if position.side == "long":
            gain_pct = ((current_price - position.entry_price) / position.entry_price) * 100
        else:
            gain_pct = ((position.entry_price - current_price) / position.entry_price) * 100

        if gain_pct >= self.take_profit_pct:
            logger.info(
                f"TAKE-PROFIT triggered for {position.symbol} "
                f"(gain: {gain_pct:.2f}%, target: {self.take_profit_pct}%)"
            )
            return True
        return False

    def open_position(self, symbol: str, side: str, entry_price: float, amount: float, order_id: str = ""):
        """Record a new open position."""
        pos = Position(symbol=symbol, side=side, entry_price=entry_price, amount=amount, order_id=order_id)
        self.positions.append(pos)
        logger.info(f"Opened {side} position: {amount} {symbol} @ {entry_price}")
        return pos

    def close_position(self, position: Position, exit_price: float):
        """Close a position and update daily PnL."""
        if position.side == "long":
            pnl = (exit_price - position.entry_price) * position.amount
        else:
            pnl = (position.entry_price - exit_price) * position.amount

        self.daily_pnl += pnl
        self.positions.remove(position)
        logger.info(
            f"Closed {position.side} {position.symbol}: "
            f"PnL={pnl:+.2f} | Daily PnL={self.daily_pnl:+.2f}"
        )
        return pnl

    def get_positions_for_symbol(self, symbol: str) -> list:
        """Get all open positions for a symbol."""
        return [p for p in self.positions if p.symbol == symbol]
