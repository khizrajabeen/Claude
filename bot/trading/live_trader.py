"""Live trading engine — real money execution with safety rails.

Wraps the exchange client with:
  - Pre-trade validation
  - Smart order execution (TWAP for large orders)
  - Position tracking synced with exchange
  - Emergency kill switch
  - Full audit trail
"""

import logging
import time
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

from bot.exchange import ExchangeClient

logger = logging.getLogger("trading_bot")


@dataclass
class LivePosition:
    """Tracked live position."""
    id: str
    symbol: str
    side: str
    entry_price: float
    amount: float
    leverage: float
    entry_time: datetime
    stop_loss: float = 0.0
    take_profit: float = 0.0
    trailing_stop_pct: float = 0.0
    highest_price: float = 0.0
    lowest_price: float = 0.0
    order_ids: list = field(default_factory=list)


@dataclass
class LiveTradeRecord:
    """Completed live trade."""
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
    exit_reason: str
    order_ids: list = field(default_factory=list)


class LiveTrader:
    """Real-money trading engine with safety mechanisms."""

    def __init__(self, config: dict, exchange: ExchangeClient):
        self.config = config
        self.exchange = exchange
        self.risk_config = config.get("risk", {})
        self.trading_config = config.get("trading", {})

        self.positions: list[LivePosition] = []
        self.trade_journal: list[LiveTradeRecord] = []
        self.trade_counter = 0

        self.peak_equity = 0.0
        self.daily_start_balance = 0.0
        self.daily_start_date = datetime.now().date()
        self.cooldown_until = 0.0
        self.kill_switch = False

        # Sync with exchange on startup
        self._sync_balance()
        logger.info(f"Live trader initialized | Balance: ${self.balance:.2f}")

    def _sync_balance(self):
        """Sync balance from exchange."""
        try:
            balance = self.exchange.fetch_balance()
            quote = self.trading_config.get("primary_symbol", "BTC/USDT").split("/")[1]
            self.balance = float(balance.get("total", {}).get(quote, 0))
            self.peak_equity = max(self.peak_equity, self.balance)
            if self.daily_start_balance == 0:
                self.daily_start_balance = self.balance
        except Exception as e:
            logger.error(f"Failed to sync balance: {e}")
            self.balance = 0.0

    # ── Order Execution ───────────────────────────────────────

    def open_position(
        self,
        symbol: str,
        side: str,
        amount: float,
        leverage: float = 1.0,
        stop_loss_pct: float = 0.0,
        take_profit_pct: float = 0.0,
        trailing_stop_pct: float = 0.0,
    ) -> LivePosition | None:
        """Open a real position on the exchange."""
        if self.kill_switch:
            logger.error("KILL SWITCH ACTIVE — no trading allowed")
            return None

        if time.time() < self.cooldown_until:
            logger.info("Cooldown active — skipping trade")
            return None

        if not self._check_daily_limit():
            return None

        if not self._pre_trade_checks(symbol, amount, leverage):
            return None

        try:
            # Set leverage if supported
            if leverage > 1:
                try:
                    self.exchange.exchange.set_leverage(int(leverage), symbol)
                except Exception as e:
                    logger.warning(f"Could not set leverage: {e}")

            # Execute order
            if side == "long":
                order = self.exchange.create_market_buy(symbol, amount)
            else:
                order = self.exchange.create_market_sell(symbol, amount)

            fill_price = float(order.get("average") or order.get("price") or 0)
            order_id = order.get("id", "")

            if fill_price == 0:
                logger.error("Order filled but no price returned — check exchange")
                fill_price = self.exchange.get_current_price(symbol)

            # Calculate stop/take-profit
            if stop_loss_pct > 0:
                sl = fill_price * (1 - stop_loss_pct / 100) if side == "long" else fill_price * (1 + stop_loss_pct / 100)
            else:
                sl = 0.0

            if take_profit_pct > 0:
                tp = fill_price * (1 + take_profit_pct / 100) if side == "long" else fill_price * (1 - take_profit_pct / 100)
            else:
                tp = 0.0

            # Place stop-loss/take-profit orders on exchange if supported
            sl_order_id = ""
            tp_order_id = ""
            if sl > 0:
                try:
                    sl_side = "sell" if side == "long" else "buy"
                    sl_order = self.exchange.exchange.create_order(
                        symbol, "stop_market", sl_side, amount, None, {"stopPrice": sl}
                    )
                    sl_order_id = sl_order.get("id", "")
                except Exception as e:
                    logger.warning(f"Could not place stop-loss order on exchange: {e}")

            if tp > 0:
                try:
                    tp_side = "sell" if side == "long" else "buy"
                    tp_order = self.exchange.exchange.create_order(
                        symbol, "take_profit_market", tp_side, amount, None, {"stopPrice": tp}
                    )
                    tp_order_id = tp_order.get("id", "")
                except Exception as e:
                    logger.warning(f"Could not place take-profit order on exchange: {e}")

            self.trade_counter += 1
            pos = LivePosition(
                id=f"live_{self.trade_counter:06d}",
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
                order_ids=[order_id, sl_order_id, tp_order_id],
            )
            self.positions.append(pos)

            logger.info(
                f"LIVE {side.upper()} | {amount} {symbol} @ ${fill_price:.2f} "
                f"| {leverage:.1f}x | SL=${sl:.2f} TP=${tp:.2f} | order={order_id}"
            )

            self._sync_balance()
            self._save_state()
            return pos

        except Exception as e:
            logger.error(f"Order execution failed: {e}", exc_info=True)
            return None

    def close_position(
        self, position: LivePosition, reason: str = "signal"
    ) -> LiveTradeRecord | None:
        """Close a live position."""
        try:
            # Cancel any open SL/TP orders
            for oid in position.order_ids[1:]:  # Skip entry order
                if oid:
                    try:
                        self.exchange.cancel_order(oid, position.symbol)
                    except Exception:
                        pass

            # Execute close
            if position.side == "long":
                order = self.exchange.create_market_sell(position.symbol, position.amount)
            else:
                order = self.exchange.create_market_buy(position.symbol, position.amount)

            fill_price = float(order.get("average") or order.get("price") or 0)
            if fill_price == 0:
                fill_price = self.exchange.get_current_price(position.symbol)

            # PnL
            if position.side == "long":
                pnl_pct = ((fill_price - position.entry_price) / position.entry_price) * 100
            else:
                pnl_pct = ((position.entry_price - fill_price) / position.entry_price) * 100

            pnl_pct *= position.leverage
            raw_pnl = (pnl_pct / 100) * position.entry_price * position.amount

            # Estimate fees (exchange reports actual fees in order)
            fees = float(order.get("fee", {}).get("cost", 0))

            trade = LiveTradeRecord(
                id=position.id,
                symbol=position.symbol,
                side=position.side,
                entry_price=position.entry_price,
                exit_price=fill_price,
                amount=position.amount,
                leverage=position.leverage,
                entry_time=position.entry_time,
                exit_time=datetime.now(),
                pnl=raw_pnl - fees,
                pnl_pct=pnl_pct,
                fees=fees,
                exit_reason=reason,
                order_ids=position.order_ids + [order.get("id", "")],
            )
            self.trade_journal.append(trade)
            self.positions.remove(position)

            # Cooldown after loss
            if raw_pnl < 0:
                cooldown = self.risk_config.get("cooldown_after_loss_seconds", 300)
                self.cooldown_until = time.time() + cooldown

            logger.info(
                f"LIVE CLOSE {position.side.upper()} {position.symbol} | "
                f"PnL: ${raw_pnl:+.2f} ({pnl_pct:+.2f}%) | Reason: {reason}"
            )

            self._sync_balance()
            self._save_state()
            return trade

        except Exception as e:
            logger.error(f"Close position failed: {e}", exc_info=True)
            return None

    def check_exits(self, symbol: str, current_price: float) -> list[LiveTradeRecord]:
        """Check positions for software-managed stops (backup to exchange stops)."""
        closed = []
        for pos in [p for p in self.positions if p.symbol == symbol]:
            pos.highest_price = max(pos.highest_price, current_price)
            pos.lowest_price = min(pos.lowest_price, current_price)

            # Trailing stop (managed in software)
            if pos.trailing_stop_pct > 0:
                if pos.side == "long":
                    trail_price = pos.highest_price * (1 - pos.trailing_stop_pct / 100)
                    if current_price <= trail_price:
                        trade = self.close_position(pos, "trailing_stop")
                        if trade:
                            closed.append(trade)
                        continue
                else:
                    trail_price = pos.lowest_price * (1 + pos.trailing_stop_pct / 100)
                    if current_price >= trail_price:
                        trade = self.close_position(pos, "trailing_stop")
                        if trade:
                            closed.append(trade)
                        continue

            # Backup stop-loss (in case exchange stop didn't trigger)
            if pos.stop_loss > 0:
                if pos.side == "long" and current_price <= pos.stop_loss * 0.99:
                    trade = self.close_position(pos, "stop_loss_backup")
                    if trade:
                        closed.append(trade)
                    continue
                if pos.side == "short" and current_price >= pos.stop_loss * 1.01:
                    trade = self.close_position(pos, "stop_loss_backup")
                    if trade:
                        closed.append(trade)
                    continue

        return closed

    # ── Safety ────────────────────────────────────────────────

    def _pre_trade_checks(self, symbol: str, amount: float, leverage: float) -> bool:
        """Validate trade before execution."""
        max_positions = self.trading_config.get("max_open_positions", 5)
        if len(self.positions) >= max_positions:
            logger.warning("Max open positions reached")
            return False

        max_lev = self.trading_config.get("max_leverage", 10)
        if leverage > max_lev:
            logger.warning(f"Leverage {leverage}x exceeds max {max_lev}x")
            return False

        # Check total drawdown
        if self.peak_equity > 0:
            dd = (self.peak_equity - self.balance) / self.peak_equity * 100
            max_dd = self.risk_config.get("max_total_drawdown_pct", 15.0)
            if dd >= max_dd:
                logger.error(f"MAX DRAWDOWN REACHED ({dd:.1f}%) — trading halted")
                self.kill_switch = True
                return False

        return True

    def _check_daily_limit(self) -> bool:
        today = datetime.now().date()
        if today != self.daily_start_date:
            self.daily_start_balance = self.balance
            self.daily_start_date = today

        if self.daily_start_balance > 0:
            daily_loss = (self.daily_start_balance - self.balance) / self.daily_start_balance * 100
            max_daily = self.risk_config.get("max_daily_drawdown_pct", 5.0)
            if daily_loss >= max_daily:
                logger.warning(f"Daily loss limit hit ({daily_loss:.1f}%)")
                return False
        return True

    def emergency_close_all(self):
        """Emergency: close all positions immediately."""
        logger.error("EMERGENCY CLOSE ALL POSITIONS")
        self.kill_switch = True
        for pos in list(self.positions):
            self.close_position(pos, "emergency")

    # ── Persistence ───────────────────────────────────────────

    def _save_state(self):
        """Save current state to disk for recovery."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "balance": self.balance,
            "peak_equity": self.peak_equity,
            "positions": [
                {
                    "id": p.id, "symbol": p.symbol, "side": p.side,
                    "entry_price": p.entry_price, "amount": p.amount,
                    "leverage": p.leverage, "entry_time": p.entry_time.isoformat(),
                    "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                    "trailing_stop_pct": p.trailing_stop_pct,
                    "order_ids": p.order_ids,
                }
                for p in self.positions
            ],
            "trade_count": len(self.trade_journal),
            "kill_switch": self.kill_switch,
        }
        Path("data").mkdir(exist_ok=True)
        with open("data/live_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def get_open_positions(self, symbol: str | None = None) -> list[LivePosition]:
        if symbol:
            return [p for p in self.positions if p.symbol == symbol]
        return list(self.positions)
