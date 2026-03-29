"""Core trading engine — orchestrates the strategy, exchange, and risk manager."""

import logging
import time
from datetime import datetime

from bot.exchange import ExchangeClient
from bot.risk_manager import RiskManager
from bot.strategies import get_strategy
from bot.strategies.base import Signal

logger = logging.getLogger("trading_bot")

# Map timeframe strings to seconds for the sleep interval
TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600,
    "12h": 43200, "1d": 86400,
}


class Trader:
    """Main trading loop."""

    def __init__(self, config: dict):
        self.config = config
        self.exchange = ExchangeClient(config)
        self.risk_manager = RiskManager.from_config(config)
        self.strategy = get_strategy(config["trading"]["strategy"], config)
        self.symbol = config["trading"]["symbol"]
        self.timeframe = config["trading"]["timeframe"]
        self.trade_amount = config["trading"]["amount"]
        self.running = False

    def run_once(self):
        """Execute a single trading cycle."""
        logger.info(f"--- Cycle @ {datetime.now().strftime('%H:%M:%S')} ---")

        # 1. Fetch candle data
        df = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=100)
        current_price = float(df["close"].iloc[-1])
        logger.info(f"{self.symbol} price: {current_price}")

        # 2. Check existing positions for stop-loss / take-profit
        self._check_exits(current_price)

        # 3. Generate signal from strategy
        signal = self.strategy.analyze(df)
        indicators = self.strategy.get_indicator_values(df)
        logger.info(f"Strategy: {self.strategy.name} | Signal: {signal.value} | {indicators}")

        # 4. Act on signal
        if signal == Signal.BUY:
            self._handle_buy(current_price)
        elif signal == Signal.SELL:
            self._handle_sell(current_price)

    def _handle_buy(self, current_price: float):
        """Handle a BUY signal."""
        # Skip if we already have a long position for this symbol
        existing = self.risk_manager.get_positions_for_symbol(self.symbol)
        if any(p.side == "long" for p in existing):
            logger.info("Already in a long position — skipping BUY")
            return

        # Risk checks
        portfolio_value = self._get_portfolio_value()
        if not self.risk_manager.can_open_trade(portfolio_value):
            return
        if not self.risk_manager.check_position_size(
            self.trade_amount, current_price, portfolio_value
        ):
            return

        # Execute buy
        order = self.exchange.create_market_buy(self.symbol, self.trade_amount)
        fill_price = order.get("average") or current_price
        self.risk_manager.open_position(
            self.symbol, "long", float(fill_price), self.trade_amount, order.get("id", "")
        )

    def _handle_sell(self, current_price: float):
        """Handle a SELL signal — close any long positions."""
        positions = self.risk_manager.get_positions_for_symbol(self.symbol)
        longs = [p for p in positions if p.side == "long"]

        if not longs:
            logger.info("No long position to close — skipping SELL")
            return

        for pos in longs:
            order = self.exchange.create_market_sell(self.symbol, pos.amount)
            fill_price = order.get("average") or current_price
            self.risk_manager.close_position(pos, float(fill_price))

    def _check_exits(self, current_price: float):
        """Check all positions for stop-loss or take-profit."""
        positions = list(self.risk_manager.get_positions_for_symbol(self.symbol))
        for pos in positions:
            if self.risk_manager.check_stop_loss(pos, current_price):
                order = self.exchange.create_market_sell(self.symbol, pos.amount)
                fill_price = order.get("average") or current_price
                self.risk_manager.close_position(pos, float(fill_price))
            elif self.risk_manager.check_take_profit(pos, current_price):
                order = self.exchange.create_market_sell(self.symbol, pos.amount)
                fill_price = order.get("average") or current_price
                self.risk_manager.close_position(pos, float(fill_price))

    def _get_portfolio_value(self) -> float:
        """Estimate total portfolio value in quote currency."""
        try:
            balance = self.exchange.fetch_balance()
            # Use total USDT (or quote currency) as a rough estimate
            quote = self.symbol.split("/")[1]
            return float(balance.get("total", {}).get(quote, 0))
        except Exception as e:
            logger.warning(f"Could not fetch portfolio value: {e}")
            return 10000.0  # fallback default

    def run(self):
        """Run the bot in a continuous loop."""
        interval = TIMEFRAME_SECONDS.get(self.timeframe, 3600)
        self.running = True
        logger.info(
            f"Bot started | {self.symbol} | {self.timeframe} | "
            f"Strategy: {self.strategy.name} | Interval: {interval}s"
        )

        while self.running:
            try:
                self.run_once()
            except KeyboardInterrupt:
                logger.info("Shutting down...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Error in trading cycle: {e}", exc_info=True)

            if self.running:
                logger.info(f"Sleeping {interval}s until next cycle...")
                time.sleep(interval)

    def stop(self):
        """Stop the bot."""
        self.running = False
        logger.info("Bot stopped")
