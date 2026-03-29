"""Exchange client wrapper using ccxt."""

import logging
import time

import ccxt
import pandas as pd

logger = logging.getLogger("trading_bot")


class ExchangeClient:
    """Wrapper around ccxt for exchange operations."""

    def __init__(self, config: dict):
        self.config = config["exchange"]
        self.exchange = self._init_exchange()

    def _init_exchange(self) -> ccxt.Exchange:
        """Initialize the ccxt exchange instance."""
        exchange_id = self.config["name"].lower()
        if exchange_id not in ccxt.exchanges:
            raise ValueError(
                f"Exchange '{exchange_id}' not supported. "
                f"Available: {', '.join(ccxt.exchanges[:20])}..."
            )

        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class(
            {
                "apiKey": self.config.get("api_key", ""),
                "secret": self.config.get("api_secret", ""),
                "enableRateLimit": True,
            }
        )

        if self.config.get("sandbox", True):
            exchange.set_sandbox_mode(True)
            logger.info("Running in SANDBOX mode")

        return exchange

    def fetch_ohlcv(
        self, symbol: str, timeframe: str, limit: int = 100
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data and return as a DataFrame."""
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(
            raw, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        return df

    def fetch_ticker(self, symbol: str) -> dict:
        """Fetch the current ticker for a symbol."""
        return self.exchange.fetch_ticker(symbol)

    def fetch_balance(self) -> dict:
        """Fetch account balance."""
        return self.exchange.fetch_balance()

    def create_market_buy(self, symbol: str, amount: float) -> dict:
        """Place a market buy order."""
        logger.info(f"BUY  {amount} {symbol} @ market")
        order = self.exchange.create_market_buy_order(symbol, amount)
        logger.info(f"Order filled: {order['id']} | avg price: {order.get('average')}")
        return order

    def create_market_sell(self, symbol: str, amount: float) -> dict:
        """Place a market sell order."""
        logger.info(f"SELL {amount} {symbol} @ market")
        order = self.exchange.create_market_sell_order(symbol, amount)
        logger.info(f"Order filled: {order['id']} | avg price: {order.get('average')}")
        return order

    def create_limit_buy(self, symbol: str, amount: float, price: float) -> dict:
        """Place a limit buy order."""
        logger.info(f"BUY  {amount} {symbol} @ {price}")
        return self.exchange.create_limit_buy_order(symbol, amount, price)

    def create_limit_sell(self, symbol: str, amount: float, price: float) -> dict:
        """Place a limit sell order."""
        logger.info(f"SELL {amount} {symbol} @ {price}")
        return self.exchange.create_limit_sell_order(symbol, amount, price)

    def fetch_open_orders(self, symbol: str) -> list:
        """Fetch open orders for a symbol."""
        return self.exchange.fetch_open_orders(symbol)

    def cancel_order(self, order_id: str, symbol: str) -> dict:
        """Cancel an open order."""
        logger.info(f"Cancelling order {order_id}")
        return self.exchange.cancel_order(order_id, symbol)

    def get_current_price(self, symbol: str) -> float:
        """Get the current price for a symbol."""
        ticker = self.fetch_ticker(symbol)
        return float(ticker["last"])
