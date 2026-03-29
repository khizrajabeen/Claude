"""Data collection and preprocessing pipeline.

Collects multi-timeframe OHLCV, orderbook snapshots, and trade data
from the exchange, then stores locally for training.
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")

DATA_DIR = Path("data")


class DataPipeline:
    """Collect, store, and serve market data."""

    def __init__(self, exchange_client, config: dict):
        self.exchange = exchange_client
        self.config = config
        self.data_config = config["data"]
        DATA_DIR.mkdir(exist_ok=True)

    # ── Collection ────────────────────────────────────────────

    def fetch_historical(
        self, symbol: str, timeframe: str, days: int = 90
    ) -> pd.DataFrame:
        """Fetch historical OHLCV with pagination (exchange rate-limit safe)."""
        all_candles = []
        tf_ms = self._timeframe_to_ms(timeframe)
        since = int((time.time() - days * 86400) * 1000)
        now_ms = int(time.time() * 1000)

        logger.info(f"Fetching {days}d of {symbol} {timeframe} data...")

        while since < now_ms:
            try:
                candles = self.exchange.exchange.fetch_ohlcv(
                    symbol, timeframe, since=since, limit=1000
                )
            except Exception as e:
                logger.warning(f"Fetch error (retrying): {e}")
                time.sleep(2)
                continue

            if not candles:
                break

            all_candles.extend(candles)
            since = candles[-1][0] + tf_ms
            time.sleep(self.exchange.exchange.rateLimit / 1000)

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.drop_duplicates(subset="timestamp").set_index("timestamp").sort_index()
        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df

    def fetch_orderbook(self, symbol: str) -> dict:
        """Fetch current orderbook snapshot."""
        depth = self.data_config.get("orderbook_depth", 20)
        return self.exchange.exchange.fetch_order_book(symbol, limit=depth)

    def fetch_recent_trades(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Fetch recent trades for order flow analysis."""
        trades = self.exchange.exchange.fetch_trades(symbol, limit=limit)
        df = pd.DataFrame(
            [
                {
                    "timestamp": t["timestamp"],
                    "price": t["price"],
                    "amount": t["amount"],
                    "side": t["side"],
                    "cost": t["cost"],
                }
                for t in trades
            ]
        )
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        return df

    def collect_all_timeframes(self, symbol: str) -> dict[str, pd.DataFrame]:
        """Fetch historical data for all configured timeframes."""
        days = self.data_config.get("lookback_days", 90)
        result = {}
        for tf in self.data_config["timeframes"]:
            result[tf] = self.fetch_historical(symbol, tf, days=days)
            self._save_df(result[tf], symbol, tf)
        return result

    # ── Storage ───────────────────────────────────────────────

    def _save_df(self, df: pd.DataFrame, symbol: str, timeframe: str):
        safe_symbol = symbol.replace("/", "_")
        path = DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"
        df.to_parquet(path)
        logger.debug(f"Saved {path}")

    def load_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        safe_symbol = symbol.replace("/", "_")
        path = DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"
        if path.exists():
            return pd.read_parquet(path)
        return self.fetch_historical(
            symbol, timeframe, self.data_config.get("lookback_days", 90)
        )

    # ── Preprocessing ─────────────────────────────────────────

    @staticmethod
    def clean(df: pd.DataFrame) -> pd.DataFrame:
        """Remove NaN rows and outliers (>5 sigma moves)."""
        df = df.dropna()
        returns = df["close"].pct_change()
        mu, sigma = returns.mean(), returns.std()
        mask = returns.abs() < mu + 5 * sigma
        # Keep first row which has NaN return
        mask.iloc[0] = True
        return df[mask].copy()

    @staticmethod
    def _timeframe_to_ms(tf: str) -> int:
        units = {"m": 60_000, "h": 3_600_000, "d": 86_400_000}
        for suffix, ms in units.items():
            if tf.endswith(suffix):
                return int(tf[: -len(suffix)]) * ms
        return 3_600_000
