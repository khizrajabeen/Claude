"""Auto data download and management for multi-coin training.

Downloads and caches historical data for all configured coins,
auto-detects the top coins by volume, and keeps data fresh.
"""

import logging
import time
from pathlib import Path

import pandas as pd

from bot.ml.data_pipeline import DataPipeline

logger = logging.getLogger("trading_bot")

DATA_DIR = Path("data")

# Top coins by market cap / volume — used when user doesn't specify
DEFAULT_COINS = [
    "BTC/USDT", "ETH/USDT", "SOL/USDT", "BNB/USDT", "XRP/USDT",
    "ADA/USDT", "DOGE/USDT", "AVAX/USDT", "DOT/USDT", "MATIC/USDT",
    "LINK/USDT", "UNI/USDT", "ATOM/USDT", "LTC/USDT", "ARB/USDT",
]


class AutoDataManager:
    """Automatically downloads and manages market data for training."""

    def __init__(self, config: dict, data_pipeline: DataPipeline):
        self.config = config
        self.pipeline = data_pipeline
        DATA_DIR.mkdir(exist_ok=True)

    def discover_top_coins(self, top_n: int = 15) -> list[str]:
        """Auto-discover top coins by 24h volume on the exchange."""
        try:
            logger.info("Discovering top coins by volume...")
            tickers = self.pipeline.exchange.exchange.fetch_tickers()

            # Filter to USDT pairs only
            usdt_pairs = {
                symbol: data
                for symbol, data in tickers.items()
                if symbol.endswith("/USDT") and data.get("quoteVolume")
            }

            # Sort by volume
            sorted_pairs = sorted(
                usdt_pairs.items(),
                key=lambda x: float(x[1].get("quoteVolume", 0)),
                reverse=True,
            )

            top_symbols = [symbol for symbol, _ in sorted_pairs[:top_n]]
            logger.info(f"Top {top_n} coins by volume: {top_symbols}")
            return top_symbols

        except Exception as e:
            logger.warning(f"Could not discover coins: {e}. Using defaults.")
            return DEFAULT_COINS[:top_n]

    def download_all(
        self,
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        days: int | None = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Download historical data for all coins and timeframes.

        Returns:
            Nested dict: {symbol: {timeframe: DataFrame}}
        """
        symbols = symbols or self.config.get("data", {}).get("symbols", DEFAULT_COINS[:5])
        timeframes = timeframes or self.config.get("data", {}).get("timeframes", ["1h"])
        days = days or self.config.get("data", {}).get("lookback_days", 90)

        all_data = {}
        total = len(symbols) * len(timeframes)
        done = 0

        logger.info(f"Downloading data: {len(symbols)} coins × {len(timeframes)} timeframes = {total} datasets")

        for symbol in symbols:
            all_data[symbol] = {}
            for tf in timeframes:
                done += 1
                cache_path = self._cache_path(symbol, tf)

                # Check cache freshness
                if self._is_cache_fresh(cache_path, tf):
                    logger.info(f"[{done}/{total}] {symbol} {tf}: cached (fresh)")
                    all_data[symbol][tf] = pd.read_parquet(cache_path)
                    continue

                logger.info(f"[{done}/{total}] {symbol} {tf}: downloading...")
                try:
                    df = self.pipeline.fetch_historical(symbol, tf, days=days)
                    df = self.pipeline.clean(df)

                    if len(df) > 0:
                        df.to_parquet(cache_path)
                        all_data[symbol][tf] = df
                        logger.info(f"  → {len(df)} candles saved")
                    else:
                        logger.warning(f"  → No data returned for {symbol} {tf}")

                except Exception as e:
                    logger.warning(f"  → Failed: {e}")
                    # Try to load stale cache
                    if cache_path.exists():
                        all_data[symbol][tf] = pd.read_parquet(cache_path)
                        logger.info(f"  → Using stale cache")

                # Rate limit courtesy
                time.sleep(0.5)

        successful = sum(len(tfs) for tfs in all_data.values())
        logger.info(f"Downloaded {successful}/{total} datasets")
        return all_data

    def get_combined_training_data(
        self,
        symbols: list[str] | None = None,
        timeframe: str = "1h",
    ) -> pd.DataFrame:
        """Combine data from multiple coins into one training DataFrame.

        Adds a 'symbol' column so the model can learn cross-asset patterns.
        This is key: training on multiple coins gives the model more data
        and helps it generalize better than single-coin training.
        """
        symbols = symbols or self.config.get("data", {}).get("symbols", DEFAULT_COINS[:5])
        dfs = []

        for symbol in symbols:
            cache_path = self._cache_path(symbol, timeframe)
            if cache_path.exists():
                df = pd.read_parquet(cache_path)
                df["symbol"] = symbol
                # Normalize prices to returns so different price scales don't matter
                df["norm_close"] = df["close"].pct_change().cumsum()
                dfs.append(df)
            else:
                logger.warning(f"No cached data for {symbol} {timeframe}")

        if not dfs:
            raise ValueError("No data available. Run download first.")

        combined = pd.concat(dfs, axis=0).sort_index()
        logger.info(
            f"Combined training data: {len(combined)} candles from {len(dfs)} coins"
        )
        return combined

    def refresh_stale_data(self, max_age_hours: int = 24):
        """Re-download any data that's older than max_age_hours."""
        symbols = self.config.get("data", {}).get("symbols", [])
        timeframes = self.config.get("data", {}).get("timeframes", [])
        refreshed = 0

        for symbol in symbols:
            for tf in timeframes:
                cache_path = self._cache_path(symbol, tf)
                if not self._is_cache_fresh(cache_path, tf, max_age_hours):
                    try:
                        df = self.pipeline.fetch_historical(symbol, tf, days=7)
                        df = self.pipeline.clean(df)
                        if len(df) > 0:
                            # Merge with existing cache
                            if cache_path.exists():
                                old = pd.read_parquet(cache_path)
                                df = pd.concat([old, df]).drop_duplicates().sort_index()
                            df.to_parquet(cache_path)
                            refreshed += 1
                    except Exception as e:
                        logger.debug(f"Refresh failed for {symbol} {tf}: {e}")
                    time.sleep(0.3)

        if refreshed > 0:
            logger.info(f"Refreshed {refreshed} datasets")

    def _cache_path(self, symbol: str, timeframe: str) -> Path:
        safe_symbol = symbol.replace("/", "_")
        return DATA_DIR / f"{safe_symbol}_{timeframe}.parquet"

    def _is_cache_fresh(
        self, path: Path, timeframe: str, max_age_hours: int | None = None
    ) -> bool:
        """Check if cached data is recent enough."""
        if not path.exists():
            return False

        import os
        age_hours = (time.time() - os.path.getmtime(path)) / 3600

        if max_age_hours:
            return age_hours < max_age_hours

        # Auto: freshness depends on timeframe
        freshness = {
            "1m": 1, "5m": 4, "15m": 8, "1h": 24, "4h": 48, "1d": 168,
        }
        max_h = freshness.get(timeframe, 24)
        return age_hours < max_h
