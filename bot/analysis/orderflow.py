"""Order flow and market microstructure analysis.

What pro traders actually look at:
  - CVD (Cumulative Volume Delta) — net buying vs selling pressure
  - Absorption — large orders that absorb selling without price drop
  - Liquidity sweeps — stop hunts, fake breakouts
  - Whale detection — unusually large orders
  - Footprint analysis — volume at each price level
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")


class OrderFlowAnalyzer:
    """Analyze order flow for institutional activity detection."""

    def __init__(self, config: dict):
        self.config = config

    def analyze_trades(self, trades_df: pd.DataFrame) -> dict:
        """Full order flow analysis from recent trades."""
        if trades_df.empty:
            return self._empty_result()

        result = {}

        # 1. CVD (Cumulative Volume Delta)
        result.update(self._calc_cvd(trades_df))

        # 2. Buy/sell pressure
        result.update(self._calc_pressure(trades_df))

        # 3. Whale detection
        result.update(self._detect_whales(trades_df))

        # 4. Trade clustering (institutional footprint)
        result.update(self._detect_clustering(trades_df))

        # 5. Aggression ratio
        result.update(self._calc_aggression(trades_df))

        return result

    def analyze_orderbook(self, orderbook: dict) -> dict:
        """Analyze orderbook for support/resistance and manipulation."""
        bids = np.array(orderbook.get("bids", []))
        asks = np.array(orderbook.get("asks", []))

        if len(bids) == 0 or len(asks) == 0:
            return {}

        result = {}

        # Bid/ask imbalance
        bid_vol = bids[:, 1].sum()
        ask_vol = asks[:, 1].sum()
        total = bid_vol + ask_vol
        result["imbalance"] = (bid_vol - ask_vol) / total if total > 0 else 0

        # Spread
        best_bid, best_ask = bids[0][0], asks[0][0]
        mid = (best_bid + best_ask) / 2
        result["spread_bps"] = ((best_ask - best_bid) / mid * 10000) if mid > 0 else 0

        # Depth analysis at different levels
        for depth in [5, 10, 20]:
            d = min(depth, len(bids), len(asks))
            b = bids[:d, 1].sum()
            a = asks[:d, 1].sum()
            t = b + a
            result[f"imbalance_top{depth}"] = (b - a) / t if t > 0 else 0

        # Wall detection — orders significantly larger than average
        avg_bid = bids[:, 1].mean()
        avg_ask = asks[:, 1].mean()
        bid_walls = bids[bids[:, 1] > avg_bid * 5]
        ask_walls = asks[asks[:, 1] > avg_ask * 5]

        result["bid_walls"] = len(bid_walls)
        result["ask_walls"] = len(ask_walls)

        if len(bid_walls) > 0:
            result["strongest_bid_wall_price"] = float(bid_walls[np.argmax(bid_walls[:, 1])][0])
            result["strongest_bid_wall_size"] = float(bid_walls[:, 1].max())
        if len(ask_walls) > 0:
            result["strongest_ask_wall_price"] = float(ask_walls[np.argmax(ask_walls[:, 1])][0])
            result["strongest_ask_wall_size"] = float(ask_walls[:, 1].max())

        # Spoofing detection — walls far from mid price with unusual size
        if len(bid_walls) > 0:
            wall_distances = np.abs(bid_walls[:, 0] - mid) / mid * 100
            result["suspicious_bid_walls"] = int((wall_distances > 1).sum())
        if len(ask_walls) > 0:
            wall_distances = np.abs(ask_walls[:, 0] - mid) / mid * 100
            result["suspicious_ask_walls"] = int((wall_distances > 1).sum())

        return result

    def _calc_cvd(self, df: pd.DataFrame) -> dict:
        """Cumulative Volume Delta — net buying vs selling."""
        buy_vol = df[df["side"] == "buy"]["cost"].sum()
        sell_vol = df[df["side"] == "sell"]["cost"].sum()
        cvd = buy_vol - sell_vol
        total = buy_vol + sell_vol

        return {
            "cvd": cvd,
            "cvd_normalized": cvd / total if total > 0 else 0,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
        }

    def _calc_pressure(self, df: pd.DataFrame) -> dict:
        """Buy/sell pressure analysis."""
        buys = df[df["side"] == "buy"]
        sells = df[df["side"] == "sell"]

        buy_count = len(buys)
        sell_count = len(sells)
        total = buy_count + sell_count

        # Volume-weighted average prices
        buy_vwap = (buys["price"] * buys["amount"]).sum() / buys["amount"].sum() if len(buys) > 0 else 0
        sell_vwap = (sells["price"] * sells["amount"]).sum() / sells["amount"].sum() if len(sells) > 0 else 0

        return {
            "buy_pressure": buy_count / total if total > 0 else 0.5,
            "buy_vwap": buy_vwap,
            "sell_vwap": sell_vwap,
        }

    def _detect_whales(self, df: pd.DataFrame) -> dict:
        """Detect whale activity (unusually large trades)."""
        if len(df) < 10:
            return {"whale_trades": 0, "whale_bias": 0}

        # Whale = trade > 95th percentile
        threshold = df["cost"].quantile(0.95)
        whales = df[df["cost"] > threshold]

        whale_buy = whales[whales["side"] == "buy"]["cost"].sum()
        whale_sell = whales[whales["side"] == "sell"]["cost"].sum()
        whale_total = whale_buy + whale_sell

        return {
            "whale_trades": len(whales),
            "whale_volume_pct": whales["cost"].sum() / df["cost"].sum() if df["cost"].sum() > 0 else 0,
            "whale_bias": (whale_buy - whale_sell) / whale_total if whale_total > 0 else 0,
        }

    def _detect_clustering(self, df: pd.DataFrame) -> dict:
        """Detect trade clustering (institutional iceberg orders)."""
        if len(df) < 20:
            return {"cluster_score": 0}

        # Look for sequences of similar-sized trades at similar prices
        sizes = df["amount"].values
        prices = df["price"].values

        # Rolling coefficient of variation (low CV = uniform sizes = likely iceberg)
        window = min(20, len(sizes))
        if window < 5:
            return {"cluster_score": 0}

        cv_size = pd.Series(sizes).rolling(window).std() / pd.Series(sizes).rolling(window).mean()
        cv_price = pd.Series(prices).rolling(window).std() / pd.Series(prices).rolling(window).mean()

        cv_size_val = cv_size.iloc[-1] if not np.isnan(cv_size.iloc[-1]) else 1.0
        cv_price_val = cv_price.iloc[-1] if not np.isnan(cv_price.iloc[-1]) else 1.0

        # Low CV in both = likely iceberg / institutional
        cluster_score = max(0, 1 - cv_size_val) * max(0, 1 - cv_price_val * 100)

        return {"cluster_score": float(cluster_score)}

    def _calc_aggression(self, df: pd.DataFrame) -> dict:
        """Aggression ratio — how much volume is market orders vs limit fills."""
        if len(df) < 10:
            return {"aggression_ratio": 0.5}

        # Market buys (taker buys hitting asks) vs market sells
        buy_vol = df[df["side"] == "buy"]["cost"].sum()
        sell_vol = df[df["side"] == "sell"]["cost"].sum()
        total = buy_vol + sell_vol

        return {
            "aggression_ratio": buy_vol / total if total > 0 else 0.5,
        }

    def _empty_result(self) -> dict:
        return {
            "cvd": 0, "cvd_normalized": 0,
            "buy_volume": 0, "sell_volume": 0,
            "buy_pressure": 0.5,
            "whale_trades": 0, "whale_bias": 0,
            "cluster_score": 0,
            "aggression_ratio": 0.5,
        }
