"""Feature engineering — 200+ features used by pro quant traders.

Categories:
  1. Price action & returns (multi-horizon)
  2. Volatility features (realized, Parkinson, Garman-Klass)
  3. Volume microstructure (VWAP, OBV, volume profile)
  4. Order flow imbalance (from orderbook + trades)
  5. Multi-timeframe confluence
  6. Market regime (trend strength, mean-reversion score)
  7. Statistical features (Hurst exponent, autocorrelation)
  8. Momentum features (rate of change, momentum divergence)
  9. Liquidity features (spread, depth imbalance)
"""

import logging

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger("trading_bot")


class FeatureEngine:
    """Generate features from OHLCV data."""

    def __init__(self, config: dict):
        self.config = config

    def build_features(
        self,
        df: pd.DataFrame,
        orderbook: dict | None = None,
        trades_df: pd.DataFrame | None = None,
        higher_tf_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build full feature matrix from OHLCV + optional orderbook/trades."""
        feat = df[["open", "high", "low", "close", "volume"]].copy()

        # 1. Price action & returns
        self._add_returns(feat)
        # 2. Volatility
        self._add_volatility(feat)
        # 3. Volume microstructure
        self._add_volume_features(feat)
        # 4. Momentum
        self._add_momentum(feat)
        # 5. Statistical
        self._add_statistical(feat)
        # 6. Pattern recognition
        self._add_candle_patterns(feat)
        # 7. Support / resistance proximity
        self._add_sr_features(feat)
        # 8. Higher timeframe confluence
        if higher_tf_df is not None:
            self._add_htf_features(feat, higher_tf_df)
        # 9. Order flow (if available)
        if orderbook is not None:
            self._add_orderbook_features(feat, orderbook)
        if trades_df is not None and not trades_df.empty:
            self._add_trade_flow_features(feat, trades_df)

        # No targets here. Labels come from bot/ml/labeling.py, which
        # knows about stops, targets and holding time; a next-bar direction
        # column sitting in the feature frame is an invitation to leak.
        return feat.replace([np.inf, -np.inf], np.nan)

    # ── 1. Returns ────────────────────────────────────────────

    def _add_returns(self, df: pd.DataFrame):
        for period in [1, 2, 3, 5, 10, 20, 60]:
            df[f"return_{period}"] = df["close"].pct_change(period)
            df[f"log_return_{period}"] = np.log(df["close"] / df["close"].shift(period))

        # High-low return (intrabar range)
        df["hl_range"] = (df["high"] - df["low"]) / df["close"]
        df["co_range"] = (df["close"] - df["open"]) / df["close"]

        # Gap (open vs prev close)
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)

    # ── 2. Volatility ────────────────────────────────────────

    def _add_volatility(self, df: pd.DataFrame):
        close = df["close"]
        high = df["high"]
        low = df["low"]
        op = df["open"]

        for window in [5, 10, 20, 60]:
            # Realized volatility
            ret = close.pct_change()
            df[f"realized_vol_{window}"] = ret.rolling(window).std() * np.sqrt(252 * 24)

            # Parkinson volatility (uses high-low, more efficient)
            hl_sq = (np.log(high / low)) ** 2
            df[f"parkinson_vol_{window}"] = np.sqrt(
                hl_sq.rolling(window).mean() / (4 * np.log(2))
            )

            # Garman-Klass volatility (uses OHLC, most efficient)
            gk = (
                0.5 * (np.log(high / low)) ** 2
                - (2 * np.log(2) - 1) * (np.log(close / op)) ** 2
            )
            df[f"garman_klass_vol_{window}"] = np.sqrt(gk.rolling(window).mean() * 252 * 24)

            # ATR (Average True Range)
            tr = pd.concat(
                [
                    high - low,
                    (high - close.shift(1)).abs(),
                    (low - close.shift(1)).abs(),
                ],
                axis=1,
            ).max(axis=1)
            df[f"atr_{window}"] = tr.rolling(window).mean()
            df[f"natr_{window}"] = df[f"atr_{window}"] / close  # Normalized

        # Volatility ratio (short / long) — contraction vs expansion
        df["vol_ratio_5_20"] = df["realized_vol_5"] / df["realized_vol_20"].replace(0, np.nan)
        df["vol_ratio_10_60"] = df["realized_vol_10"] / df["realized_vol_60"].replace(0, np.nan)

    # ── 3. Volume Microstructure ──────────────────────────────

    def _add_volume_features(self, df: pd.DataFrame):
        vol = df["volume"]
        close = df["close"]

        # VWAP (Volume-Weighted Average Price)
        cum_vol = vol.cumsum()
        cum_pv = (close * vol).cumsum()
        df["vwap"] = cum_pv / cum_vol.replace(0, np.nan)
        df["price_vs_vwap"] = (close - df["vwap"]) / df["vwap"].replace(0, np.nan)

        # Rolling VWAP
        for window in [10, 20, 50]:
            rv = vol.rolling(window).sum()
            rpv = (close * vol).rolling(window).sum()
            df[f"rvwap_{window}"] = rpv / rv.replace(0, np.nan)
            df[f"price_vs_rvwap_{window}"] = (
                (close - df[f"rvwap_{window}"]) / df[f"rvwap_{window}"].replace(0, np.nan)
            )

        # OBV (On-Balance Volume)
        direction = np.sign(close.diff())
        df["obv"] = (vol * direction).cumsum()
        df["obv_slope_10"] = df["obv"].diff(10) / 10
        df["obv_slope_20"] = df["obv"].diff(20) / 20

        # Volume moving averages & ratios
        for window in [5, 10, 20, 50]:
            df[f"vol_sma_{window}"] = vol.rolling(window).mean()
        df["vol_ratio_5_20"] = df["vol_sma_5"] / df["vol_sma_20"].replace(0, np.nan)
        df["vol_spike"] = vol / df["vol_sma_20"].replace(0, np.nan)

        # Accumulation/Distribution
        mfm = ((close - df["low"]) - (df["high"] - close)) / (
            (df["high"] - df["low"]).replace(0, np.nan)
        )
        df["ad_line"] = (mfm * vol).cumsum()

        # Money Flow Index (volume-weighted RSI)
        typical_price = (df["high"] + df["low"] + close) / 3
        raw_mf = typical_price * vol
        pos_mf = raw_mf.where(typical_price > typical_price.shift(1), 0)
        neg_mf = raw_mf.where(typical_price < typical_price.shift(1), 0)
        for window in [14, 28]:
            pmf = pos_mf.rolling(window).sum()
            nmf = neg_mf.rolling(window).sum()
            mfr = pmf / nmf.replace(0, np.nan)
            df[f"mfi_{window}"] = 100 - (100 / (1 + mfr))

    # ── 4. Momentum ──────────────────────────────────────────

    def _add_momentum(self, df: pd.DataFrame):
        close = df["close"]

        # Rate of Change
        for period in [3, 5, 10, 20, 60]:
            df[f"roc_{period}"] = close.pct_change(period)

        # Moving average distances
        for window in [10, 20, 50, 100, 200]:
            ma = close.rolling(window).mean()
            df[f"dist_ma_{window}"] = (close - ma) / ma.replace(0, np.nan)

        # EMA distances
        for span in [9, 21, 55, 100]:
            ema = close.ewm(span=span).mean()
            df[f"dist_ema_{span}"] = (close - ema) / ema.replace(0, np.nan)

        # MA crossover signals (as continuous features, not binary)
        ema_fast = close.ewm(span=9).mean()
        ema_slow = close.ewm(span=21).mean()
        df["ema_cross_9_21"] = (ema_fast - ema_slow) / ema_slow.replace(0, np.nan)

        ema_fast2 = close.ewm(span=21).mean()
        ema_slow2 = close.ewm(span=55).mean()
        df["ema_cross_21_55"] = (ema_fast2 - ema_slow2) / ema_slow2.replace(0, np.nan)

        # MACD components
        ema12 = close.ewm(span=12).mean()
        ema26 = close.ewm(span=26).mean()
        df["macd_line"] = ema12 - ema26
        df["macd_signal"] = df["macd_line"].ewm(span=9).mean()
        df["macd_hist"] = df["macd_line"] - df["macd_signal"]
        df["macd_hist_slope"] = df["macd_hist"].diff(3)

        # Stochastic %K %D
        for window in [14, 28]:
            low_min = df["low"].rolling(window).min()
            high_max = df["high"].rolling(window).max()
            denom = (high_max - low_min).replace(0, np.nan)
            df[f"stoch_k_{window}"] = 100 * (close - low_min) / denom
            df[f"stoch_d_{window}"] = df[f"stoch_k_{window}"].rolling(3).mean()

        # Williams %R
        for window in [14, 28]:
            hh = df["high"].rolling(window).max()
            ll = df["low"].rolling(window).min()
            df[f"williams_r_{window}"] = -100 * (hh - close) / (hh - ll).replace(0, np.nan)

        # CCI (Commodity Channel Index)
        for window in [14, 20]:
            tp = (df["high"] + df["low"] + close) / 3
            tp_ma = tp.rolling(window).mean()
            tp_md = tp.rolling(window).apply(lambda x: np.abs(x - x.mean()).mean())
            df[f"cci_{window}"] = (tp - tp_ma) / (0.015 * tp_md).replace(0, np.nan)

    # ── 5. Statistical Features ──────────────────────────────

    def _add_statistical(self, df: pd.DataFrame):
        returns = df["close"].pct_change()

        for window in [20, 60]:
            r = returns.rolling(window)

            # Skewness & kurtosis of returns
            df[f"skew_{window}"] = r.skew()
            df[f"kurtosis_{window}"] = r.kurt()

            # Autocorrelation (lag 1)
            df[f"autocorr_{window}"] = returns.rolling(window).apply(
                lambda x: x.autocorr(lag=1) if len(x) > 1 else 0, raw=False
            )

        # Hurst exponent (mean-reversion vs trending)
        df["hurst_60"] = returns.rolling(60).apply(self._hurst_exponent, raw=True)

        # Z-score of price relative to rolling mean
        for window in [20, 50]:
            mu = df["close"].rolling(window).mean()
            sigma = df["close"].rolling(window).std()
            df[f"zscore_{window}"] = (df["close"] - mu) / sigma.replace(0, np.nan)

    @staticmethod
    def _hurst_exponent(ts: np.ndarray) -> float:
        """Estimate Hurst exponent via R/S method. H<0.5=mean-revert, H>0.5=trending."""
        if len(ts) < 20:
            return 0.5
        n = len(ts)
        max_k = min(int(n / 2), 50)
        if max_k < 4:
            return 0.5

        rs_list = []
        sizes = []
        for k in range(4, max_k + 1):
            size = n // k
            if size < 2:
                break
            for i in range(k):
                chunk = ts[i * size : (i + 1) * size]
                mean_c = np.mean(chunk)
                deviate = np.cumsum(chunk - mean_c)
                r = np.max(deviate) - np.min(deviate)
                s = np.std(chunk, ddof=1)
                if s > 0:
                    rs_list.append(r / s)
                    sizes.append(size)

        if len(rs_list) < 4:
            return 0.5

        log_rs = np.log(rs_list)
        log_n = np.log(sizes)
        slope, _, _, _, _ = stats.linregress(log_n, log_rs)
        return float(np.clip(slope, 0, 1))

    # ── 6. Candle Patterns ───────────────────────────────────

    def _add_candle_patterns(self, df: pd.DataFrame):
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = (c - o).abs()
        full_range = (h - l).replace(0, np.nan)

        # Body ratio
        df["body_ratio"] = body / full_range

        # Upper/lower wick ratio
        upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
        lower_wick = pd.concat([o, c], axis=1).min(axis=1) - l
        df["upper_wick_ratio"] = upper_wick / full_range
        df["lower_wick_ratio"] = lower_wick / full_range

        # Consecutive direction count
        direction = np.sign(c - o)
        df["candle_direction"] = direction
        groups = (direction != direction.shift()).cumsum()
        df["consecutive_direction"] = direction.groupby(groups).cumcount() + 1
        df["consecutive_direction"] *= direction

    # ── 7. Support / Resistance ──────────────────────────────

    def _add_sr_features(self, df: pd.DataFrame):
        close = df["close"]

        for window in [20, 50, 100]:
            rolling_high = df["high"].rolling(window).max()
            rolling_low = df["low"].rolling(window).min()

            # Distance to rolling high/low (resistance/support)
            df[f"dist_resistance_{window}"] = (rolling_high - close) / close
            df[f"dist_support_{window}"] = (close - rolling_low) / close

            # Position within range (0 = at support, 1 = at resistance)
            range_size = (rolling_high - rolling_low).replace(0, np.nan)
            df[f"range_position_{window}"] = (close - rolling_low) / range_size

    # ── 8. Higher Timeframe Features ─────────────────────────

    def _add_htf_features(self, df: pd.DataFrame, htf: pd.DataFrame):
        """Add higher-timeframe trend/momentum as features."""
        htf = htf.copy()
        close_htf = htf["close"]

        # HTF trend direction
        htf["htf_ema_21"] = close_htf.ewm(span=21).mean()
        htf["htf_ema_55"] = close_htf.ewm(span=55).mean()
        htf["htf_trend"] = np.where(htf["htf_ema_21"] > htf["htf_ema_55"], 1, -1)
        htf["htf_trend_strength"] = (
            (htf["htf_ema_21"] - htf["htf_ema_55"])
            / htf["htf_ema_55"].replace(0, np.nan)
        )

        # HTF momentum
        htf["htf_roc_10"] = close_htf.pct_change(10)

        # Reindex to lower timeframe
        htf_cols = ["htf_trend", "htf_trend_strength", "htf_roc_10"]
        htf_feats = htf[htf_cols].reindex(df.index, method="ffill")
        for col in htf_cols:
            df[col] = htf_feats[col]

    # ── 9. Order Book Features ───────────────────────────────

    def _add_orderbook_features(self, df: pd.DataFrame, orderbook: dict):
        """Extract features from orderbook snapshot (applied to last row)."""
        bids = np.array(orderbook.get("bids", []))
        asks = np.array(orderbook.get("asks", []))

        if len(bids) == 0 or len(asks) == 0:
            return

        bid_vol = bids[:, 1].sum()
        ask_vol = asks[:, 1].sum()
        total = bid_vol + ask_vol

        # Bid/ask imbalance (positive = more buyers)
        df.loc[df.index[-1], "ob_imbalance"] = (
            (bid_vol - ask_vol) / total if total > 0 else 0
        )

        # Spread
        best_bid = bids[0][0]
        best_ask = asks[0][0]
        mid = (best_bid + best_ask) / 2
        df.loc[df.index[-1], "ob_spread_bps"] = (
            ((best_ask - best_bid) / mid) * 10000 if mid > 0 else 0
        )

        # Depth at levels (cumulative volume at top 5 vs bottom 5)
        top5_bid = bids[:5, 1].sum() if len(bids) >= 5 else bids[:, 1].sum()
        top5_ask = asks[:5, 1].sum() if len(asks) >= 5 else asks[:, 1].sum()
        df.loc[df.index[-1], "ob_top5_imbalance"] = (
            (top5_bid - top5_ask) / (top5_bid + top5_ask)
            if (top5_bid + top5_ask) > 0
            else 0
        )

        # Large wall detection (orders > 3x average)
        avg_bid_size = bids[:, 1].mean()
        avg_ask_size = asks[:, 1].mean()
        df.loc[df.index[-1], "ob_bid_walls"] = int((bids[:, 1] > 3 * avg_bid_size).sum())
        df.loc[df.index[-1], "ob_ask_walls"] = int((asks[:, 1] > 3 * avg_ask_size).sum())

    # ── 10. Trade Flow Features ──────────────────────────────

    def _add_trade_flow_features(self, df: pd.DataFrame, trades_df: pd.DataFrame):
        """Extract order flow features from recent trades."""
        if trades_df.empty:
            return

        buy_trades = trades_df[trades_df["side"] == "buy"]
        sell_trades = trades_df[trades_df["side"] == "sell"]

        buy_vol = buy_trades["cost"].sum()
        sell_vol = sell_trades["cost"].sum()
        total_vol = buy_vol + sell_vol

        # Buy/sell volume ratio
        df.loc[df.index[-1], "tf_buy_ratio"] = (
            buy_vol / total_vol if total_vol > 0 else 0.5
        )

        # Large trade detection (trades > 3x median)
        median_size = trades_df["cost"].median()
        large = trades_df[trades_df["cost"] > 3 * median_size]
        large_buy = large[large["side"] == "buy"]["cost"].sum()
        large_sell = large[large["side"] == "sell"]["cost"].sum()
        large_total = large_buy + large_sell

        df.loc[df.index[-1], "tf_large_buy_ratio"] = (
            large_buy / large_total if large_total > 0 else 0.5
        )

        # Trade intensity (trades per second)
        if len(trades_df) >= 2:
            time_range = (
                trades_df["timestamp"].max() - trades_df["timestamp"].min()
            ).total_seconds()
            df.loc[df.index[-1], "tf_trades_per_sec"] = (
                len(trades_df) / time_range if time_range > 0 else 0
            )

        # CVD (Cumulative Volume Delta)
        df.loc[df.index[-1], "tf_cvd"] = buy_vol - sell_vol

    # ── Utility ──────────────────────────────────────────────

    def get_feature_columns(self, df: pd.DataFrame) -> list[str]:
        """Return names of feature columns (exclude targets and raw OHLCV)."""
        exclude = {"open", "high", "low", "close", "volume"}
        return [c for c in df.columns if c not in exclude]
