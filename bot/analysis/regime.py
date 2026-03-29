"""Market regime detection.

Classifies the market into regimes so the bot adapts its behavior:
  - TRENDING_UP: Strong uptrend, momentum strategies work
  - TRENDING_DOWN: Strong downtrend, short or stay flat
  - RANGING: Sideways, mean-reversion works
  - VOLATILE: High volatility, reduce position sizes
  - QUIET: Low volatility, breakout watch

Uses multiple signals: ADX, Hurst exponent, volatility percentile,
and price structure analysis.
"""

import logging
from enum import Enum

import numpy as np
import pandas as pd

logger = logging.getLogger("trading_bot")


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    QUIET = "quiet"


class RegimeDetector:
    """Detect current market regime using multiple signals."""

    def __init__(self, config: dict):
        self.config = config

    def detect(self, df: pd.DataFrame) -> dict:
        """Analyze market and return regime + confidence."""
        if len(df) < 60:
            return {"regime": MarketRegime.QUIET, "confidence": 0.0, "details": {}}

        close = df["close"]
        high = df["high"]
        low = df["low"]

        # 1. ADX — trend strength
        adx, plus_di, minus_di = self._calc_adx(high, low, close, period=14)

        # 2. Hurst exponent
        hurst = self._calc_hurst(close.values[-60:])

        # 3. Volatility percentile
        returns = close.pct_change().dropna()
        current_vol = returns.iloc[-20:].std()
        vol_history = returns.rolling(20).std().dropna()
        vol_percentile = (vol_history < current_vol).mean() if len(vol_history) > 0 else 0.5

        # 4. Trend direction (EMA slope)
        ema_21 = close.ewm(span=21).mean()
        ema_55 = close.ewm(span=55).mean()
        trend_direction = 1 if ema_21.iloc[-1] > ema_55.iloc[-1] else -1
        trend_strength = abs(ema_21.iloc[-1] - ema_55.iloc[-1]) / ema_55.iloc[-1]

        # 5. Price structure (higher highs/lows vs lower)
        structure = self._analyze_structure(df)

        # Decision logic
        adx_val = adx.iloc[-1] if not np.isnan(adx.iloc[-1]) else 20

        details = {
            "adx": float(adx_val),
            "hurst": float(hurst),
            "vol_percentile": float(vol_percentile),
            "trend_direction": trend_direction,
            "trend_strength": float(trend_strength),
            "structure": structure,
        }

        regime, confidence = self._classify(adx_val, hurst, vol_percentile,
                                             trend_direction, trend_strength)

        logger.debug(f"Regime: {regime.value} (conf={confidence:.2f}) | {details}")
        return {"regime": regime, "confidence": confidence, "details": details}

    def _classify(self, adx, hurst, vol_pct, trend_dir, trend_str) -> tuple:
        """Classify regime from indicators."""
        # Strong trend
        if adx > 30 and trend_str > 0.01:
            if trend_dir > 0:
                conf = min(1.0, (adx - 25) / 30 + trend_str * 10)
                return MarketRegime.TRENDING_UP, conf
            else:
                conf = min(1.0, (adx - 25) / 30 + trend_str * 10)
                return MarketRegime.TRENDING_DOWN, conf

        # High volatility (but no clear trend)
        if vol_pct > 0.85:
            return MarketRegime.VOLATILE, min(1.0, vol_pct)

        # Mean-reverting (Hurst < 0.4 or low ADX)
        if hurst < 0.4 or adx < 20:
            conf = max(0.5, 1 - hurst) if hurst < 0.5 else 0.5
            return MarketRegime.RANGING, conf

        # Quiet (low vol)
        if vol_pct < 0.2:
            return MarketRegime.QUIET, 0.7

        # Default: ranging
        return MarketRegime.RANGING, 0.5

    def _calc_adx(self, high, low, close, period=14):
        """Calculate ADX (Average Directional Index)."""
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period).mean()
        plus_di = 100 * (plus_dm.ewm(span=period).mean() / atr.replace(0, np.nan))
        minus_di = 100 * (minus_dm.ewm(span=period).mean() / atr.replace(0, np.nan))

        dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan))
        adx = dx.ewm(span=period).mean()

        return adx, plus_di, minus_di

    def _calc_hurst(self, ts: np.ndarray) -> float:
        """Simplified Hurst exponent."""
        if len(ts) < 20:
            return 0.5
        returns = np.diff(ts) / ts[:-1]
        returns = returns[~np.isnan(returns)]
        if len(returns) < 20:
            return 0.5

        lags = range(2, min(20, len(returns) // 2))
        tau = []
        for lag in lags:
            std = np.std(returns[lag:] - returns[:-lag])
            if std > 0:
                tau.append(std)
            else:
                tau.append(1e-10)

        if len(tau) < 2:
            return 0.5

        log_lags = np.log(list(lags)[:len(tau)])
        log_tau = np.log(tau)

        coeffs = np.polyfit(log_lags, log_tau, 1)
        return float(np.clip(coeffs[0], 0, 1))

    def _analyze_structure(self, df: pd.DataFrame) -> str:
        """Analyze higher-high/higher-low structure."""
        if len(df) < 20:
            return "unknown"

        # Find swing highs and lows (simple method)
        highs = df["high"].rolling(5, center=True).max()
        lows = df["low"].rolling(5, center=True).min()

        recent_highs = highs.iloc[-20:]
        recent_lows = lows.iloc[-20:]

        # Check if making higher highs + higher lows (uptrend structure)
        hh = recent_highs.iloc[-5:].mean() > recent_highs.iloc[-15:-10].mean()
        hl = recent_lows.iloc[-5:].mean() > recent_lows.iloc[-15:-10].mean()

        if hh and hl:
            return "higher_highs_higher_lows"
        elif not hh and not hl:
            return "lower_highs_lower_lows"
        else:
            return "mixed"

    def get_regime_adjustments(self, regime: MarketRegime) -> dict:
        """Return trading adjustments based on regime."""
        adjustments = {
            MarketRegime.TRENDING_UP: {
                "bias": "long",
                "position_multiplier": 1.3,
                "stop_loss_multiplier": 1.5,  # Wider stops in trends
                "take_profit_multiplier": 2.0,  # Let winners run
            },
            MarketRegime.TRENDING_DOWN: {
                "bias": "short",
                "position_multiplier": 1.0,
                "stop_loss_multiplier": 1.2,
                "take_profit_multiplier": 1.5,
            },
            MarketRegime.RANGING: {
                "bias": "neutral",
                "position_multiplier": 0.8,
                "stop_loss_multiplier": 0.8,  # Tighter stops
                "take_profit_multiplier": 0.8,  # Take profits quickly
            },
            MarketRegime.VOLATILE: {
                "bias": "neutral",
                "position_multiplier": 0.5,  # Reduce size
                "stop_loss_multiplier": 2.0,  # Much wider stops
                "take_profit_multiplier": 1.5,
            },
            MarketRegime.QUIET: {
                "bias": "neutral",
                "position_multiplier": 0.6,
                "stop_loss_multiplier": 0.7,
                "take_profit_multiplier": 0.7,
            },
        }
        return adjustments.get(regime, adjustments[MarketRegime.RANGING])
