"""Dynamic position sizer — scales with portfolio growth.

Combines:
  1. Portfolio-tier scaling (small portfolio = conservative, big = aggressive)
  2. Kelly Criterion (mathematically optimal sizing)
  3. RL agent suggestion (learned from historical data)
  4. Market regime adjustment (reduce in volatile, increase in trending)
  5. Model confidence weighting

This is what separates pros from amateurs: position sizing is more
important than entry signals.
"""

import logging

import numpy as np

from bot.analysis.regime import MarketRegime

logger = logging.getLogger("trading_bot")


class DynamicPositionSizer:
    """Compute position size and leverage based on all available signals."""

    def __init__(self, config: dict):
        self.config = config
        self.scaling_config = config.get("scaling", {})
        self.trading_config = config.get("trading", {})
        self.risk_config = config.get("risk", {})
        self.tiers = self.scaling_config.get("tiers", [])

    def compute(
        self,
        portfolio_value: float,
        model_confidence: float,
        predicted_return: float,
        regime: MarketRegime,
        regime_confidence: float,
        kelly_fraction: float,
        rl_suggestion: float | None = None,
        current_drawdown_pct: float = 0.0,
        consecutive_losses: int = 0,
    ) -> dict:
        """Compute final position size and leverage.

        Returns:
            dict with:
                position_pct: % of portfolio to allocate
                leverage: leverage multiplier
                direction: "long" or "short"
                raw_components: breakdown of each input
        """
        # 1. Get tier limits based on portfolio size
        tier = self._get_tier(portfolio_value)
        max_pos_pct = tier["max_position_pct"]
        max_lev = tier["max_leverage"]
        risk_per_trade = tier["risk_per_trade_pct"]

        # 2. Base size from Kelly Criterion
        base_pct = kelly_fraction * 100  # Convert to percentage
        base_pct = min(base_pct, max_pos_pct)  # Cap at tier limit

        # 3. Scale by model confidence (0.65–1.0 maps to 0.5–1.5x)
        min_conf = self.config.get("model", {}).get("min_confidence", 0.65)
        if model_confidence < min_conf:
            # Below minimum confidence — no trade
            return self._no_trade("confidence below threshold")

        conf_multiplier = 0.5 + (model_confidence - min_conf) / (1 - min_conf)
        conf_multiplier = np.clip(conf_multiplier, 0.5, 1.5)

        # 4. Regime adjustment
        from bot.analysis.regime import RegimeDetector
        regime_adj = RegimeDetector(self.config).get_regime_adjustments(regime)
        regime_multiplier = regime_adj["position_multiplier"]

        # 5. RL agent suggestion (if available)
        rl_multiplier = 1.0
        if rl_suggestion is not None:
            # RL gives [-1, 1], use magnitude as multiplier
            rl_multiplier = 0.5 + abs(rl_suggestion) * 0.5  # 0.5 to 1.0

        # 6. Drawdown reduction — reduce size when in drawdown
        dd_multiplier = 1.0
        max_dd = self.risk_config.get("max_total_drawdown_pct", 15.0)
        if current_drawdown_pct > 0:
            # Linear reduction: at max_dd/2, cut size in half
            dd_multiplier = max(0.25, 1 - (current_drawdown_pct / max_dd))

        # 7. Consecutive loss penalty
        loss_multiplier = 1.0
        if consecutive_losses >= 3:
            loss_multiplier = max(0.3, 1 - consecutive_losses * 0.15)

        # Combine everything
        final_pct = (
            base_pct
            * conf_multiplier
            * regime_multiplier
            * rl_multiplier
            * dd_multiplier
            * loss_multiplier
        )

        # Hard caps
        final_pct = np.clip(final_pct, 0.5, max_pos_pct)

        # Leverage calculation
        leverage = self._calc_leverage(
            portfolio_value, model_confidence, predicted_return,
            regime, max_lev, current_drawdown_pct,
        )

        # Direction
        if rl_suggestion is not None and rl_suggestion < 0:
            direction = "short"
        elif regime_adj["bias"] == "short" and predicted_return < 0:
            direction = "short"
        else:
            direction = "long"

        result = {
            "position_pct": round(float(final_pct), 2),
            "leverage": round(float(leverage), 1),
            "direction": direction,
            "notional_usd": round(portfolio_value * (final_pct / 100) * leverage, 2),
            "risk_usd": round(portfolio_value * (risk_per_trade / 100), 2),
            "raw_components": {
                "tier": tier,
                "kelly_base_pct": round(base_pct, 2),
                "conf_multiplier": round(conf_multiplier, 2),
                "regime_multiplier": round(regime_multiplier, 2),
                "rl_multiplier": round(rl_multiplier, 2),
                "dd_multiplier": round(dd_multiplier, 2),
                "loss_multiplier": round(loss_multiplier, 2),
            },
        }

        logger.info(
            f"Position: {final_pct:.1f}% @ {leverage:.1f}x "
            f"(${result['notional_usd']:.0f}) | {direction} | "
            f"conf={model_confidence:.2f} regime={regime.value}"
        )

        return result

    def _get_tier(self, portfolio_value: float) -> dict:
        """Get the scaling tier for current portfolio value."""
        for tier in self.tiers:
            if portfolio_value <= tier["max_portfolio"]:
                return tier

        # If above all tiers, use the last (most aggressive)
        if self.tiers:
            return self.tiers[-1]

        # Default conservative tier
        return {
            "max_portfolio": 1_000_000,
            "max_leverage": 3,
            "max_position_pct": 5.0,
            "risk_per_trade_pct": 1.0,
        }

    def _calc_leverage(
        self,
        portfolio_value: float,
        confidence: float,
        predicted_return: float,
        regime: MarketRegime,
        max_leverage: float,
        drawdown_pct: float,
    ) -> float:
        """Calculate leverage based on signals and limits."""
        # Start at 1x
        leverage = 1.0

        # Scale up with confidence
        if confidence > 0.8:
            leverage += (confidence - 0.8) * 10  # +0 to +2x for 0.8–1.0

        # Scale up with portfolio (bigger portfolio = can handle more leverage)
        if portfolio_value > 10000:
            leverage += min(2.0, np.log10(portfolio_value / 10000))

        # Scale up with predicted return magnitude
        if abs(predicted_return) > 0.005:  # >0.5% predicted move
            leverage += min(2.0, abs(predicted_return) * 100)

        # Reduce in volatile regime
        if regime == MarketRegime.VOLATILE:
            leverage *= 0.5
        elif regime == MarketRegime.QUIET:
            leverage *= 0.7

        # Reduce in drawdown
        if drawdown_pct > 5:
            leverage *= 0.5

        # Hard cap
        hard_max = self.trading_config.get("max_leverage", 10)
        leverage = np.clip(leverage, 1.0, min(max_leverage, hard_max))

        return float(leverage)

    def _no_trade(self, reason: str) -> dict:
        logger.info(f"No trade: {reason}")
        return {
            "position_pct": 0.0,
            "leverage": 0.0,
            "direction": "none",
            "notional_usd": 0.0,
            "risk_usd": 0.0,
            "raw_components": {"skip_reason": reason},
        }
