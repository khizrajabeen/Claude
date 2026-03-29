"""Reinforcement Learning agent for dynamic position sizing and leverage.

Uses PPO (Proximal Policy Optimization) to learn optimal position sizes
based on portfolio state, market conditions, and model confidence.

The RL agent decides:
  1. Position size (% of portfolio)
  2. Leverage multiplier
  3. Whether to scale in/out of positions

Reward function: risk-adjusted returns (Sharpe or Sortino ratio).
"""

import logging
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

logger = logging.getLogger("trading_bot")

MODEL_DIR = Path("models")


class TradingEnv(gym.Env):
    """Custom Gym environment for RL-based position sizing.

    Observation space:
        - model_confidence: ensemble prediction confidence [0, 1]
        - predicted_return: predicted return magnitude
        - portfolio_value: normalized portfolio value
        - current_position: current position size [-1, 1]
        - unrealized_pnl: current unrealized PnL
        - volatility: recent realized volatility
        - drawdown: current drawdown from peak
        - win_rate: rolling win rate of recent trades
        - market_regime: trend/range/volatile indicator
        - consecutive_losses: number of consecutive losing trades

    Action space (continuous):
        - position_size: target position as fraction of portfolio [-1, 1]
                         negative = short, positive = long, 0 = flat
    """

    metadata = {"render_modes": []}

    def __init__(self, price_data: np.ndarray, features: np.ndarray,
                 initial_balance: float = 10000.0,
                 max_leverage: int = 10,
                 maker_fee: float = 0.0002,
                 taker_fee: float = 0.0004):
        super().__init__()

        self.price_data = price_data
        self.features = features
        self.initial_balance = initial_balance
        self.max_leverage = max_leverage
        self.maker_fee = maker_fee
        self.taker_fee = taker_fee

        # Observation: 10 features
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
        )

        # Action: position size [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_idx = 60  # Start after warmup
        self.balance = self.initial_balance
        self.peak_balance = self.initial_balance
        self.position = 0.0  # Current position in base currency
        self.entry_price = 0.0
        self.trades = []
        self.consecutive_losses = 0
        self.returns_history = []
        return self._get_obs(), {}

    def step(self, action):
        target_position_frac = float(np.clip(action[0], -1, 1))
        current_price = self.price_data[self.step_idx]
        prev_price = self.price_data[self.step_idx - 1]

        # Calculate unrealized PnL from existing position
        if self.position != 0:
            price_change = (current_price - prev_price) / prev_price
            unrealized = self.position * self.balance * price_change
            self.balance += unrealized

        # Rebalance to target position
        position_change = target_position_frac - self.position
        if abs(position_change) > 0.01:  # Min trade threshold
            trade_cost = abs(position_change) * self.balance * self.taker_fee
            self.balance -= trade_cost

            if self.position != 0 and target_position_frac == 0:
                # Closing — record trade
                pnl = self.balance - self.initial_balance
                self.trades.append(pnl)
                if pnl < 0:
                    self.consecutive_losses += 1
                else:
                    self.consecutive_losses = 0

            self.position = target_position_frac
            if target_position_frac != 0:
                self.entry_price = current_price

        # Update peak
        self.peak_balance = max(self.peak_balance, self.balance)

        # Calculate reward (risk-adjusted return)
        step_return = (self.balance - self.initial_balance) / self.initial_balance
        self.returns_history.append(step_return)

        reward = self._calculate_reward(step_return)

        self.step_idx += 1
        terminated = self.step_idx >= len(self.price_data) - 1
        truncated = self.balance <= self.initial_balance * 0.5  # Liquidation at -50%

        if truncated:
            reward -= 10.0  # Heavy penalty for ruin

        return self._get_obs(), reward, terminated, truncated, {}

    def _calculate_reward(self, step_return: float) -> float:
        """Reward = step return penalized by risk."""
        reward = step_return * 100  # Scale up

        # Drawdown penalty
        drawdown = (self.peak_balance - self.balance) / self.peak_balance
        reward -= drawdown * 5

        # Consecutive loss penalty
        if self.consecutive_losses > 3:
            reward -= self.consecutive_losses * 0.5

        return float(reward)

    def _get_obs(self) -> np.ndarray:
        idx = self.step_idx
        price = self.price_data[idx]

        # Volatility (20-period realized)
        if idx >= 20:
            returns = np.diff(self.price_data[idx - 20 : idx + 1]) / self.price_data[idx - 20 : idx]
            volatility = np.std(returns)
        else:
            volatility = 0.0

        # Drawdown
        drawdown = (self.peak_balance - self.balance) / self.peak_balance if self.peak_balance > 0 else 0

        # Win rate
        if len(self.trades) >= 5:
            recent = self.trades[-20:]
            win_rate = sum(1 for t in recent if t > 0) / len(recent)
        else:
            win_rate = 0.5

        # Use features if available
        feat_row = self.features[idx] if idx < len(self.features) else np.zeros(3)

        obs = np.array([
            feat_row[0] if len(feat_row) > 0 else 0,  # model confidence proxy
            feat_row[1] if len(feat_row) > 1 else 0,  # predicted return proxy
            self.balance / self.initial_balance,         # normalized portfolio
            self.position,                               # current position
            (price - self.entry_price) / price if self.entry_price > 0 else 0,  # unrealized PnL
            volatility * 100,                            # volatility
            drawdown,                                    # drawdown
            win_rate,                                    # win rate
            feat_row[2] if len(feat_row) > 2 else 0,   # market regime proxy
            float(self.consecutive_losses) / 10,         # consecutive losses (normalized)
        ], dtype=np.float32)

        return obs


class RLPositionSizer:
    """RL-based position sizing using PPO."""

    def __init__(self, config: dict):
        self.config = config
        self.rl_config = config.get("rl_agent", {})
        self.model = None
        self.is_trained = False

    def train(self, price_data: np.ndarray, features: np.ndarray):
        """Train the PPO agent on historical data."""
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        logger.info("Training RL position sizer (PPO)...")

        env = DummyVecEnv([lambda: TradingEnv(
            price_data, features,
            initial_balance=self.config.get("paper", {}).get("initial_balance", 10000),
        )])

        self.model = PPO(
            "MlpPolicy", env,
            learning_rate=self.rl_config.get("learning_rate", 0.0003),
            gamma=self.rl_config.get("gamma", 0.99),
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            ent_coef=0.01,
            verbose=1,
        )

        total_steps = self.rl_config.get("total_timesteps", 500000)
        self.model.learn(total_timesteps=total_steps)
        self.is_trained = True

        # Save
        self.model.save(str(MODEL_DIR / "rl_position_sizer"))
        logger.info("RL agent trained and saved")

    def get_position_size(self, obs: np.ndarray) -> float:
        """Get the optimal position size for current state.

        Returns:
            float: position fraction [-1, 1]. Magnitude = size, sign = direction.
        """
        if not self.is_trained or self.model is None:
            return 0.0

        action, _ = self.model.predict(obs, deterministic=True)
        return float(np.clip(action[0], -1, 1))

    def load(self):
        """Load trained model."""
        from stable_baselines3 import PPO

        path = MODEL_DIR / "rl_position_sizer.zip"
        if path.exists():
            self.model = PPO.load(str(MODEL_DIR / "rl_position_sizer"))
            self.is_trained = True
            logger.info("RL agent loaded")
        else:
            logger.warning("No trained RL agent found")


class KellyCriterionSizer:
    """Kelly Criterion for position sizing — used as fallback or combined with RL.

    Kelly fraction = (win_rate * avg_win - (1-win_rate) * avg_loss) / avg_win
    We use fractional Kelly (typically 25-50%) for safety.
    """

    def __init__(self, kelly_fraction: float = 0.25):
        self.kelly_fraction = kelly_fraction
        self.trades: list[float] = []

    def update(self, pnl: float):
        """Record a trade result."""
        self.trades.append(pnl)

    def get_kelly_size(self) -> float:
        """Calculate fractional Kelly position size."""
        if len(self.trades) < 10:
            return 0.02  # Default 2% until enough history

        wins = [t for t in self.trades if t > 0]
        losses = [t for t in self.trades if t < 0]

        if not wins or not losses:
            return 0.02

        win_rate = len(wins) / len(self.trades)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))

        if avg_win == 0:
            return 0.02

        kelly = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_win
        kelly = max(0, kelly) * self.kelly_fraction  # Fractional Kelly

        # Cap at 20%
        return min(kelly, 0.20)
