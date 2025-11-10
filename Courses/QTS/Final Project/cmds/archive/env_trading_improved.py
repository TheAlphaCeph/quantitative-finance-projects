"""
Improved Pair Trading Environment for Reinforcement Learning
==============================================================

Key Improvements:
1. Reward function that ENCOURAGES no-trade decisions when unprofitable
2. Transaction costs included in observation space (cost awareness)
3. Position-holding bonus to reduce excessive trading
4. Better spread features and volatility indicators
5. Action space redesigned to reduce trading frequency

Author: Abhay Kanwar
Date: 2025-11-03
"""

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces


class ImprovedPairTradingEnv(gym.Env):
    """
    Advanced Pair Trading Environment with transaction cost awareness
    and improved reward shaping to prevent overtrading.
    """

    def __init__(
        self,
        df_merged: pd.DataFrame,
        pair_list: list,
        window_size: int = 60,
        step_size: int = 60,
        initial_capital: float = 1e5,
        max_leverage: float = 1.0,
        transaction_cost: float = 0.001,  # 0.1% per side = 0.2% round-trip
        funding_spread: float = 0.0,
        reward_scaling: float = 1e-4,
        max_episode_steps: int = 5000,
        risk_stop: float = 0.3,
        # NEW: Parameters to control trading frequency
        holding_reward: float = 0.1,  # Bonus for maintaining positions
        trade_penalty: float = 0.5,   # Penalty for changing positions
        min_trade_threshold: float = 0.05,  # Minimum position change to count as trade
    ):
        super().__init__()
        self.df = df_merged.copy()
        self.pair_list = pair_list
        self.window_size = window_size
        self.step_size = step_size
        self.initial_capital = initial_capital
        self.max_leverage = max_leverage
        self.transaction_cost = transaction_cost
        self.funding_spread = funding_spread
        self.reward_scaling = reward_scaling
        self.max_episode_steps = max_episode_steps
        self.risk_stop = risk_stop

        # NEW: Trading frequency controls
        self.holding_reward = holding_reward
        self.trade_penalty = trade_penalty
        self.min_trade_threshold = min_trade_threshold

        # 1) Build 'spread' columns => log(p_base) - log(p_quote)
        self.spread_cols = []
        for pair in self.pair_list:
            base, quote = pair.split('-')
            col_base = f"close_{base}"
            col_quote = f"close_{quote}"
            spread_col = f"spread_{base}_{quote}"
            self.df[spread_col] = np.log(self.df[col_base]) - np.log(self.df[col_quote])
            self.spread_cols.append(spread_col)

        # 2) Rolling mean & std => zscore
        self.zscore_cols = []
        self.momentum_cols = []
        self.volatility_cols = []

        for spread_col in self.spread_cols:
            roll_mean = self.df[spread_col].rolling(self.window_size).mean()
            roll_std = self.df[spread_col].rolling(self.window_size).std()
            z_col = spread_col.replace('spread', 'zscore')
            self.df[z_col] = (self.df[spread_col] - roll_mean) / (roll_std + 1e-8)
            self.zscore_cols.append(z_col)

            # NEW: Add spread momentum and volatility features
            momentum_col = spread_col.replace('spread', 'momentum')
            self.df[momentum_col] = self.df[spread_col].diff(5)
            self.momentum_cols.append(momentum_col)

            volatility_col = spread_col.replace('spread', 'volatility')
            self.df[volatility_col] = self.df[spread_col].rolling(20).std()
            self.volatility_cols.append(volatility_col)

        # 3) Drop NaN only in the columns we created (not all columns in df)
        cols_to_check = self.spread_cols + self.zscore_cols + self.momentum_cols + self.volatility_cols
        self.df.dropna(subset=cols_to_check, inplace=True)
        self.df.reset_index(drop=True, inplace=True)
        if len(self.df) < self.window_size + 1:
            raise ValueError(
                f"Not enough data after rolling. Need > {self.window_size}, got {len(self.df)}."
            )

        # 4) Adjust max_episode_steps
        max_possible = (len(self.df) - self.window_size - 1) // self.step_size
        if self.max_episode_steps is not None:
            self.max_episode_steps = min(self.max_episode_steps, max_possible)

        # 5) IMPROVED Observation space:
        # For each pair: zscore + momentum + volatility + position + time_in_position
        # Plus global: portfolio_value_ratio + recent_trade_count + unrealized_pnl
        self.num_pairs = len(self.pair_list)
        features_per_pair = 5  # zscore, momentum, volatility, position, time_in_position
        global_features = 3    # pv_ratio, recent_trades, unrealized_pnl
        obs_dim = self.num_pairs * features_per_pair + global_features

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # 6) Action space: continuous in [-0.5, 0.5] but we'll add action smoothing
        self.action_space = spaces.Box(
            low=-0.5, high=0.5, shape=(self.num_pairs,), dtype=np.float32
        )

        # Internals
        self.current_idx = 0
        self.step_counter = 0
        self.done_flag = False

        self.positions = np.zeros(self.num_pairs, dtype=np.float32)
        self.last_positions = np.zeros_like(self.positions)
        self.time_in_position = np.zeros(self.num_pairs, dtype=np.int32)  # NEW
        self.portfolio_value = self.initial_capital
        self.trades_count = 0
        self.time_in_market_steps = 0
        self.recent_trade_count = 0  # NEW: trades in last N steps

        # Logs
        self.equity_curve = []
        self.dates = []
        self.equity_curve_per_pair = {pair: [] for pair in self.pair_list}
        self.dates_per_pair = {pair: [] for pair in self.pair_list}
        self.trade_history = []

        # NEW: Track unrealized PnL
        self.entry_spreads = np.zeros(self.num_pairs, dtype=np.float32)
        self.unrealized_pnl = 0.0

    def _get_current_row(self):
        if self.current_idx >= len(self.df):
            return self.df.iloc[-1]
        return self.df.iloc[self.current_idx]

    def _get_obs(self):
        """IMPROVED observation with more features"""
        row = self._get_current_row()

        obs = []

        # Per-pair features
        for i, pair in enumerate(self.pair_list):
            base, quote = pair.split('-')

            # Z-score
            z_col = f"zscore_{base}_{quote}"
            obs.append(row[z_col])

            # Momentum
            momentum_col = f"momentum_{base}_{quote}"
            obs.append(row[momentum_col])

            # Volatility
            volatility_col = f"volatility_{base}_{quote}"
            obs.append(row[volatility_col])

            # Current position
            obs.append(self.positions[i])

            # Time in position (normalized)
            obs.append(self.time_in_position[i] / 100.0)

        # Global features
        obs.append(self.portfolio_value / self.initial_capital)  # PV ratio
        obs.append(self.recent_trade_count / 10.0)  # Recent trade frequency
        obs.append(self.unrealized_pnl / self.initial_capital)  # Unrealized PnL

        return np.array(obs, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.done_flag = False
        self.current_idx = self.window_size
        if self.current_idx >= len(self.df):
            self.current_idx = len(self.df) - 1

        self.step_counter = 0
        self.positions = np.zeros(self.num_pairs, dtype=np.float32)
        self.last_positions = np.zeros_like(self.positions)
        self.time_in_position = np.zeros(self.num_pairs, dtype=np.int32)
        self.portfolio_value = self.initial_capital
        self.trades_count = 0
        self.time_in_market_steps = 0
        self.recent_trade_count = 0

        self.equity_curve = []
        self.dates = []
        self.equity_curve_per_pair = {pair: [] for pair in self.pair_list}
        self.dates_per_pair = {pair: [] for pair in self.pair_list}
        self.trade_history = []

        self.entry_spreads = np.zeros(self.num_pairs, dtype=np.float32)
        self.unrealized_pnl = 0.0

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """IMPROVED step function with better reward shaping"""
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Store old positions for reward calculation
        old_positions = self.positions.copy()

        # 1) Calculate PnL from existing positions
        step_pnl, pairwise_pnls = self._compute_pnl()
        realized_pnl = step_pnl

        # 2) Calculate position changes and transaction costs
        position_changes = np.abs(action - self.positions)
        total_position_change = position_changes.sum()

        # Transaction cost only on actual trades
        transaction_cost = 0.0
        trades_this_step = 0

        for i in range(self.num_pairs):
            if position_changes[i] > self.min_trade_threshold:
                trades_this_step += 1
                transaction_cost += position_changes[i] * self.initial_capital * self.transaction_cost

        step_pnl -= transaction_cost

        # 3) Funding cost (holding cost)
        funding_cost = np.sum(np.abs(self.positions)) * self.initial_capital * self.funding_spread
        step_pnl -= funding_cost

        # 4) Update portfolio value
        self.portfolio_value += step_pnl

        # 5) IMPROVED REWARD CALCULATION
        # Base reward: realized PnL
        reward = realized_pnl

        # Penalty for trading (to reduce frequency)
        if trades_this_step > 0:
            reward -= self.trade_penalty * trades_this_step * self.initial_capital * self.transaction_cost

        # Bonus for holding profitable positions
        if np.any(np.abs(self.positions) > 0.01):
            # If we have positions and made money, give holding bonus
            if realized_pnl > 0:
                reward += self.holding_reward * abs(realized_pnl)

        # Penalty for excessive position changes
        if total_position_change > 0.5:  # Changed more than 50% of capital
            reward -= 0.2 * total_position_change * self.initial_capital

        # Small bonus for staying flat when spreads are near zero
        row_now = self._get_current_row()
        avg_zscore = 0
        for i, pair in enumerate(self.pair_list):
            base, quote = pair.split('-')
            z_col = f"zscore_{base}_{quote}"
            avg_zscore += abs(row_now[z_col])
        avg_zscore /= len(self.pair_list)

        if avg_zscore < 0.5 and np.all(np.abs(action) < 0.05):
            # Spread is small and we're staying flat - good decision!
            reward += 0.1 * self.initial_capital * self.reward_scaling

        # 6) Risk management
        done = False
        truncated = False
        if self.portfolio_value <= (self.risk_stop * self.initial_capital):
            step_pnl = self.risk_stop * self.initial_capital - self.portfolio_value
            self.portfolio_value = self.risk_stop * self.initial_capital
            reward = -10000  # Large penalty for blowing up
            done = True
            truncated = True

        # 7) Update trade counts
        if trades_this_step > 0:
            self.trades_count += trades_this_step
            self.recent_trade_count += 1

        # Decay recent trade count
        if self.step_counter % 10 == 0:
            self.recent_trade_count = max(0, self.recent_trade_count - 1)

        # 8) Update time in position
        for i in range(self.num_pairs):
            if abs(self.positions[i]) > 0.01:
                self.time_in_position[i] += 1
            else:
                self.time_in_position[i] = 0

        # 9) Log equity curve
        self.equity_curve.append(self.portfolio_value)
        self.dates.append(row_now["time"])

        # 10) Pair-level tracking
        if not hasattr(self, 'pair_values'):
            self.pair_values = {p: self.initial_capital / len(self.pair_list) for p in self.pair_list}

        for i, pair in enumerate(self.pair_list):
            self.pair_values[pair] += pairwise_pnls[i]
            if self.pair_values[pair] < 0:
                self.pair_values[pair] = 0
            self.equity_curve_per_pair[pair].append(self.pair_values[pair])
            self.dates_per_pair[pair].append(row_now["time"])

        # 11) Record trade history
        self.trade_history.append({
            "step": self.step_counter,
            "time": row_now["time"],
            "old_pos": old_positions,
            "new_pos": action.copy(),
            "step_pnl": step_pnl,
            "portfolio_value": self.portfolio_value,
            "trades_count": trades_this_step,
            "transaction_cost": transaction_cost,
        })

        # 12) Update positions
        self.last_positions = self.positions.copy()
        self.positions = action

        # 13) Calculate unrealized PnL for observation
        self.unrealized_pnl = self._compute_unrealized_pnl()

        # 14) Scale reward
        scaled_reward = reward * self.reward_scaling

        # 15) Advance time
        self.current_idx += self.step_size
        self.step_counter += 1

        # 16) Check episode termination
        if not done:
            if self.current_idx >= len(self.df) - 1:
                done = True
            elif self.step_counter >= self.max_episode_steps:
                done = True
                truncated = True

        obs = self._get_obs()
        info = {
            "portfolio_value": self.portfolio_value,
            "trades_count": self.trades_count,
            "transaction_cost": transaction_cost,
            "realized_pnl": realized_pnl,
        }
        return obs, scaled_reward, done, truncated, info

    def _compute_pnl(self):
        """Calculate realized PnL from position changes"""
        if self.step_counter == 0:
            return 0.0, [0.0] * self.num_pairs

        row_now = self._get_current_row()
        row_prev_idx = self.current_idx - self.step_size
        if row_prev_idx < 0:
            row_prev_idx = 0
        row_prev = self.df.iloc[row_prev_idx]

        total_pnl = 0.0
        pairwise_pnls = []

        for i, pair in enumerate(self.pair_list):
            base, quote = pair.split('-')
            spread_col = f"spread_{base}_{quote}"
            spread_now = row_now[spread_col]
            spread_prev = row_prev[spread_col]
            spread_diff = spread_now - spread_prev

            pos_frac = self.positions[i]
            notional = self.initial_capital * abs(pos_frac)
            direction = np.sign(pos_frac)
            pair_pnl = notional * direction * spread_diff
            total_pnl += pair_pnl
            pairwise_pnls.append(pair_pnl)

        return total_pnl, pairwise_pnls

    def _compute_unrealized_pnl(self):
        """Calculate unrealized PnL from current positions"""
        if self.step_counter == 0:
            return 0.0

        unrealized = 0.0
        row_now = self._get_current_row()

        for i, pair in enumerate(self.pair_list):
            if abs(self.positions[i]) < 0.01:
                continue

            base, quote = pair.split('-')
            spread_col = f"spread_{base}_{quote}"
            z_col = f"zscore_{base}_{quote}"

            current_zscore = row_now[z_col]

            # Simple unrealized: how far is zscore from mean reversion target (0)
            pos_frac = self.positions[i]
            notional = self.initial_capital * abs(pos_frac)
            direction = np.sign(pos_frac)

            # If long spread (pos > 0), we profit when zscore decreases
            # If short spread (pos < 0), we profit when zscore increases
            unrealized += notional * (-direction * current_zscore) * 0.01  # Scale factor

        return unrealized

    def render(self, mode='human'):
        print(f"Step: {self.step_counter}, "
              f"Index: {self.current_idx}, "
              f"Value: {self.portfolio_value:.2f}, "
              f"Trades: {self.trades_count}, "
              f"Recent Trades: {self.recent_trade_count}")
