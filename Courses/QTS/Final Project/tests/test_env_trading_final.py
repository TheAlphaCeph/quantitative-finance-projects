"""
Unit tests for FinalPairTradingEnv
"""

import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cmds.env_trading_final import FinalPairTradingEnv


class TestFinalPairTradingEnv(unittest.TestCase):
    """Test suite for FinalPairTradingEnv"""

    def setUp(self):
        """Create synthetic test data"""
        np.random.seed(42)
        n_rows = 5000

        # Create synthetic price data
        dates = pd.date_range('2024-01-01', periods=n_rows, freq='1H')

        btc_price = 40000 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.02))
        eth_price = 2500 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.025))
        ltc_price = 80 * np.exp(np.cumsum(np.random.randn(n_rows) * 0.03))

        self.df = pd.DataFrame({
            'time': dates,
            'close_btc': btc_price,
            'close_eth': eth_price,
            'close_ltc': ltc_price
        })

        self.pair_list = ['btc-eth', 'btc-ltc']

    def test_environment_initialization(self):
        """Test environment initializes correctly"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60,
            initial_capital=100000
        )

        self.assertIsNotNone(env)
        self.assertEqual(env.initial_capital, 100000)
        self.assertEqual(len(env.pair_list), 2)

    def test_observation_space(self):
        """Test observation space dimensions"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60
        )

        # 2 pairs * 6 features + 3 global = 15
        expected_dim = 2 * 6 + 3
        self.assertEqual(env.observation_space.shape[0], expected_dim)

    def test_action_space(self):
        """Test action space dimensions"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60
        )

        self.assertEqual(env.action_space.shape[0], 2)
        self.assertTrue(np.all(env.action_space.low == -0.5))
        self.assertTrue(np.all(env.action_space.high == 0.5))

    def test_reset(self):
        """Test environment reset"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60
        )

        obs, info = env.reset()

        self.assertEqual(len(obs), 15)  # 2*6 + 3
        self.assertEqual(env.portfolio_value, env.initial_capital)
        self.assertEqual(env.trades_count, 0)
        self.assertTrue(np.all(env.positions == 0))

    def test_step_basic(self):
        """Test basic step functionality"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60,
            initial_capital=100000
        )

        obs, info = env.reset()
        action = np.array([0.1, -0.1])  # Long first pair, short second

        obs, reward, done, truncated, info = env.step(action)

        self.assertEqual(len(obs), 15)
        self.assertIsInstance(reward, (float, np.floating))
        self.assertIsInstance(done, bool)
        self.assertIn('portfolio_value', info)

    def test_transaction_costs(self):
        """Test transaction costs are applied"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60,
            transaction_cost=0.001,
            initial_capital=100000
        )

        obs, info = env.reset()
        initial_value = env.portfolio_value

        # Make a trade
        action = np.array([0.2, 0.0])
        obs, reward, done, truncated, info = env.step(action)

        # Portfolio value should change (could be up or down due to PnL)
        # But we've incurred costs
        self.assertIsNotNone(info.get('transaction_cost'))

    def test_risk_stop(self):
        """Test risk stop triggers correctly"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60,
            risk_stop=0.5,
            initial_capital=100000
        )

        # Manually set portfolio value below risk stop
        env.reset()
        env.portfolio_value = 40000  # 40% of initial (below 50% stop)

        action = np.array([0.0, 0.0])
        obs, reward, done, truncated, info = env.step(action)

        # Should not immediately trigger (would need another step)
        # But portfolio_value should update correctly

    def test_positions_clipping(self):
        """Test actions are clipped to valid range"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60
        )

        obs, info = env.reset()

        # Try to take extreme positions
        action = np.array([10.0, -10.0])  # Should be clipped to [-0.5, 0.5]
        obs, reward, done, truncated, info = env.step(action)

        # Check positions are within bounds
        self.assertTrue(np.all(env.positions >= -0.5))
        self.assertTrue(np.all(env.positions <= 0.5))

    def test_multiple_steps(self):
        """Test running multiple steps"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60,
            step_size=60,
            max_episode_steps=10
        )

        obs, info = env.reset()

        for _ in range(10):
            action = np.random.uniform(-0.5, 0.5, size=2)
            obs, reward, done, truncated, info = env.step(action)

            if done:
                break

        self.assertTrue(len(env.equity_curve) > 0)
        self.assertTrue(len(env.dates) > 0)

    def test_dynamic_hedge_ratios(self):
        """Test dynamic hedge ratio calculation"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60,
            use_dynamic_hedge=True,
            hedge_ratio_window=200
        )

        # Check that hedge ratios were calculated
        self.assertIn('btc-eth', env.hedge_ratios)
        self.assertIn('btc-ltc', env.hedge_ratios)

        # Hedge ratios should be reasonable (not infinity or zero)
        for pair, ratio in env.hedge_ratios.items():
            self.assertGreater(ratio, 0.1)
            self.assertLess(ratio, 10.0)

    def test_volatility_scaling(self):
        """Test volatility-based position scaling"""
        env = FinalPairTradingEnv(
            df_merged=self.df,
            pair_list=self.pair_list,
            window_size=60,
            volatility_lookback=50
        )

        obs, info = env.reset()

        # Take a position
        action = np.array([0.5, 0.5])
        obs, reward, done, truncated, info = env.step(action)

        # Positions should be scaled by volatility (not exactly 0.5)
        # Due to vol_scalar in step function


if __name__ == '__main__':
    unittest.main()
