"""
Unit tests for BacktestEngine

Tests trade execution, transaction costs, and return calculations.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.backtest_engine import BacktestEngine


class TestBacktestEngine(unittest.TestCase):
    """Test cases for BacktestEngine class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample signals
        dates = pd.date_range('2024-01-01 09:30:00', '2024-01-01 16:00:00', freq='1S')
        positions = np.zeros(len(dates))
        positions[100:200] = 1  # Long position
        positions[300:400] = -1  # Short position

        self.signals = pd.DataFrame({
            'position': positions
        }, index=dates)

        # Create sample price series (random walk)
        np.random.seed(42)
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.01)
        self.prices = pd.Series(prices, index=dates)

        # Initialize engine
        self.engine = BacktestEngine(slippage_bps=5.0)

    def test_basic_execution(self):
        """Test basic trade execution"""
        trades = self.engine.run(
            signals=self.signals,
            price_series=self.prices,
            holding_period=60
        )

        self.assertGreater(len(trades), 0, "Should execute trades")
        self.assertTrue(all(col in trades.columns for col in [
            'entry_time', 'exit_time', 'position', 'entry_price',
            'exit_price', 'raw_return', 'net_return'
        ]), "Trades should have required columns")

    def test_transaction_costs(self):
        """Test that transaction costs are applied"""
        trades = self.engine.run(self.signals, self.prices, holding_period=60)

        for _, trade in trades.iterrows():
            # Net return should be less than raw return due to costs
            self.assertLess(
                trade['net_return'],
                trade['raw_return'] + 0.001,  # Small tolerance
                "Net return should account for transaction costs"
            )

    def test_long_position(self):
        """Test long position returns"""
        # Create simple long signal
        signals = pd.DataFrame({'position': [1]}, index=[self.prices.index[0]])
        trades = self.engine.run(signals, self.prices, holding_period=60)

        self.assertEqual(len(trades), 1, "Should execute one trade")
        trade = trades.iloc[0]
        self.assertEqual(trade['position'], 1, "Should be long")

        # Return sign should match price change
        price_change = trade['exit_price'] - trade['entry_price']
        if price_change > 0:
            self.assertGreater(trade['raw_return'], 0, "Long profits on price increase")

    def test_short_position(self):
        """Test short position returns"""
        # Create simple short signal
        signals = pd.DataFrame({'position': [-1]}, index=[self.prices.index[0]])
        trades = self.engine.run(signals, self.prices, holding_period=60)

        self.assertEqual(len(trades), 1, "Should execute one trade")
        trade = trades.iloc[0]
        self.assertEqual(trade['position'], -1, "Should be short")

    def test_no_signals(self):
        """Test with no trading signals"""
        signals = pd.DataFrame({'position': np.zeros(100)}, index=self.prices.index[:100])
        trades = self.engine.run(signals, self.prices, holding_period=60)

        self.assertEqual(len(trades), 0, "Should not execute trades with no signals")

    def test_monte_carlo(self):
        """Test Monte Carlo simulation"""
        trades = self.engine.run(self.signals, self.prices, holding_period=60)

        if len(trades) > 0:
            returns = trades['net_return'].values
            mc_result = self.engine.monte_carlo_simulation(
                trade_returns=returns,
                num_paths=1000,
                seed=42
            )

            self.assertIn('median_final_return', mc_result)
            self.assertIn('pct_5_return', mc_result)
            self.assertIn('pct_95_return', mc_result)


if __name__ == '__main__':
    unittest.main()
