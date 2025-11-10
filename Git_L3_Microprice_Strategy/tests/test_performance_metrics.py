"""
Unit tests for PerformanceMetrics

Tests computation of performance statistics and risk metrics.
"""

import unittest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.backtesting.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample trades
        np.random.seed(42)
        n_trades = 100

        dates = pd.date_range('2024-01-01', periods=n_trades, freq='1H')
        returns = np.random.randn(n_trades) * 0.0001 + 0.00005  # Slightly positive expected return

        self.trades_df = pd.DataFrame({
            'entry_time': dates,
            'exit_time': dates + pd.Timedelta(minutes=1),
            'position': np.random.choice([1, -1], n_trades),
            'entry_price': 100.0,
            'exit_price': 100.0,
            'raw_return': returns,
            'net_return': returns - 0.0001  # Transaction costs
        })

    def test_basic_metrics(self):
        """Test basic metric computation"""
        metrics = PerformanceMetrics.compute(self.trades_df)

        # Check all expected keys are present
        expected_keys = [
            'total_trades', 'win_rate', 'avg_return_bps',
            'total_return', 'buyhold_return', 'excess_return',
            'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
            'calmar_ratio', 'profit_factor'
        ]

        for key in expected_keys:
            self.assertIn(key, metrics, f"Metrics should include {key}")

    def test_empty_trades(self):
        """Test with empty trades dataframe"""
        empty_df = pd.DataFrame()
        metrics = PerformanceMetrics.compute(empty_df)

        self.assertEqual(metrics, {}, "Empty trades should return empty dict")

    def test_win_rate(self):
        """Test win rate calculation"""
        # Create predictable trades
        trades = self.trades_df.copy()
        trades['net_return'] = [0.001] * 60 + [-0.001] * 40  # 60% wins

        metrics = PerformanceMetrics.compute(trades)
        self.assertAlmostEqual(metrics['win_rate'], 0.6, places=2)

    def test_sharpe_ratio(self):
        """Test Sharpe ratio calculation"""
        # Create high Sharpe scenario (consistent positive returns)
        trades = self.trades_df.copy()
        trades['net_return'] = np.random.randn(len(trades)) * 0.0001 + 0.001  # High mean, low std

        metrics = PerformanceMetrics.compute(trades)
        self.assertGreater(metrics['sharpe_ratio'], 0, "Positive returns should yield positive Sharpe")

    def test_max_drawdown(self):
        """Test maximum drawdown calculation"""
        # Create scenario with known drawdown
        trades = self.trades_df.copy()
        # Sequence: +10%, -15%, +5% (max DD should be -15%)
        trades['net_return'] = [0.1, -0.15, 0.05] + [0.0] * (len(trades) - 3)

        metrics = PerformanceMetrics.compute(trades)
        self.assertLess(metrics['max_drawdown'], 0, "Drawdown should be negative")

    def test_profit_factor(self):
        """Test profit factor calculation"""
        trades = self.trades_df.copy()
        # 2:1 profit factor (wins = 2x losses)
        trades['net_return'] = [0.002] * 50 + [-0.001] * 50

        metrics = PerformanceMetrics.compute(trades)
        self.assertGreater(metrics['profit_factor'], 1.5, "Should have decent profit factor")


if __name__ == '__main__':
    unittest.main()
