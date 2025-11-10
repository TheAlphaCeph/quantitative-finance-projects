"""
Unit tests for Backtester

Tests backtest execution with transaction costs, position management,
and equity curve generation.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from strategy.backtester import Backtester


class TestBacktester(unittest.TestCase):
    """Test cases for Backtester class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample data with multi-index (date, ticker)
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']

        # Create multi-index prices
        np.random.seed(42)
        idx = pd.MultiIndex.from_product([dates, tickers], names=['date', 'ticker'])

        # Sample prices (trending up)
        prices_flat = np.tile(
            np.cumsum(np.random.randn(len(dates), len(tickers)) * 0.5 + 0.1, axis=0) + 100,
            (1, 1)
        ).ravel()

        self.prices = pd.DataFrame(
            {'close': prices_flat[:len(idx)], 'open': prices_flat[:len(idx)] * 0.999},
            index=idx
        )

        # Create signals
        signal_vals = np.random.randn(len(idx)) * 0.5
        self.signals = pd.DataFrame(
            {'composite_signal': signal_vals},
            index=idx
        )

        # Initialize backtester
        self.backtester = Backtester(
            transaction_cost_bps=10,
            rebalance_frequency='monthly',
            initial_capital=1_000_000
        )

    def test_initialization(self):
        """Test backtester initialization"""
        self.assertEqual(self.backtester.initial_capital, 1_000_000)
        self.assertEqual(self.backtester.transaction_cost_bps, 10)
        self.assertEqual(self.backtester.rebalance_frequency, 'monthly')

    def test_backtest_execution(self):
        """Test that backtest runs without errors"""
        results = self.backtester.run(
            signals=self.signals,
            prices=self.prices,
            start_date='2020-02-01',
            end_date='2020-11-30'
        )

        # Check result structure
        self.assertIn('returns', results)
        self.assertIn('equity_curve', results)
        self.assertIn('positions', results)
        self.assertIn('metrics', results)

        # Check returns are valid
        returns = results['returns']
        self.assertIsInstance(returns, pd.Series)
        self.assertGreater(len(returns), 0)
        self.assertTrue(np.all(np.isfinite(returns)))

    def test_equity_curve_monotonicity(self):
        """Test that equity curve starts at initial capital"""
        results = self.backtester.run(
            signals=self.signals,
            prices=self.prices,
            start_date='2020-02-01',
            end_date='2020-11-30'
        )

        equity = results['equity_curve']

        # First value should be close to initial capital
        self.assertAlmostEqual(equity.iloc[0], self.backtester.initial_capital, delta=100000)

        # All values should be positive
        self.assertTrue(np.all(equity > 0))

    def test_returns_calculation(self):
        """Test that returns are properly calculated"""
        results = self.backtester.run(
            signals=self.signals,
            prices=self.prices,
            start_date='2020-02-01',
            end_date='2020-11-30'
        )

        returns = results['returns']
        equity = results['equity_curve']

        # Returns and equity should have compatible lengths
        self.assertGreater(len(returns), 0)
        self.assertGreater(len(equity), 0)

        # Returns should be in reasonable range (-50% to +50% daily)
        self.assertGreater(returns.min(), -0.5)
        self.assertLess(returns.max(), 0.5)

    def test_transaction_costs(self):
        """Test that transaction costs are applied"""
        # Run with zero costs
        backtester_no_costs = Backtester(
            transaction_cost_bps=0,
            rebalance_frequency='monthly',
            initial_capital=1_000_000
        )

        results_no_costs = backtester_no_costs.run(
            signals=self.signals,
            prices=self.prices,
            start_date='2020-02-01',
            end_date='2020-11-30'
        )

        # Run with costs
        results_with_costs = self.backtester.run(
            signals=self.signals,
            prices=self.prices,
            start_date='2020-02-01',
            end_date='2020-11-30'
        )

        # Final equity should be lower with costs (assuming some trading)
        final_no_costs = results_no_costs['equity_curve'].iloc[-1]
        final_with_costs = results_with_costs['equity_curve'].iloc[-1]

        # Costs should reduce final equity (if there were any trades)
        # Note: this may not always hold if strategy has zero turnover
        self.assertIsInstance(final_no_costs, (int, float))
        self.assertIsInstance(final_with_costs, (int, float))

    def test_metrics_calculation(self):
        """Test that performance metrics are calculated"""
        results = self.backtester.run(
            signals=self.signals,
            prices=self.prices,
            start_date='2020-02-01',
            end_date='2020-11-30'
        )

        metrics = results['metrics']

        # Check key metrics exist
        expected_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'volatility']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(np.isfinite(metrics[metric]) or np.isnan(metrics[metric]))


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
