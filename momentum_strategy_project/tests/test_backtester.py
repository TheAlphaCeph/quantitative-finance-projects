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
        # Create sample data
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Sample prices (trending up)
        np.random.seed(42)
        self.prices = pd.DataFrame(
            np.cumsum(np.random.randn(len(dates), len(tickers)) * 0.5 + 0.1, axis=0) + 100,
            index=dates,
            columns=tickers
        )
        
        # Sample positions (monthly rebalancing)
        position_dates = pd.date_range('2020-01-01', '2020-12-31', freq='MS')
        self.positions = pd.DataFrame(
            {
                'AAPL': [0.3, 0.25, 0.35, 0.3, 0.25, 0.3, 0.35, 0.3, 0.25, 0.3, 0.35, 0.3],
                'MSFT': [0.2, 0.25, 0.15, 0.2, 0.25, 0.2, 0.15, 0.2, 0.25, 0.2, 0.15, 0.2],
                'GOOGL': [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]
            },
            index=position_dates
        )
        
        # Initialize backtester
        self.backtester = Backtester(
            initial_capital=1_000_000,
            commission_rate=0.0001,  # 1 bp
            slippage_bps=10,         # 10 bps
            borrow_cost_bps=30       # 30 bps short cost
        )
    
    def test_initialization(self):
        """Test backtester initialization"""
        self.assertEqual(self.backtester.initial_capital, 1_000_000)
        self.assertEqual(self.backtester.commission_rate, 0.0001)
        self.assertEqual(self.backtester.slippage_bps, 10)
        self.assertEqual(self.backtester.borrow_cost_bps, 30)
    
    def test_backtest_execution(self):
        """Test basic backtest execution"""
        results = self.backtester.run(
            positions=self.positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Check results structure
        self.assertIn('returns', results)
        self.assertIn('equity_curve', results)
        self.assertIn('positions', results)
        
        # Check returns series
        self.assertIsInstance(results['returns'], pd.Series)
        self.assertGreater(len(results['returns']), 0)
        
        # Check equity curve
        self.assertIsInstance(results['equity_curve'], pd.Series)
        self.assertEqual(results['equity_curve'].iloc[0], self.backtester.initial_capital)
    
    def test_equity_curve_monotonicity(self):
        """Test that equity curve starts at initial capital"""
        results = self.backtester.run(
            positions=self.positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # First value should be initial capital
        self.assertAlmostEqual(
            results['equity_curve'].iloc[0],
            self.backtester.initial_capital,
            places=2
        )
        
        # All values should be positive
        self.assertTrue(np.all(results['equity_curve'] > 0))
    
    def test_transaction_costs(self):
        """Test that transaction costs reduce returns"""
        # Run with costs
        results_with_costs = self.backtester.run(
            positions=self.positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Run without costs
        backtester_no_costs = Backtester(
            initial_capital=1_000_000,
            commission_rate=0.0,
            slippage_bps=0,
            borrow_cost_bps=0
        )
        
        results_no_costs = backtester_no_costs.run(
            positions=self.positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Final equity with costs should be lower
        final_with_costs = results_with_costs['equity_curve'].iloc[-1]
        final_no_costs = results_no_costs['equity_curve'].iloc[-1]
        
        self.assertLess(final_with_costs, final_no_costs)
    
    def test_long_only_portfolio(self):
        """Test with long-only positions"""
        # Create long-only positions
        position_dates = pd.date_range('2020-01-01', '2020-12-31', freq='MS')
        long_only_positions = pd.DataFrame(
            {
                'AAPL': [0.4] * 12,
                'MSFT': [0.3] * 12,
                'GOOGL': [0.3] * 12
            },
            index=position_dates
        )
        
        results = self.backtester.run(
            positions=long_only_positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Should complete successfully
        self.assertGreater(len(results['returns']), 0)
        self.assertGreater(results['equity_curve'].iloc[-1], 0)
    
    def test_short_only_portfolio(self):
        """Test with short-only positions"""
        # Create short-only positions
        position_dates = pd.date_range('2020-01-01', '2020-12-31', freq='MS')
        short_only_positions = pd.DataFrame(
            {
                'AAPL': [-0.4] * 12,
                'MSFT': [-0.3] * 12,
                'GOOGL': [-0.3] * 12
            },
            index=position_dates
        )
        
        results = self.backtester.run(
            positions=short_only_positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Should complete successfully
        # With upward trending prices, shorts should lose money
        self.assertGreater(len(results['returns']), 0)
    
    def test_rebalancing_detection(self):
        """Test detection of rebalancing events"""
        results = self.backtester.run(
            positions=self.positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Should detect rebalancing events
        # We have monthly positions, so should have ~12 rebalances
        position_changes = results['positions'].diff().abs().sum(axis=1)
        num_rebalances = (position_changes > 0.01).sum()
        
        self.assertGreater(num_rebalances, 10)  # At least 10 rebalances
        self.assertLess(num_rebalances, 15)     # Not too many
    
    def test_returns_calculation(self):
        """Test daily returns calculation"""
        results = self.backtester.run(
            positions=self.positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Returns should be reasonable
        returns = results['returns']
        
        # Mean return should be small (daily)
        self.assertLess(abs(returns.mean()), 0.01)
        
        # Std should be reasonable
        self.assertGreater(returns.std(), 0)
        self.assertLess(returns.std(), 0.1)
        
        # Should be mostly finite
        self.assertLess(returns.isna().sum(), len(returns) * 0.1)
    
    def test_equity_curve_consistency(self):
        """Test equity curve is consistent with returns"""
        results = self.backtester.run(
            positions=self.positions,
            prices=self.prices,
            start_date=pd.Timestamp('2020-01-01'),
            end_date=pd.Timestamp('2020-12-31')
        )
        
        # Equity curve should compound returns correctly
        equity = results['equity_curve']
        returns = results['returns']
        
        # Check first few returns match equity changes
        for i in range(1, min(10, len(returns))):
            expected_return = (equity.iloc[i] - equity.iloc[i-1]) / equity.iloc[i-1]
            actual_return = returns.iloc[i]
            
            # Should be close (allowing for rounding)
            self.assertAlmostEqual(expected_return, actual_return, places=6)
    
    def test_empty_positions(self):
        """Test handling of empty positions"""
        empty_positions = pd.DataFrame()
        
        # Should raise error or handle gracefully
        with self.assertRaises((ValueError, KeyError)):
            self.backtester.run(
                positions=empty_positions,
                prices=self.prices,
                start_date=pd.Timestamp('2020-01-01'),
                end_date=pd.Timestamp('2020-12-31')
            )
    
    def test_missing_price_data(self):
        """Test handling of missing price data"""
        # Create positions with ticker not in prices
        bad_positions = self.positions.copy()
        bad_positions['UNKNOWN'] = 0.1
        
        # Should handle gracefully (skip unknown ticker)
        try:
            results = self.backtester.run(
                positions=bad_positions,
                prices=self.prices,
                start_date=pd.Timestamp('2020-01-01'),
                end_date=pd.Timestamp('2020-12-31')
            )
            # If it doesn't error, check results are still valid
            self.assertGreater(len(results['returns']), 0)
        except (ValueError, KeyError):
            # Also acceptable to raise an error
            pass


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
