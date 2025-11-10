"""
Unit tests for PerformanceMetrics

Tests calculation of performance metrics including Sharpe ratio,
drawdowns, volatility, and risk-adjusted returns.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from analysis.performance_metrics import PerformanceMetrics


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceMetrics class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample return series
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')

        # Positive trending returns with volatility
        self.returns_positive = pd.Series(
            np.random.randn(len(dates)) * 0.01 + 0.0005,  # ~12% annual with 16% vol
            index=dates,
            name='return'
        )

        # Negative trending returns
        self.returns_negative = pd.Series(
            np.random.randn(len(dates)) * 0.01 - 0.0005,
            index=dates,
            name='return'
        )

        # High Sharpe returns (low vol, positive trend)
        self.returns_high_sharpe = pd.Series(
            np.random.randn(len(dates)) * 0.005 + 0.0008,  # ~20% return, 8% vol
            index=dates,
            name='return'
        )

    def test_sharpe_ratio_calculation(self):
        """Test Sharpe ratio calculation"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        sharpe = metrics.sharpe_ratio()

        # Check Sharpe ratio is reasonable
        self.assertIsInstance(sharpe, (int, float))
        self.assertGreater(sharpe, 0)  # Positive returns -> positive Sharpe
        self.assertLess(sharpe, 5)     # Not unrealistically high

    def test_high_sharpe_strategy(self):
        """Test that high Sharpe strategy has higher ratio"""
        metrics_regular = PerformanceMetrics(returns=self.returns_positive)
        metrics_high_sharpe = PerformanceMetrics(returns=self.returns_high_sharpe)

        # High Sharpe strategy should have higher Sharpe ratio
        self.assertGreater(
            metrics_high_sharpe.sharpe_ratio(),
            metrics_regular.sharpe_ratio()
        )

    def test_negative_sharpe(self):
        """Test negative Sharpe ratio for losing strategy"""
        metrics = PerformanceMetrics(returns=self.returns_negative)

        # Negative returns should produce negative Sharpe
        self.assertLess(metrics.sharpe_ratio(), 0)

    def test_cagr_calculation(self):
        """Test CAGR (Compound Annual Growth Rate) calculation"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        cagr = metrics.cagr()

        # Check CAGR is reasonable
        self.assertIsInstance(cagr, (int, float))

        # CAGR should be positive for positive returns
        self.assertGreater(cagr, 0)

        # Should be within reasonable range (~10-30% for this data)
        self.assertGreater(cagr, 0.05)
        self.assertLess(cagr, 0.35)

    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Create returns with known drawdown
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        returns_with_drawdown = pd.Series(0.01, index=dates, name='return')
        # Insert 20% drawdown
        returns_with_drawdown.iloc[50:100] = -0.005

        metrics = PerformanceMetrics(returns=returns_with_drawdown)

        max_dd = metrics.max_drawdown()

        # Check max drawdown
        self.assertLess(max_dd, 0)  # Drawdowns are negative
        self.assertGreater(max_dd, -0.5)  # Not catastrophic

    def test_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        sortino = metrics.sortino_ratio()
        sharpe = metrics.sharpe_ratio()

        # Sortino should be higher than Sharpe (only penalizes downside vol)
        self.assertGreaterEqual(sortino, sharpe)

    def test_calmar_ratio(self):
        """Test Calmar ratio (CAGR / Max Drawdown)"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        calmar = metrics.calmar_ratio()

        # Check Calmar exists
        self.assertIsInstance(calmar, (int, float))

        # For positive returns, Calmar should be positive
        self.assertGreater(calmar, 0)

    def test_volatility_calculation(self):
        """Test annualized volatility calculation"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        vol = metrics.volatility()

        # Check volatility
        self.assertGreater(vol, 0)

        # Should be reasonable (~10-20% for this data)
        self.assertGreater(vol, 0.05)
        self.assertLess(vol, 0.30)

    def test_var_and_cvar(self):
        """Test Value at Risk and Conditional VaR"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        var_95 = metrics.value_at_risk(confidence=0.95)
        cvar_95 = metrics.conditional_var(confidence=0.95)

        # VaR and CVaR should be negative (losses)
        self.assertLess(var_95, 0)
        self.assertLess(cvar_95, 0)

        # CVaR should be more negative than VaR (tail risk)
        self.assertLess(cvar_95, var_95)

    def test_win_rate(self):
        """Test win rate calculation"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        win_rate = metrics.win_rate()

        # Should be between 0 and 1
        self.assertGreaterEqual(win_rate, 0)
        self.assertLessEqual(win_rate, 1)

        # For positive expected returns, win rate should be > 50%
        self.assertGreater(win_rate, 0.50)

    def test_total_return_calculation(self):
        """Test total return calculation"""
        # Create simple returns for easy verification
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        simple_returns = pd.Series([0.01] * 10, index=dates, name='return')

        metrics = PerformanceMetrics(returns=simple_returns)

        # Total return should be approximately (1.01)^10 - 1 ≈ 10.46%
        expected_total = (1.01 ** 10) - 1
        self.assertAlmostEqual(metrics.total_return(), expected_total, places=4)

    def test_empty_returns(self):
        """Test handling of empty returns"""
        empty_returns = pd.Series([], dtype=float, name='return')

        # Should handle gracefully
        metrics = PerformanceMetrics(returns=empty_returns)
        # Empty returns should produce NaN or zero metrics
        self.assertTrue(np.isnan(metrics.sharpe_ratio()) or metrics.sharpe_ratio() == 0)

    def test_all_zero_returns(self):
        """Test handling of all zero returns"""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        zero_returns = pd.Series(0.0, index=dates, name='return')

        metrics = PerformanceMetrics(returns=zero_returns)

        # Should produce zero/nan for most metrics
        self.assertEqual(metrics.total_return(), 0)
        self.assertEqual(metrics.cagr(), 0)
        # Sharpe will be 0/0 = nan or 0
        self.assertTrue(np.isnan(metrics.sharpe_ratio()) or metrics.sharpe_ratio() == 0)

    def test_summary(self):
        """Test summary method returns all metrics"""
        metrics = PerformanceMetrics(returns=self.returns_positive)

        summary = metrics.summary()

        # Should be a dict
        self.assertIsInstance(summary, dict)

        # Should contain key metrics
        expected_keys = ['total_return', 'cagr', 'volatility', 'sharpe_ratio',
                        'sortino_ratio', 'max_drawdown', 'calmar_ratio']

        for key in expected_keys:
            self.assertIn(key, summary)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
