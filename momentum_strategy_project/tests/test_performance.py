"""
Unit tests for PerformanceAnalyzer

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

from analysis.performance_metrics import PerformanceAnalyzer


class TestPerformanceMetrics(unittest.TestCase):
    """Test cases for PerformanceAnalyzer class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.analyzer = PerformanceAnalyzer()
        
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
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check Sharpe ratio exists and is reasonable
        self.assertIn('sharpe_ratio', metrics)
        self.assertIsInstance(metrics['sharpe_ratio'], (int, float))
        self.assertGreater(metrics['sharpe_ratio'], 0)  # Positive returns -> positive Sharpe
        self.assertLess(metrics['sharpe_ratio'], 5)     # Not unrealistically high
    
    def test_high_sharpe_strategy(self):
        """Test that high Sharpe strategy has higher ratio"""
        metrics_regular = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        metrics_high_sharpe = self.analyzer.calculate_metrics(
            returns=self.returns_high_sharpe,
            positions=None,
            benchmark_returns=None
        )
        
        # High Sharpe strategy should have higher Sharpe ratio
        self.assertGreater(
            metrics_high_sharpe['sharpe_ratio'],
            metrics_regular['sharpe_ratio']
        )
    
    def test_negative_sharpe(self):
        """Test negative Sharpe ratio for losing strategy"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_negative,
            positions=None,
            benchmark_returns=None
        )
        
        # Negative returns should produce negative Sharpe
        self.assertLess(metrics['sharpe_ratio'], 0)
    
    def test_cagr_calculation(self):
        """Test CAGR (Compound Annual Growth Rate) calculation"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check CAGR exists and is reasonable
        self.assertIn('cagr', metrics)
        self.assertIsInstance(metrics['cagr'], (int, float))
        
        # CAGR should be positive for positive returns
        self.assertGreater(metrics['cagr'], 0)
        
        # Should be within reasonable range (~10-15% for this data)
        self.assertGreater(metrics['cagr'], 0.05)
        self.assertLess(metrics['cagr'], 0.25)
    
    def test_max_drawdown_calculation(self):
        """Test maximum drawdown calculation"""
        # Create returns with known drawdown
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        returns_with_drawdown = pd.Series(0.01, index=dates, name='return')
        # Insert 20% drawdown
        returns_with_drawdown.iloc[50:100] = -0.005
        
        metrics = self.analyzer.calculate_metrics(
            returns=returns_with_drawdown,
            positions=None,
            benchmark_returns=None
        )
        
        # Check max drawdown
        self.assertIn('max_drawdown', metrics)
        self.assertLess(metrics['max_drawdown'], 0)  # Drawdowns are negative
        self.assertGreater(metrics['max_drawdown'], -0.5)  # Not catastrophic
    
    def test_sortino_ratio(self):
        """Test Sortino ratio calculation"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check Sortino exists
        self.assertIn('sortino_ratio', metrics)
        
        # Sortino should be higher than Sharpe (only penalizes downside vol)
        self.assertGreaterEqual(metrics['sortino_ratio'], metrics['sharpe_ratio'])
    
    def test_calmar_ratio(self):
        """Test Calmar ratio (CAGR / Max Drawdown)"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check Calmar exists
        self.assertIn('calmar_ratio', metrics)
        self.assertIsInstance(metrics['calmar_ratio'], (int, float))
        
        # For positive returns, Calmar should be positive
        self.assertGreater(metrics['calmar_ratio'], 0)
    
    def test_volatility_calculation(self):
        """Test annualized volatility calculation"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check volatility
        self.assertIn('volatility', metrics)
        self.assertGreater(metrics['volatility'], 0)
        
        # Should be reasonable (~10-20% for this data)
        self.assertGreater(metrics['volatility'], 0.05)
        self.assertLess(metrics['volatility'], 0.30)
    
    def test_skewness_and_kurtosis(self):
        """Test distribution statistics"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check they exist
        self.assertIn('skewness', metrics)
        self.assertIn('kurtosis', metrics)
        
        # Should be finite
        self.assertTrue(np.isfinite(metrics['skewness']))
        self.assertTrue(np.isfinite(metrics['kurtosis']))
    
    def test_var_and_cvar(self):
        """Test Value at Risk and Conditional VaR"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check VaR and CVaR exist
        self.assertIn('var_95', metrics)
        self.assertIn('cvar_95', metrics)
        
        # VaR and CVaR should be negative (losses)
        self.assertLess(metrics['var_95'], 0)
        self.assertLess(metrics['cvar_95'], 0)
        
        # CVaR should be more negative than VaR (tail risk)
        self.assertLess(metrics['cvar_95'], metrics['var_95'])
    
    def test_win_rate(self):
        """Test win rate calculation"""
        metrics = self.analyzer.calculate_metrics(
            returns=self.returns_positive,
            positions=None,
            benchmark_returns=None
        )
        
        # Check win rate
        self.assertIn('win_rate', metrics)
        
        # Should be between 0 and 1
        self.assertGreaterEqual(metrics['win_rate'], 0)
        self.assertLessEqual(metrics['win_rate'], 1)
        
        # For positive expected returns, win rate should be > 50%
        self.assertGreater(metrics['win_rate'], 0.50)
    
    def test_total_return_calculation(self):
        """Test total return calculation"""
        # Create simple returns for easy verification
        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        simple_returns = pd.Series([0.01] * 10, index=dates, name='return')
        
        metrics = self.analyzer.calculate_metrics(
            returns=simple_returns,
            positions=None,
            benchmark_returns=None
        )
        
        # Total return should be approximately (1.01)^10 - 1 ≈ 10.46%
        expected_total = (1.01 ** 10) - 1
        self.assertAlmostEqual(metrics['total_return'], expected_total, places=4)
    
    def test_empty_returns(self):
        """Test handling of empty returns"""
        empty_returns = pd.Series([], dtype=float, name='return')
        
        # Should handle gracefully
        with self.assertRaises(ValueError):
            self.analyzer.calculate_metrics(
                returns=empty_returns,
                positions=None,
                benchmark_returns=None
            )
    
    def test_all_zero_returns(self):
        """Test handling of all zero returns"""
        dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
        zero_returns = pd.Series(0.0, index=dates, name='return')
        
        metrics = self.analyzer.calculate_metrics(
            returns=zero_returns,
            positions=None,
            benchmark_returns=None
        )
        
        # Should produce zero/nan for most metrics
        self.assertEqual(metrics['total_return'], 0)
        self.assertEqual(metrics['cagr'], 0)
        # Sharpe will be 0/0 = nan or 0
        self.assertTrue(np.isnan(metrics['sharpe_ratio']) or metrics['sharpe_ratio'] == 0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
