"""
Unit tests for SignalConstructor

Tests composite signal construction from price momentum,
sentiment momentum, and Frog-in-the-Pan scores.
"""

import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from signals.signal_constructor import SignalConstructor
from signals.frog_in_pan_detector import FrogInPanDetector


class TestSignalConstructor(unittest.TestCase):
    """Test cases for SignalConstructor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        # Create sample data
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        tickers = ['AAPL', 'MSFT', 'GOOGL']
        
        # Sample prices (trending up with noise)
        np.random.seed(42)
        self.prices = pd.DataFrame(
            np.cumsum(np.random.randn(len(dates), len(tickers)) * 0.02 + 0.001, axis=0) + 100,
            index=dates,
            columns=tickers
        )
        
        # Sample sentiment (mean-reverting with trend)
        self.sentiment = pd.DataFrame(
            np.random.randn(len(dates), len(tickers)) * 0.1 + 
            np.linspace(-0.2, 0.2, len(dates))[:, np.newaxis],
            index=dates,
            columns=tickers
        ).clip(-1, 1)
        
        # Initialize constructor
        self.constructor = SignalConstructor(
            price_weight=0.40,
            sentiment_weight=0.35,
            frog_weight=0.25,
            normalize_cross_section=False  # Disable for testing
        )
        
        # Initialize detector
        self.detector = FrogInPanDetector(
            gradual_window=63,
            sudden_threshold=2.0,
            min_persistence=126
        )
    
    def test_initialization(self):
        """Test constructor initialization"""
        self.assertAlmostEqual(self.constructor.price_weight, 0.40)
        self.assertAlmostEqual(self.constructor.sentiment_weight, 0.35)
        self.assertAlmostEqual(self.constructor.frog_weight, 0.25)
        self.assertFalse(self.constructor.normalize_cross_section)
    
    def test_weights_sum_to_one(self):
        """Test that signal weights sum to 1.0"""
        total_weight = (
            self.constructor.price_weight +
            self.constructor.sentiment_weight +
            self.constructor.frog_weight
        )
        self.assertAlmostEqual(total_weight, 1.0)
    
    def test_signal_construction(self):
        """Test composite signal construction"""
        signals = self.constructor.construct_signals(
            prices=self.prices,
            sentiment=self.sentiment,
            frog_detector=self.detector,
        )
        
        # Check output shape
        self.assertIsInstance(signals, pd.DataFrame)
        self.assertEqual(signals.shape[1], len(self.prices.columns))
        
        # Check dates
        self.assertGreaterEqual(signals.index[0], pd.Timestamp('2021-01-01'))
        self.assertLessEqual(signals.index[-1], pd.Timestamp('2023-12-31'))
        
        # Check signals are finite
        self.assertTrue(np.all(np.isfinite(signals.dropna())))
    
    def test_signal_range(self):
        """Test that signals have reasonable range"""
        signals = self.constructor.construct_signals(
            prices=self.prices,
            sentiment=self.sentiment,
            frog_detector=self.detector,
        )
        
        # Signals should typically be in reasonable range after standardization
        signal_mean = signals.mean().mean()
        signal_std = signals.std().mean()
        
        self.assertLess(abs(signal_mean), 1.0)  # Mean near zero
        self.assertGreater(signal_std, 0.1)      # Has variation
        self.assertLess(signal_std, 5.0)         # Not too extreme
    
    def test_price_momentum_component(self):
        """Test price momentum calculation"""
        # Create simple trending prices
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        simple_prices = pd.DataFrame(
            {'UP': np.linspace(100, 200, len(dates)),
             'DOWN': np.linspace(200, 100, len(dates))},
            index=dates
        )
        
        signals = self.constructor.construct_signals(
            prices=simple_prices,
            sentiment=self.sentiment[['AAPL', 'MSFT']].copy(),
            frog_detector=self.detector
        )
        
        # UP stock should generally have higher signal than DOWN stock
        avg_signal_up = signals['UP'].mean()
        avg_signal_down = signals['DOWN'].mean()
        self.assertGreater(avg_signal_up, avg_signal_down)
    
    def test_different_weight_combinations(self):
        """Test with different signal weight combinations"""
        weights = [
            (0.5, 0.3, 0.2),
            (0.33, 0.33, 0.34),
            (0.6, 0.25, 0.15)
        ]
        
        for w_price, w_sentiment, w_frog in weights:
            constructor = SignalConstructor(
                price_weight=w_price,
                sentiment_weight=w_sentiment,
                frog_weight=w_frog,
                normalize_cross_section=False
            )
            
            signals = constructor.construct_signals(
                prices=self.prices,
                sentiment=self.sentiment,
                frog_detector=self.detector
            )
            
            # Should produce valid signals
            self.assertIsInstance(signals, pd.DataFrame)
            self.assertGreater(len(signals), 0)
    
    def test_missing_data_handling(self):
        """Test handling of missing data"""
        # Create data with missing values
        prices_with_nan = self.prices.copy()
        prices_with_nan.iloc[100:110, 0] = np.nan
        
        signals = self.constructor.construct_signals(
            prices=prices_with_nan,
            sentiment=self.sentiment,
            frog_detector=self.detector,
        )
        
        # Should handle NaNs gracefully
        self.assertIsInstance(signals, pd.DataFrame)
        # Some values may be NaN, but not all
        self.assertLess(signals.isna().sum().sum(), len(signals) * len(signals.columns) * 0.5)
    
    def test_cross_sectional_standardization(self):
        """Test that signals are cross-sectionally standardized"""
        signals = self.constructor.construct_signals(
            prices=self.prices,
            sentiment=self.sentiment,
            frog_detector=self.detector,
        )
        
        # For each date, signals should be approximately standardized
        for date in signals.index[::50]:  # Sample every 50 days
            cross_section = signals.loc[date].dropna()
            if len(cross_section) > 1:
                mean = cross_section.mean()
                std = cross_section.std()
                
                # Should be approximately mean 0, std 1
                self.assertLess(abs(mean), 0.5)
                self.assertGreater(std, 0.5)
                self.assertLess(std, 2.0)


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False, verbosity=2)


if __name__ == '__main__':
    run_tests()
