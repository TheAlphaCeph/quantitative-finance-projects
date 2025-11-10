"""
Unit tests for MicropriceFeatures

Tests gamma-weighted microprice computation and pressure signals.
"""

import unittest
import pandas as pd
import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.microprice import MicropriceFeatures


class TestMicropriceFeatures(unittest.TestCase):
    """Test cases for MicropriceFeatures class"""

    def setUp(self):
        """Set up test fixtures"""
        # Create sample NBBO data
        dates = pd.date_range('2024-01-01 09:30:00', '2024-01-01 10:30:00', freq='1S')
        n = len(dates)

        np.random.seed(42)
        self.nbbo_df = pd.DataFrame({
            'best_bid_price': 100.0 + np.random.randn(n) * 0.01,
            'best_ask_price': 100.05 + np.random.randn(n) * 0.01,
            'best_bid_size': np.random.randint(100, 1000, n),
            'best_ask_size': np.random.randint(100, 1000, n)
        }, index=dates)

        # Ensure bid < ask
        self.nbbo_df['best_ask_price'] = self.nbbo_df['best_bid_price'] + 0.05

        self.features_computer = MicropriceFeatures(gamma_shape=2.0, pressure_half_life=60.0)

    def test_feature_computation(self):
        """Test basic feature computation"""
        features = self.features_computer.compute_features(self.nbbo_df)

        # Check all expected columns are present
        expected_cols = [
            'microprice', 'mid_price', 'weighted_deviation',
            'integrated_pressure', 'spread', 'volume_imbalance'
        ]

        for col in expected_cols:
            self.assertIn(col, features.columns, f"Features should include {col}")

        self.assertEqual(len(features), len(self.nbbo_df), "Should have same length as input")

    def test_microprice_bounds(self):
        """Test that microprice is between bid and ask"""
        features = self.features_computer.compute_features(self.nbbo_df)

        # Microprice should be between bid and ask
        self.assertTrue(
            (features['microprice'] >= self.nbbo_df['best_bid_price']).all(),
            "Microprice should be >= bid"
        )
        self.assertTrue(
            (features['microprice'] <= self.nbbo_df['best_ask_price']).all(),
            "Microprice should be <= ask"
        )

    def test_mid_price(self):
        """Test mid price calculation"""
        features = self.features_computer.compute_features(self.nbbo_df)

        expected_mid = (self.nbbo_df['best_bid_price'] + self.nbbo_df['best_ask_price']) / 2.0
        pd.testing.assert_series_equal(
            features['mid_price'],
            expected_mid,
            check_names=False
        )

    def test_spread(self):
        """Test spread calculation"""
        features = self.features_computer.compute_features(self.nbbo_df)

        expected_spread = self.nbbo_df['best_ask_price'] - self.nbbo_df['best_bid_price']
        pd.testing.assert_series_equal(
            features['spread'],
            expected_spread,
            check_names=False
        )

    def test_volume_imbalance(self):
        """Test volume imbalance calculation"""
        features = self.features_computer.compute_features(self.nbbo_df)

        # Volume imbalance should be between -1 and 1
        self.assertTrue(
            (features['volume_imbalance'] >= -1).all(),
            "Volume imbalance should be >= -1"
        )
        self.assertTrue(
            (features['volume_imbalance'] <= 1).all(),
            "Volume imbalance should be <= 1"
        )

    def test_gamma_parameter_update(self):
        """Test gamma parameter update"""
        original_gamma = self.features_computer.gamma_shape
        self.features_computer.set_gamma_shape(3.0)

        self.assertEqual(self.features_computer.gamma_shape, 3.0)
        self.assertNotEqual(self.features_computer.gamma_shape, original_gamma)


if __name__ == '__main__':
    unittest.main()
