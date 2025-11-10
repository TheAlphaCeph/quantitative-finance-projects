"""
Gamma-Weighted Microprice Strategy
"""

import pandas as pd
import numpy as np
from typing import Dict
from ..features.microprice import MicropriceFeatures
from ..features.ofi import OFIFeatures


class GammaMicropriceStrategy:
    """
    High-frequency trading strategy based on gamma-weighted microprice
    and Order Flow Imbalance signals
    """

    def __init__(
        self,
        gamma_shape: float = 2.0,
        pressure_half_life: float = 60.0,
        ofi_windows: list = None,
        signal_threshold: float = 0.5,
        signal_weights: Dict[str, float] = None
    ):
        """
        Parameters:
        -----------
        gamma_shape : float
            Gamma shape parameter for microprice weighting
        pressure_half_life : float
            EMA half-life for pressure integration (seconds)
        ofi_windows : list
            Time windows for OFI aggregation (seconds)
        signal_threshold : float
            Threshold for signal generation
        signal_weights : dict
            Weights for composite signal components
        """
        self.gamma_shape = gamma_shape
        self.pressure_half_life = pressure_half_life
        self.ofi_windows = ofi_windows or [60, 300, 900]
        self.signal_threshold = signal_threshold
        self.signal_weights = signal_weights or {
            'pressure': 0.5,
            'ofi': 0.3,
            'volume': 0.2
        }

        # Initialize feature computers
        self.microprice_features = MicropriceFeatures(
            gamma_shape=gamma_shape,
            pressure_half_life=pressure_half_life
        )
        self.ofi_features = OFIFeatures(windows=ofi_windows)

    def compute_features(
        self,
        nbbo_df: pd.DataFrame,
        l3_ofi: pd.Series = None
    ) -> pd.DataFrame:
        """
        Compute all strategy features

        Parameters:
        -----------
        nbbo_df : pd.DataFrame
            NBBO data
        l3_ofi : pd.Series, optional
            Pre-computed L3 OFI

        Returns:
        --------
        pd.DataFrame
            Complete feature set
        """
        # Compute microprice features
        micro_feats = self.microprice_features.compute_features(nbbo_df)

        # Compute OFI features
        ofi_feats = self.ofi_features.compute_features(nbbo_df, l3_ofi)

        # Combine features
        features = pd.concat([micro_feats, ofi_feats], axis=1)

        # Remove any rows with NaN
        features = features.replace([np.inf, -np.inf], np.nan).dropna(how='any')

        return features

    def generate_signals(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from features

        Parameters:
        -----------
        features : pd.DataFrame
            Feature dataframe from compute_features()

        Returns:
        --------
        pd.DataFrame
            Signals with columns:
            - pressure_signal: Standardized pressure component
            - ofi_signal: Standardized OFI component
            - volume_signal: Standardized volume component
            - composite: Combined signal score
            - position: Trading position (+1, -1, 0)
        """
        signals = pd.DataFrame(index=features.index)

        # Standardize features using expanding window to avoid lookahead bias
        feat_norm = features.copy()
        for col in feat_norm.columns:
            expanding_mean = feat_norm[col].expanding(min_periods=100).mean()
            expanding_std = feat_norm[col].expanding(min_periods=100).std()
            feat_norm[col] = (feat_norm[col] - expanding_mean) / (expanding_std + 1e-8)

        # Component signals
        signals['pressure_signal'] = feat_norm['integrated_pressure'].fillna(0)

        # Average normalized OFI across all windows
        ofi_cols = [c for c in feat_norm.columns
                   if c.startswith('ofi_') and c.endswith('_norm')]
        if ofi_cols:
            signals['ofi_signal'] = feat_norm[ofi_cols].mean(axis=1)
        else:
            signals['ofi_signal'] = 0.0

        signals['volume_signal'] = feat_norm['volume_imbalance'].fillna(0)

        # Composite signal (weighted sum)
        signals['composite'] = (
            self.signal_weights['pressure'] * signals['pressure_signal'] +
            self.signal_weights['ofi'] * signals['ofi_signal'] +
            self.signal_weights['volume'] * signals['volume_signal']
        )

        # Generate positions based on threshold
        signals['position'] = 0
        signals.loc[signals['composite'] > self.signal_threshold, 'position'] = 1
        signals.loc[signals['composite'] < -self.signal_threshold, 'position'] = -1

        return signals

    def run(
        self,
        nbbo_df: pd.DataFrame,
        l3_ofi: pd.Series = None
    ) -> tuple:
        """
        Run complete strategy pipeline

        Parameters:
        -----------
        nbbo_df : pd.DataFrame
            NBBO data
        l3_ofi : pd.Series, optional
            L3 OFI data

        Returns:
        --------
        tuple : (features, signals)
        """
        features = self.compute_features(nbbo_df, l3_ofi)
        signals = self.generate_signals(features)
        return features, signals

    def update_parameters(self, **kwargs):
        """Update strategy parameters"""
        if 'gamma_shape' in kwargs:
            self.gamma_shape = kwargs['gamma_shape']
            self.microprice_features.set_gamma_shape(kwargs['gamma_shape'])

        if 'pressure_half_life' in kwargs:
            self.pressure_half_life = kwargs['pressure_half_life']
            self.microprice_features.set_pressure_half_life(kwargs['pressure_half_life'])

        if 'ofi_windows' in kwargs:
            self.ofi_windows = kwargs['ofi_windows']
            self.ofi_features.set_windows(kwargs['ofi_windows'])

        if 'signal_threshold' in kwargs:
            self.signal_threshold = kwargs['signal_threshold']

        if 'signal_weights' in kwargs:
            self.signal_weights.update(kwargs['signal_weights'])
