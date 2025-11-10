"""
Microprice feature computation
"""

import numpy as np
import pandas as pd
from ..utils.numba_functions import compute_weighted_microprice, integrate_ema


class MicropriceFeatures:
    """Compute microprice-based features from orderbook data"""

    def __init__(self, gamma_shape: float = 2.0, pressure_half_life: float = 60.0):
        """
        Parameters:
        -----------
        gamma_shape : float
            Shape parameter for gamma weighting
        pressure_half_life : float
            Half-life in seconds for pressure EMA integration
        """
        self.gamma_shape = gamma_shape
        self.pressure_half_life = pressure_half_life

    def compute_features(self, nbbo_df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute microprice features from NBBO data

        Parameters:
        -----------
        nbbo_df : pd.DataFrame
            NBBO data with columns: best_bid_price, best_ask_price,
            best_bid_size, best_ask_size

        Returns:
        --------
        pd.DataFrame
            Features with columns:
            - microprice: Gamma-weighted microprice
            - mid_price: Simple mid price
            - weighted_deviation: Weighted deviation from mid
            - integrated_pressure: EMA-integrated pressure signal
            - spread: Bid-ask spread
            - volume_imbalance: Top-of-book volume imbalance
        """
        # Extract price and size arrays
        bp = nbbo_df['best_bid_price'].values.astype(np.float64)
        ap = nbbo_df['best_ask_price'].values.astype(np.float64)
        bs = nbbo_df['best_bid_size'].values.astype(np.float64)
        az = nbbo_df['best_ask_size'].values.astype(np.float64)

        # Compute weighted microprice
        micro, wdev, weights = compute_weighted_microprice(
            bp, bs, ap, az, self.gamma_shape
        )

        # Integrate pressure signal with EMA
        timestamps = nbbo_df.index.view(np.int64)
        integrated = integrate_ema(wdev, timestamps, self.pressure_half_life)

        # Calculate additional features
        mid = 0.5 * (bp + ap)
        spread = ap - bp
        vol_imbalance = (bs - az) / (bs + az + 1e-9)

        # Build features dataframe
        features = pd.DataFrame({
            'microprice': micro,
            'mid_price': mid,
            'weighted_deviation': wdev,
            'integrated_pressure': integrated,
            'spread': spread,
            'volume_imbalance': vol_imbalance
        }, index=nbbo_df.index)

        return features

    def set_gamma_shape(self, gamma_shape: float):
        """Update gamma shape parameter"""
        self.gamma_shape = gamma_shape

    def set_pressure_half_life(self, half_life: float):
        """Update pressure EMA half-life"""
        self.pressure_half_life = half_life
