"""
Order Flow Imbalance (OFI) feature computation
"""

import numpy as np
import pandas as pd
from typing import List, Optional


class OFIFeatures:
    """Compute Order Flow Imbalance features"""

    def __init__(self, windows: List[int] = None):
        """
        Parameters:
        -----------
        windows : List[int]
            List of time windows in seconds for OFI aggregation
            Default: [60, 300, 900] (1min, 5min, 15min)
        """
        self.windows = windows or [60, 300, 900]

    def compute_ofi_nbbo(self, nbbo_df: pd.DataFrame) -> pd.Series:
        """
        Compute OFI from NBBO changes (fallback if L3 unavailable)

        Uses NBBO changes to infer order flow direction

        Parameters:
        -----------
        nbbo_df : pd.DataFrame
            NBBO data

        Returns:
        --------
        pd.Series
            OFI time series
        """
        df = nbbo_df[['best_bid_price', 'best_bid_size',
                     'best_ask_price', 'best_ask_size']].copy()

        # Mid price changes indicate flow direction
        mid = (df['best_bid_price'] + df['best_ask_price']) / 2.0
        d_mid = mid.diff()

        # Size changes
        d_bid_sz = df['best_bid_size'].diff()
        d_ask_sz = df['best_ask_size'].diff()

        # OFI rule:
        # - If mid price up: attribute to bid flow
        # - If mid price down: attribute to ask flow
        # - If mid unchanged: use net size change
        arr = np.where(d_mid.values > 0, d_bid_sz.values,
              np.where(d_mid.values < 0, -d_ask_sz.values,
                       (d_bid_sz.values - d_ask_sz.values)))

        return pd.Series(arr, index=df.index).fillna(0.0).rename('ofi_nbbo')

    def compute_features(
        self,
        nbbo_df: pd.DataFrame,
        l3_ofi: Optional[pd.Series] = None
    ) -> pd.DataFrame:
        """
        Compute multi-timeframe OFI features

        Parameters:
        -----------
        nbbo_df : pd.DataFrame
            NBBO data (for alignment)
        l3_ofi : pd.Series, optional
            Pre-computed L3 OFI. If None, computes from NBBO

        Returns:
        --------
        pd.DataFrame
            OFI features with columns:
            - ofi_{window}s: Raw OFI sum over window
            - ofi_{window}s_norm: Normalized OFI (per-second rate)
        """
        idx = nbbo_df.index

        # Use L3 OFI if available, otherwise compute from NBBO
        if l3_ofi is not None and len(l3_ofi) > 0:
            ofi_1s = l3_ofi.reindex(idx, method='ffill').fillna(0.0)
        else:
            ofi_1s = self.compute_ofi_nbbo(nbbo_df)

        # Compute OFI over multiple windows
        features = {}
        for window_sec in self.windows:
            window_str = f'{window_sec}S'

            # Aggregate OFI over rolling window (left-closed to avoid lookahead bias)
            agg = ofi_1s.resample(window_str, closed='left', label='left').sum()
            cnt = ofi_1s.resample(window_str, closed='left', label='left').count().replace(0, np.nan)

            # Reindex to 1-second grid
            agg_1s = agg.reindex(idx, method='ffill').fillna(0)
            cnt_1s = cnt.reindex(idx, method='ffill')

            # Store raw and normalized OFI
            features[f'ofi_{window_sec}s'] = agg_1s
            features[f'ofi_{window_sec}s_norm'] = (agg_1s / cnt_1s).fillna(0.0)

        return pd.DataFrame(features, index=idx)

    def set_windows(self, windows: List[int]):
        """Update OFI time windows"""
        self.windows = windows
