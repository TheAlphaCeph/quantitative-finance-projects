"""
Market microstructure variable calculations.

This module computes the seven market microstructure variables
specified in the analysis, plus additional quality metrics
and statistical tests.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple, Optional


def compute_trade_returns(data: np.ndarray, price_col: int = 2) -> np.ndarray:
    """
    Compute trade price returns.

    Returns: (P_t - P_{t-1}) / P_{t-1}

    Args:
        data: Order book data (seq_len, n_features) or (n_days, seq_len, n_features)
        price_col: Column index for trade price

    Returns:
        Array of returns
    """
    if data.ndim == 2:
        prices = data[:, price_col].astype(float)
        returns = np.diff(prices) / prices[:-1]
        return returns[np.isfinite(returns)]
    else:
        all_returns = []
        for day in data:
            returns = compute_trade_returns(day, price_col)
            all_returns.append(returns)
        return all_returns


def compute_mid_returns(data: np.ndarray,
                        ask_col: int = 5, bid_col: int = 6) -> np.ndarray:
    """
    Compute mid-quote returns.

    Mid-quote = (SP1 + BP1) / 2
    Returns = (mid_t - mid_{t-1}) / mid_{t-1}

    Args:
        data: Order book data
        ask_col: Column index for best ask (SP1)
        bid_col: Column index for best bid (BP1)

    Returns:
        Array of mid-quote returns
    """
    if data.ndim == 2:
        mid = (data[:, ask_col].astype(float) + data[:, bid_col].astype(float)) / 2
        returns = np.diff(mid) / mid[:-1]
        return returns[np.isfinite(returns)]
    else:
        all_returns = []
        for day in data:
            returns = compute_mid_returns(day, ask_col, bid_col)
            all_returns.append(returns)
        return all_returns


def compute_trade_size(data: np.ndarray, size_col: int = 3) -> np.ndarray:
    """
    Extract trade sizes.

    Args:
        data: Order book data
        size_col: Column index for trade size

    Returns:
        Array of trade sizes
    """
    if data.ndim == 2:
        return data[:, size_col].astype(float)
    else:
        return [day[:, size_col].astype(float) for day in data]


def compute_spread(data: np.ndarray,
                   ask_col: int = 5, bid_col: int = 6) -> np.ndarray:
    """
    Compute bid-ask spread.

    Spread = SP1 - BP1

    Args:
        data: Order book data
        ask_col: Column index for best ask
        bid_col: Column index for best bid

    Returns:
        Array of spreads
    """
    if data.ndim == 2:
        return data[:, ask_col].astype(float) - data[:, bid_col].astype(float)
    else:
        return [compute_spread(day, ask_col, bid_col) for day in data]


def compute_spread_diff(data: np.ndarray,
                        ask_col: int = 5, bid_col: int = 6) -> np.ndarray:
    """
    Compute first-order difference of spread.

    Args:
        data: Order book data

    Returns:
        Array of spread changes
    """
    if data.ndim == 2:
        spread = compute_spread(data, ask_col, bid_col)
        return np.diff(spread)
    else:
        return [compute_spread_diff(day, ask_col, bid_col) for day in data]


def compute_ob_pressure_1(data: np.ndarray,
                          bv1_col: int = 8, sv1_col: int = 7) -> np.ndarray:
    """
    Compute 1-level order book pressure.

    Pressure = (BV1 - SV1) / (BV1 + SV1)

    Interpretation:
    - Positive: More buying pressure (larger bid volume)
    - Negative: More selling pressure (larger ask volume)
    - Zero: Balanced

    Args:
        data: Order book data
        bv1_col: Column index for best bid volume (BV1)
        sv1_col: Column index for best ask volume (SV1)

    Returns:
        Array of order book pressure values
    """
    if data.ndim == 2:
        bv1 = data[:, bv1_col].astype(float)
        sv1 = data[:, sv1_col].astype(float)
        return (bv1 - sv1) / (bv1 + sv1 + 1e-10)
    else:
        return [compute_ob_pressure_1(day, bv1_col, sv1_col) for day in data]


def compute_ob_pressure_5(data: np.ndarray,
                          bv_cols: List[int] = None,
                          sv_cols: List[int] = None) -> np.ndarray:
    """
    Compute 5-level order book pressure.

    Pressure = Sum(BVi - SVi) / Sum(BVi + SVi) for i = 1..5

    Args:
        data: Order book data
        bv_cols: Column indices for bid volumes (BV1-5)
        sv_cols: Column indices for ask volumes (SV1-5)

    Returns:
        Array of 5-level order book pressure values
    """
    # Default column indices (assuming standard order)
    if bv_cols is None:
        bv_cols = [8, 12, 16, 20, 24]  # BV1-5
    if sv_cols is None:
        sv_cols = [7, 11, 15, 19, 23]  # SV1-5

    if data.ndim == 2:
        bv_sum = data[:, bv_cols].astype(float).sum(axis=1)
        sv_sum = data[:, sv_cols].astype(float).sum(axis=1)
        return (bv_sum - sv_sum) / (bv_sum + sv_sum + 1e-10)
    else:
        return [compute_ob_pressure_5(day, bv_cols, sv_cols) for day in data]


def compute_daily_statistics(data: np.ndarray, day_idx: int) -> Dict:
    """
    Compute all microstructure statistics for a single day.

    Args:
        data: Full order book data (n_days, seq_len, n_features)
        day_idx: Index of day to analyze

    Returns:
        Dictionary of statistics
    """
    day_data = data[day_idx]

    # Compute all variables
    trade_returns = compute_trade_returns(day_data)
    mid_returns = compute_mid_returns(day_data)
    trade_size = compute_trade_size(day_data)
    spread = compute_spread(day_data)
    spread_diff = compute_spread_diff(day_data)
    pressure_1 = compute_ob_pressure_1(day_data)
    pressure_5 = compute_ob_pressure_5(day_data)

    return {
        'trade_returns_mean': np.nanmean(trade_returns),
        'trade_returns_std': np.nanstd(trade_returns),
        'trade_returns_skew': stats.skew(trade_returns[np.isfinite(trade_returns)]),
        'trade_returns_kurtosis': stats.kurtosis(trade_returns[np.isfinite(trade_returns)]),

        'mid_returns_mean': np.nanmean(mid_returns),
        'mid_returns_std': np.nanstd(mid_returns),

        'trade_size_mean': np.nanmean(trade_size),
        'trade_size_std': np.nanstd(trade_size),

        'spread_mean': np.nanmean(spread),
        'spread_std': np.nanstd(spread),

        'spread_diff_mean': np.nanmean(spread_diff),
        'spread_diff_std': np.nanstd(spread_diff),

        'pressure_1_mean': np.nanmean(pressure_1),
        'pressure_1_std': np.nanstd(pressure_1),

        'pressure_5_mean': np.nanmean(pressure_5),
        'pressure_5_std': np.nanstd(pressure_5)
    }


def compute_all_variables(data: np.ndarray, indices: List[int]) -> pd.DataFrame:
    """
    Compute microstructure statistics for specified days.

    Args:
        data: Full order book data
        indices: Day indices to analyze

    Returns:
        DataFrame with one row per day
    """
    results = []
    for idx in indices:
        stats_dict = compute_daily_statistics(data, idx)
        stats_dict['day_idx'] = idx
        results.append(stats_dict)

    return pd.DataFrame(results)


def run_ks_tests(abnormal_df: pd.DataFrame,
                 normal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Kolmogorov-Smirnov tests comparing abnormal vs normal days.

    The K-S test measures the maximum difference between two empirical CDFs.
    H0: The two samples come from the same distribution.

    Args:
        abnormal_df: Microstructure stats for abnormal days
        normal_df: Microstructure stats for normal days

    Returns:
        DataFrame with K-S statistics and p-values
    """
    variables = [
        'trade_returns_mean',
        'mid_returns_mean',
        'trade_size_mean',
        'spread_mean',
        'spread_diff_mean',
        'pressure_1_mean',
        'pressure_5_mean'
    ]

    results = []
    for var in variables:
        if var in abnormal_df.columns and var in normal_df.columns:
            abn = abnormal_df[var].dropna().values
            norm = normal_df[var].dropna().values

            if len(abn) > 0 and len(norm) > 0:
                ks_stat, p_value = stats.ks_2samp(abn, norm)
                results.append({
                    'Variable': var,
                    'KS_Statistic': ks_stat,
                    'P_Value': p_value,
                    'Significant_0.05': 'Yes' if p_value < 0.05 else 'No',
                    'Significant_0.01': 'Yes' if p_value < 0.01 else 'No'
                })

    return pd.DataFrame(results)


def compare_statistics(abnormal_df: pd.DataFrame,
                       normal_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare mean statistics between abnormal and normal days.

    Args:
        abnormal_df: Microstructure stats for abnormal days
        normal_df: Microstructure stats for normal days

    Returns:
        Comparison DataFrame
    """
    variables = [
        'trade_returns_mean',
        'mid_returns_mean',
        'trade_size_mean',
        'spread_mean',
        'spread_diff_mean',
        'pressure_1_mean',
        'pressure_5_mean'
    ]

    results = []
    for var in variables:
        if var in abnormal_df.columns and var in normal_df.columns:
            abn_mean = abnormal_df[var].mean()
            norm_mean = normal_df[var].mean()
            abn_std = abnormal_df[var].std()
            norm_std = normal_df[var].std()

            diff_pct = ((abn_mean - norm_mean) / abs(norm_mean) * 100
                       if norm_mean != 0 else np.nan)

            results.append({
                'Variable': var,
                'Normal_Mean': norm_mean,
                'Normal_Std': norm_std,
                'Abnormal_Mean': abn_mean,
                'Abnormal_Std': abn_std,
                'Difference_%': diff_pct
            })

    return pd.DataFrame(results)
