"""
Synthetic order book generation and quality assessment.

This module provides functionality for generating synthetic order book
data using the trained Generator and assessing data quality.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
import pandas as pd

from .models import Generator


class SyntheticGenerator:
    """
    Generator for synthetic order book data.

    Uses a trained GAN generator to create realistic synthetic order book
    sequences that can be used for data augmentation or analysis.

    Example:
        >>> synth_gen = SyntheticGenerator(generator, X_mean, X_std)
        >>> synthetic_data = synth_gen.generate(n_days=10)
    """

    def __init__(self, generator: Generator,
                 X_mean: np.ndarray,
                 X_std: np.ndarray,
                 device: str = "cpu"):
        """
        Initialize synthetic generator.

        Args:
            generator: Trained generator model
            X_mean: Mean statistics from training data
            X_std: Standard deviation statistics from training data
            device: Device to use (cpu/cuda)
        """
        self.generator = generator
        self.X_mean = X_mean
        self.X_std = X_std
        self.device = torch.device(device)
        self.generator.to(self.device)
        self.generator.eval()

    def generate(self, n_days: int = 1,
                 seq_len: int = 265,
                 n_features: int = 20,
                 denormalize: bool = True) -> np.ndarray:
        """
        Generate synthetic order book sequences.

        Args:
            n_days: Number of days (sequences) to generate
            seq_len: Sequence length (minutes per day)
            n_features: Number of features
            denormalize: Whether to convert back to original scale

        Returns:
            Synthetic data of shape (n_days, seq_len, n_features)
        """
        with torch.no_grad():
            # Generate random noise
            noise = torch.randn(n_days, seq_len, n_features).to(self.device)

            # Generate synthetic data
            synthetic = self.generator(noise).cpu().numpy()

        if denormalize:
            synthetic = self._denormalize(synthetic)

        return synthetic

    def _denormalize(self, data: np.ndarray) -> np.ndarray:
        """
        Convert normalized data back to original scale.

        Args:
            data: Normalized data

        Returns:
            Denormalized data in original scale
        """
        # Use average statistics across training days
        mean_avg = self.X_mean.mean(axis=0)
        std_avg = self.X_std.mean(axis=0)

        # Reverse z-score normalization
        denorm = data.copy()
        for i in range(len(denorm)):
            denorm[i] = denorm[i] * (2 * std_avg) + mean_avg

        # Reverse log transform for volumes (last 10 features)
        denorm[:, :, -10:] = np.exp(denorm[:, :, -10:]) - 1
        denorm[:, :, -10:] = np.maximum(denorm[:, :, -10:], 0)  # Ensure non-negative

        return denorm


def assess_quality(synthetic_data: np.ndarray,
                   ask_cols: List[int] = None,
                   bid_cols: List[int] = None,
                   sv_cols: List[int] = None,
                   bv_cols: List[int] = None) -> Dict:
    """
    Assess quality of synthetic order book data.

    Checks for:
    1. Arbitrage violations (ask < bid)
    2. Negative volumes
    3. Crossed quotes at any level

    Args:
        synthetic_data: Generated order book data
        ask_cols: Column indices for ask prices (SP1-5)
        bid_cols: Column indices for bid prices (BP1-5)
        sv_cols: Column indices for ask volumes (SV1-5)
        bv_cols: Column indices for bid volumes (BV1-5)

    Returns:
        Dictionary with quality metrics
    """
    # Default column indices (5-level LOB structure)
    if ask_cols is None:
        ask_cols = [0, 4, 8, 12, 16]  # SP1-5
    if bid_cols is None:
        bid_cols = [1, 5, 9, 13, 17]  # BP1-5
    if sv_cols is None:
        sv_cols = [2, 6, 10, 14, 18]  # SV1-5
    if bv_cols is None:
        bv_cols = [3, 7, 11, 15, 19]  # BV1-5

    n_days = len(synthetic_data)
    total_obs = n_days * synthetic_data.shape[1]

    results = {
        'n_days': n_days,
        'total_observations': total_obs
    }

    # Check arbitrage violations (ask <= bid at any level)
    arbitrage_violations = 0
    for i, (ask_col, bid_col) in enumerate(zip(ask_cols, bid_cols)):
        violations = (synthetic_data[:, :, ask_col] <= synthetic_data[:, :, bid_col]).sum()
        arbitrage_violations += violations
        results[f'level_{i+1}_arbitrage_violations'] = int(violations)

    results['total_arbitrage_violations'] = int(arbitrage_violations)
    results['arbitrage_violation_rate'] = arbitrage_violations / (total_obs * len(ask_cols))

    # Check negative volumes
    negative_volumes = 0
    for i, (sv_col, bv_col) in enumerate(zip(sv_cols, bv_cols)):
        sv_neg = (synthetic_data[:, :, sv_col] < 0).sum()
        bv_neg = (synthetic_data[:, :, bv_col] < 0).sum()
        negative_volumes += sv_neg + bv_neg
        results[f'level_{i+1}_negative_volumes'] = int(sv_neg + bv_neg)

    results['total_negative_volumes'] = int(negative_volumes)
    results['negative_volume_rate'] = negative_volumes / (total_obs * len(sv_cols) * 2)

    # Check price ordering (SP1 < SP2 < ... and BP1 > BP2 > ...)
    price_order_violations = 0

    # Ask prices should be increasing (SP1 < SP2 < SP3...)
    for i in range(len(ask_cols) - 1):
        violations = (synthetic_data[:, :, ask_cols[i]] >= synthetic_data[:, :, ask_cols[i+1]]).sum()
        price_order_violations += violations

    # Bid prices should be decreasing (BP1 > BP2 > BP3...)
    for i in range(len(bid_cols) - 1):
        violations = (synthetic_data[:, :, bid_cols[i]] <= synthetic_data[:, :, bid_cols[i+1]]).sum()
        price_order_violations += violations

    results['price_order_violations'] = int(price_order_violations)

    # Overall quality score (1 = perfect, 0 = all violations)
    total_possible_violations = (
        total_obs * len(ask_cols) +  # Arbitrage
        total_obs * len(sv_cols) * 2 +  # Volumes
        total_obs * (len(ask_cols) - 1) * 2  # Price ordering
    )
    total_violations = arbitrage_violations + negative_volumes + price_order_violations
    results['quality_score'] = 1 - (total_violations / total_possible_violations)

    return results


def compare_distributions(real_data: np.ndarray,
                          synthetic_data: np.ndarray,
                          feature_names: List[str] = None) -> pd.DataFrame:
    """
    Compare statistical distributions between real and synthetic data.

    Args:
        real_data: Real order book data
        synthetic_data: Generated order book data
        feature_names: Names of features

    Returns:
        DataFrame with distribution comparison statistics
    """
    from scipy import stats

    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(real_data.shape[-1])]

    results = []

    for i, name in enumerate(feature_names):
        # Flatten to 1D for comparison
        real_flat = real_data[:, :, i].flatten()
        synth_flat = synthetic_data[:, :, i].flatten()

        # Remove NaN/Inf
        real_flat = real_flat[np.isfinite(real_flat)]
        synth_flat = synth_flat[np.isfinite(synth_flat)]

        # Basic statistics
        result = {
            'Feature': name,
            'Real_Mean': np.mean(real_flat),
            'Synth_Mean': np.mean(synth_flat),
            'Real_Std': np.std(real_flat),
            'Synth_Std': np.std(synth_flat),
            'Real_Min': np.min(real_flat),
            'Synth_Min': np.min(synth_flat),
            'Real_Max': np.max(real_flat),
            'Synth_Max': np.max(synth_flat)
        }

        # K-S test
        if len(real_flat) > 0 and len(synth_flat) > 0:
            ks_stat, ks_pval = stats.ks_2samp(real_flat, synth_flat)
            result['KS_Statistic'] = ks_stat
            result['KS_PValue'] = ks_pval

        results.append(result)

    return pd.DataFrame(results)


def compute_temporal_statistics(data: np.ndarray) -> Dict:
    """
    Compute temporal statistics for order book sequences.

    Args:
        data: Order book data (n_days, seq_len, n_features)

    Returns:
        Dictionary with temporal statistics
    """
    # Compute first differences (returns/changes)
    diffs = np.diff(data, axis=1)

    # Autocorrelation at lag 1
    autocorr = []
    for i in range(data.shape[-1]):
        feature_data = data[:, :, i].flatten()
        if len(feature_data) > 1:
            ac = np.corrcoef(feature_data[:-1], feature_data[1:])[0, 1]
            autocorr.append(ac if np.isfinite(ac) else 0)
        else:
            autocorr.append(0)

    return {
        'mean_diff': np.nanmean(diffs),
        'std_diff': np.nanstd(diffs),
        'mean_autocorr': np.nanmean(autocorr),
        'min_autocorr': np.nanmin(autocorr),
        'max_autocorr': np.nanmax(autocorr)
    }


def generate_quality_report(synthetic_data: np.ndarray,
                            real_data: Optional[np.ndarray] = None,
                            feature_names: List[str] = None) -> str:
    """
    Generate a comprehensive quality assessment report.

    Args:
        synthetic_data: Generated order book data
        real_data: Optional real data for comparison
        feature_names: Names of features

    Returns:
        Formatted string report
    """
    report = []
    report.append("=" * 60)
    report.append("SYNTHETIC ORDER BOOK QUALITY ASSESSMENT REPORT")
    report.append("=" * 60)

    # Basic quality assessment
    quality = assess_quality(synthetic_data)

    report.append("\n1. DATA OVERVIEW")
    report.append("-" * 40)
    report.append(f"   Generated Days: {quality['n_days']}")
    report.append(f"   Total Observations: {quality['total_observations']}")

    report.append("\n2. QUALITY METRICS")
    report.append("-" * 40)
    report.append(f"   Overall Quality Score: {quality['quality_score']:.4f}")
    report.append(f"   Arbitrage Violations: {quality['total_arbitrage_violations']} "
                  f"({quality['arbitrage_violation_rate']*100:.2f}%)")
    report.append(f"   Negative Volumes: {quality['total_negative_volumes']} "
                  f"({quality['negative_volume_rate']*100:.2f}%)")
    report.append(f"   Price Order Violations: {quality['price_order_violations']}")

    # Temporal statistics
    temporal = compute_temporal_statistics(synthetic_data)
    report.append("\n3. TEMPORAL CHARACTERISTICS")
    report.append("-" * 40)
    report.append(f"   Mean Change: {temporal['mean_diff']:.6f}")
    report.append(f"   Change Volatility: {temporal['std_diff']:.6f}")
    report.append(f"   Avg Autocorrelation: {temporal['mean_autocorr']:.4f}")

    # Comparison with real data if provided
    if real_data is not None:
        report.append("\n4. REAL vs SYNTHETIC COMPARISON")
        report.append("-" * 40)

        comparison = compare_distributions(real_data, synthetic_data, feature_names)

        # Summary statistics
        mean_ks = comparison['KS_Statistic'].mean()
        report.append(f"   Mean K-S Statistic: {mean_ks:.4f}")
        report.append(f"   Features with p < 0.05: "
                      f"{(comparison['KS_PValue'] < 0.05).sum()}/{len(comparison)}")

        # Best and worst matching features
        comparison_sorted = comparison.sort_values('KS_Statistic')
        report.append(f"\n   Best Matching Features:")
        for _, row in comparison_sorted.head(3).iterrows():
            report.append(f"      - {row['Feature']}: KS={row['KS_Statistic']:.4f}")

        report.append(f"\n   Worst Matching Features:")
        for _, row in comparison_sorted.tail(3).iterrows():
            report.append(f"      - {row['Feature']}: KS={row['KS_Statistic']:.4f}")

    report.append("\n" + "=" * 60)

    return "\n".join(report)
