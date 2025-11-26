"""
Anomaly detection using trained discriminator.

This module implements anomaly detection by scoring real data with
the trained discriminator. Low scores indicate data that differs
from the learned "normal" patterns.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from typing import List, Tuple, Optional, Dict
import pandas as pd

from .models import Discriminator
from .data_loader import OrderBookDataset
from .config import DetectionConfig


class AnomalyDetector:
    """
    Anomaly detector using trained GAN discriminator.

    The discriminator, trained to distinguish real from synthetic data,
    can be repurposed for anomaly detection: real samples that receive
    low scores (classified as "fake") may represent genuine anomalies.

    Example:
        >>> detector = AnomalyDetector(discriminator)
        >>> scores = detector.get_scores(test_data)
        >>> anomalies = detector.detect(test_data, threshold=0.5)
    """

    def __init__(self, discriminator: Discriminator,
                 config: Optional[DetectionConfig] = None,
                 device: str = "cpu"):
        """
        Initialize detector.

        Args:
            discriminator: Trained discriminator network
            config: Detection configuration
            device: Device to use (cpu/cuda)
        """
        self.discriminator = discriminator
        self.config = config or DetectionConfig()
        self.device = torch.device(device)
        self.discriminator.to(self.device)
        self.discriminator.eval()

    def get_scores(self, data: np.ndarray,
                   X_mean: Optional[np.ndarray] = None,
                   X_std: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Get discriminator scores for all samples.

        Args:
            data: Order book data (n_days, seq_len, n_features)
            X_mean: Mean for normalization (from training)
            X_std: Std for normalization (from training)

        Returns:
            Array of scores (n_days,), range [0, 1]
        """
        # Normalize if stats provided
        if X_mean is not None and X_std is not None:
            data = self._normalize(data, X_mean, X_std)

        # Create dataset and loader
        dataset = OrderBookDataset(data)
        loader = DataLoader(dataset, batch_size=1, shuffle=False)

        scores = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                score = self.discriminator(batch)
                scores.append(score.item())

        return np.array(scores)

    def detect(self, data: np.ndarray,
               threshold: Optional[float] = None,
               X_mean: Optional[np.ndarray] = None,
               X_std: Optional[np.ndarray] = None) -> Tuple[List[int], List[int]]:
        """
        Detect anomalies using fixed threshold.

        Args:
            data: Order book data
            threshold: Score threshold (below = abnormal)
            X_mean: Normalization mean
            X_std: Normalization std

        Returns:
            Tuple of (abnormal_indices, normal_indices)
        """
        threshold = threshold or self.config.threshold
        scores = self.get_scores(data, X_mean, X_std)

        abnormal_idx = np.where(scores <= threshold)[0].tolist()
        normal_idx = np.where(scores > threshold)[0].tolist()

        return abnormal_idx, normal_idx

    def detect_percentile(self, data: np.ndarray,
                          percentile: Optional[float] = None,
                          X_mean: Optional[np.ndarray] = None,
                          X_std: Optional[np.ndarray] = None) -> Tuple[List[int], List[int]]:
        """
        Detect anomalies using percentile-based threshold.

        This approach identifies the bottom X% of scores as "relatively abnormal",
        useful when the fixed threshold classifies all data as normal.

        Args:
            data: Order book data
            percentile: Bottom percentile to classify as abnormal (default: 10)
            X_mean: Normalization mean
            X_std: Normalization std

        Returns:
            Tuple of (abnormal_indices, normal_indices)
        """
        percentile = percentile or self.config.percentile
        scores = self.get_scores(data, X_mean, X_std)

        # Calculate threshold from percentile
        threshold = np.percentile(scores, percentile)

        abnormal_idx = np.where(scores <= threshold)[0].tolist()
        normal_idx = np.where(scores > threshold)[0].tolist()

        return abnormal_idx, normal_idx

    def get_statistics(self, data: np.ndarray,
                       X_mean: Optional[np.ndarray] = None,
                       X_std: Optional[np.ndarray] = None) -> Dict:
        """
        Get comprehensive score statistics.

        Args:
            data: Order book data
            X_mean: Normalization mean
            X_std: Normalization std

        Returns:
            Dictionary with score statistics
        """
        scores = self.get_scores(data, X_mean, X_std)

        return {
            'n_days': len(scores),
            'mean': float(scores.mean()),
            'std': float(scores.std()),
            'min': float(scores.min()),
            'max': float(scores.max()),
            'percentile_10': float(np.percentile(scores, 10)),
            'percentile_25': float(np.percentile(scores, 25)),
            'median': float(np.percentile(scores, 50)),
            'percentile_75': float(np.percentile(scores, 75)),
            'percentile_90': float(np.percentile(scores, 90)),
            'above_0.5': int((scores > 0.5).sum()),
            'below_0.5': int((scores <= 0.5).sum())
        }

    def _normalize(self, data: np.ndarray,
                   X_mean: np.ndarray, X_std: np.ndarray) -> np.ndarray:
        """
        Normalize data using training statistics.

        Uses average statistics across training days for consistent
        normalization of test data.
        """
        data = data.copy().astype(float)

        # Log transform volumes (last 10 features)
        data[:, :, -10:] = np.log(1 + data[:, :, -10:])

        # Use average across training days
        mean_avg = X_mean.mean(axis=0)
        std_avg = X_std.mean(axis=0)

        # Normalize each day
        for i in range(len(data)):
            data[i] = (data[i] - mean_avg) / (2 * std_avg + 1e-10)

        data = np.nan_to_num(data, nan=0, posinf=0, neginf=0)

        return data

    @classmethod
    def load(cls, path: str, stock: str, device: str = "cpu") -> 'AnomalyDetector':
        """
        Load detector from saved discriminator.

        Args:
            path: Directory containing saved model
            stock: Stock ticker

        Returns:
            Initialized AnomalyDetector
        """
        discriminator = torch.load(
            f"{path}/{stock}_discriminator.pth",
            weights_only=False
        )
        return cls(discriminator, device=device)


def analyze_detection_results(abnormal_idx: List[int], normal_idx: List[int],
                              scores: np.ndarray, threshold: float) -> pd.DataFrame:
    """
    Create summary DataFrame of detection results.

    Args:
        abnormal_idx: Indices of abnormal days
        normal_idx: Indices of normal days
        scores: All discriminator scores
        threshold: Threshold used for classification

    Returns:
        Summary DataFrame
    """
    total = len(scores)
    n_abnormal = len(abnormal_idx)
    n_normal = len(normal_idx)

    summary = pd.DataFrame([{
        'Total Days': total,
        'Abnormal Days': n_abnormal,
        'Normal Days': n_normal,
        'Abnormal %': n_abnormal / total * 100,
        'Threshold': threshold,
        'Mean Score': scores.mean(),
        'Std Score': scores.std(),
        'Min Score': scores.min(),
        'Max Score': scores.max()
    }])

    return summary
