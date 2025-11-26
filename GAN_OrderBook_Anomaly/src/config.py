"""
Configuration and hyperparameters for GAN training and analysis.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import os


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    # Data paths
    data_dir: str = "./data"
    output_dir: str = "./outputs"
    models_dir: str = "./models"

    # Securities to analyze
    stocks: List[str] = field(default_factory=lambda: ["0050", "0056", "2330"])

    # Column names for order book data
    columns: List[str] = field(default_factory=lambda: [
        "date", "time", "lastPx", "size", "volume",
        "SP1", "BP1", "SV1", "BV1",
        "SP2", "BP2", "SV2", "BV2",
        "SP3", "BP3", "SV3", "BV3",
        "SP4", "BP4", "SV4", "BV4",
        "SP5", "BP5", "SV5", "BV5"
    ])

    # Training period (Q4 2023)
    train_months: List[str] = field(default_factory=lambda: ["202310", "202311", "202312"])

    # Test period (Q1 2024)
    test_months: List[str] = field(default_factory=lambda: ["202401", "202402", "202403"])

    # Minutes per trading day (Taiwan market: 9:00-13:30)
    minutes_per_day: int = 265

    # Number of order book levels
    n_levels: int = 5

    # Number of features (prices + volumes for each level)
    n_features: int = 20  # 5 levels * 4 (SP, BP, SV, BV)


@dataclass
class TrainingConfig:
    """Configuration for GAN training."""

    # Learning rates (asymmetric for stability)
    lr_generator: float = 0.00375  # Lower for generator
    lr_discriminator: float = 0.001  # Higher for discriminator

    # Optimizer parameters
    beta1: float = 0.99
    beta2: float = 0.999

    # Training parameters
    epochs: int = 200
    batch_size: int = 50

    # Gradient clipping (prevents exploding gradients in GRUs)
    clip_grad_generator: float = 0.3
    clip_grad_discriminator: float = 0.1

    # Train/validation split
    train_ratio: float = 0.8

    # Early stopping
    early_stopping: bool = True
    patience: int = 20  # Epochs to wait before stopping
    min_delta: float = 0.001  # Minimum improvement required

    # Moment matching (for generator loss)
    moment_order: int = 3  # Match up to 3rd order moments

    # Random seed for reproducibility
    seed: int = 307

    # Device
    device: str = "cuda"  # Will fallback to CPU if CUDA unavailable


@dataclass
class DetectionConfig:
    """Configuration for anomaly detection."""

    # Default threshold (discriminator score below this = abnormal)
    threshold: float = 0.5

    # Percentile-based threshold (alternative approach)
    percentile: float = 10  # Bottom 10% = relatively abnormal

    # Statistical tests
    significance_level: float = 0.05  # For K-S tests


@dataclass
class SyntheticConfig:
    """Configuration for synthetic data generation."""

    # Number of synthetic days to generate
    n_days: int = 5

    # Quality assessment
    assess_arbitrage: bool = True
    assess_volumes: bool = True
    assess_correlations: bool = True


def get_default_config():
    """Get default configuration."""
    return {
        "data": DataConfig(),
        "training": TrainingConfig(),
        "detection": DetectionConfig(),
        "synthetic": SyntheticConfig()
    }


def ensure_directories(config: DataConfig):
    """Create necessary directories if they don't exist."""
    dirs = [
        config.output_dir,
        config.models_dir,
        os.path.join(config.output_dir, "figures"),
        os.path.join(config.output_dir, "results")
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
