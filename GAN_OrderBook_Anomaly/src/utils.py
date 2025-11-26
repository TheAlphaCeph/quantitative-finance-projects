"""
Utility functions for GAN-based order book anomaly detection.

This module provides helper functions for visualization, logging,
evaluation metrics, and other common operations.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union
import os
import json
from datetime import datetime


def setup_plotting_style():
    """Configure matplotlib for publication-quality figures."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 100,
        'savefig.dpi': 150,
        'savefig.bbox': 'tight'
    })


def plot_training_losses(train_g_losses: List[float],
                         train_d_losses: List[float],
                         val_g_losses: Optional[List[float]] = None,
                         val_d_losses: Optional[List[float]] = None,
                         title: str = "GAN Training Progress",
                         save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot GAN training losses over time.

    Args:
        train_g_losses: Generator training losses
        train_d_losses: Discriminator training losses
        val_g_losses: Optional validation generator losses
        val_d_losses: Optional validation discriminator losses
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Generator loss
    axes[0].plot(train_g_losses, label='Train G Loss', alpha=0.7)
    if val_g_losses:
        epochs = np.linspace(0, len(train_g_losses), len(val_g_losses))
        axes[0].plot(epochs, val_g_losses, label='Val G Loss', linewidth=2)
    axes[0].set_xlabel('Step')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Generator Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Discriminator loss
    axes[1].plot(train_d_losses, label='Train D Loss', alpha=0.7, color='orange')
    if val_d_losses:
        epochs = np.linspace(0, len(train_d_losses), len(val_d_losses))
        axes[1].plot(epochs, val_d_losses, label='Val D Loss', linewidth=2, color='red')
    axes[1].set_xlabel('Step')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Discriminator Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_score_distribution(scores: np.ndarray,
                            threshold: float = 0.5,
                            title: str = "Discriminator Score Distribution",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Plot distribution of discriminator scores.

    Args:
        scores: Discriminator scores for each day
        threshold: Classification threshold
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram
    ax.hist(scores, bins=30, edgecolor='black', alpha=0.7, density=True)

    # Add threshold line
    ax.axvline(x=threshold, color='red', linestyle='--',
               label=f'Threshold = {threshold}', linewidth=2)

    # Add percentile lines
    p10 = np.percentile(scores, 10)
    p90 = np.percentile(scores, 90)
    ax.axvline(x=p10, color='orange', linestyle=':',
               label=f'10th percentile = {p10:.3f}')
    ax.axvline(x=p90, color='green', linestyle=':',
               label=f'90th percentile = {p90:.3f}')

    ax.set_xlabel('Discriminator Score')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_anomaly_comparison(abnormal_stats: pd.DataFrame,
                            normal_stats: pd.DataFrame,
                            variables: List[str] = None,
                            title: str = "Abnormal vs Normal Days Comparison",
                            save_path: Optional[str] = None) -> plt.Figure:
    """
    Create comparison plot of microstructure variables.

    Args:
        abnormal_stats: Statistics for abnormal days
        normal_stats: Statistics for normal days
        variables: List of variables to plot
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if variables is None:
        variables = [
            'trade_returns_mean', 'mid_returns_mean', 'trade_size_mean',
            'spread_mean', 'spread_diff_mean', 'pressure_1_mean', 'pressure_5_mean'
        ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()

    for i, var in enumerate(variables):
        if i >= len(axes):
            break

        ax = axes[i]

        # Get data
        norm_data = normal_stats[var].dropna()
        abn_data = abnormal_stats[var].dropna()

        # Create box plot
        data = [norm_data, abn_data]
        bp = ax.boxplot(data, labels=['Normal', 'Abnormal'], patch_artist=True)

        # Color boxes
        bp['boxes'][0].set_facecolor('lightblue')
        bp['boxes'][1].set_facecolor('lightcoral')

        ax.set_title(var.replace('_', ' ').title())
        ax.grid(True, alpha=0.3)

    # Hide unused subplot
    if len(variables) < len(axes):
        axes[-1].set_visible(False)

    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path)

    return fig


def plot_order_book_snapshot(data: np.ndarray,
                             timestep: int = 0,
                             n_levels: int = 5,
                             title: str = "Order Book Snapshot",
                             save_path: Optional[str] = None) -> plt.Figure:
    """
    Visualize order book at a specific timestep.

    Args:
        data: Order book data (seq_len, n_features) or single row
        timestep: Timestep to visualize
        n_levels: Number of price levels
        title: Plot title
        save_path: Optional path to save figure

    Returns:
        Matplotlib figure
    """
    if data.ndim == 2:
        snapshot = data[timestep]
    else:
        snapshot = data

    fig, ax = plt.subplots(figsize=(10, 6))

    # Extract prices and volumes (assuming standard ordering)
    # SP1, BP1, SV1, BV1, SP2, BP2, ...
    ask_prices = snapshot[0::4][:n_levels]
    bid_prices = snapshot[1::4][:n_levels]
    ask_volumes = snapshot[2::4][:n_levels]
    bid_volumes = snapshot[3::4][:n_levels]

    # Create bar chart
    x = np.arange(n_levels)
    width = 0.35

    ax.bar(x - width/2, ask_volumes, width, label='Ask Volume', color='red', alpha=0.7)
    ax.bar(x + width/2, bid_volumes, width, label='Bid Volume', color='green', alpha=0.7)

    ax.set_xlabel('Price Level')
    ax.set_ylabel('Volume')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels([f'L{i+1}\n({bid_prices[i]:.2f}/{ask_prices[i]:.2f})' for i in range(n_levels)])
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path)

    return fig


def format_results_table(results: pd.DataFrame,
                         float_format: str = '.4f') -> str:
    """
    Format results DataFrame as publication-ready table.

    Args:
        results: Results DataFrame
        float_format: Format string for floats

    Returns:
        Formatted string representation
    """
    # Create formatted copy
    formatted = results.copy()

    for col in formatted.select_dtypes(include=[np.number]).columns:
        formatted[col] = formatted[col].apply(lambda x: f'{x:{float_format}}' if pd.notna(x) else '')

    return formatted.to_string(index=False)


def save_results(results: Dict,
                 output_dir: str,
                 prefix: str = "results") -> None:
    """
    Save results to JSON file.

    Args:
        results: Dictionary of results
        output_dir: Output directory
        prefix: Filename prefix
    """
    os.makedirs(output_dir, exist_ok=True)

    # Convert numpy types to Python types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    results_converted = convert(results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"{prefix}_{timestamp}.json")

    with open(filepath, 'w') as f:
        json.dump(results_converted, f, indent=2)

    print(f"Results saved to {filepath}")


def load_results(filepath: str) -> Dict:
    """
    Load results from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def print_summary(title: str, metrics: Dict, width: int = 60) -> None:
    """
    Print formatted summary of metrics.

    Args:
        title: Summary title
        metrics: Dictionary of metric names and values
        width: Total width of output
    """
    print("=" * width)
    print(f"{title:^{width}}")
    print("=" * width)

    for name, value in metrics.items():
        if isinstance(value, float):
            print(f"  {name}: {value:.4f}")
        else:
            print(f"  {name}: {value}")

    print("=" * width)


def calculate_detection_metrics(y_true: np.ndarray,
                                y_pred: np.ndarray) -> Dict:
    """
    Calculate classification metrics for anomaly detection.

    Args:
        y_true: True labels (1 = anomaly)
        y_pred: Predicted labels

    Returns:
        Dictionary with precision, recall, F1, etc.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy
    }


def get_feature_names(n_levels: int = 5) -> List[str]:
    """
    Generate standard feature names for n-level order book.

    Args:
        n_levels: Number of price levels

    Returns:
        List of feature names
    """
    names = []
    for i in range(1, n_levels + 1):
        names.extend([f'SP{i}', f'BP{i}', f'SV{i}', f'BV{i}'])
    return names


def validate_order_book(data: np.ndarray,
                        n_levels: int = 5) -> Tuple[bool, List[str]]:
    """
    Validate order book data for common issues.

    Args:
        data: Order book data
        n_levels: Number of price levels

    Returns:
        Tuple of (is_valid, list of issues)
    """
    issues = []

    # Check for NaN values
    nan_count = np.isnan(data).sum()
    if nan_count > 0:
        issues.append(f"Contains {nan_count} NaN values")

    # Check for negative prices
    # Assuming SP columns are at indices 0, 4, 8, ... and BP at 1, 5, 9, ...
    for i in range(n_levels):
        sp_idx = i * 4
        bp_idx = i * 4 + 1

        if (data[..., sp_idx] < 0).any():
            issues.append(f"Negative ask prices at level {i+1}")
        if (data[..., bp_idx] < 0).any():
            issues.append(f"Negative bid prices at level {i+1}")

    # Check for crossed quotes (ask <= bid)
    for i in range(n_levels):
        sp_idx = i * 4
        bp_idx = i * 4 + 1
        if (data[..., sp_idx] <= data[..., bp_idx]).any():
            issues.append(f"Crossed quotes at level {i+1}")

    # Check for negative volumes
    for i in range(n_levels):
        sv_idx = i * 4 + 2
        bv_idx = i * 4 + 3

        if (data[..., sv_idx] < 0).any():
            issues.append(f"Negative ask volumes at level {i+1}")
        if (data[..., bv_idx] < 0).any():
            issues.append(f"Negative bid volumes at level {i+1}")

    return len(issues) == 0, issues


class ProgressLogger:
    """Simple progress logger for training loops."""

    def __init__(self, total: int, prefix: str = "Progress"):
        """
        Initialize logger.

        Args:
            total: Total number of iterations
            prefix: Prefix for progress messages
        """
        self.total = total
        self.prefix = prefix
        self.current = 0
        self.start_time = datetime.now()

    def update(self, step: int = 1, message: str = "") -> None:
        """
        Update progress.

        Args:
            step: Number of steps completed
            message: Optional message to display
        """
        self.current += step
        elapsed = (datetime.now() - self.start_time).total_seconds()
        rate = self.current / elapsed if elapsed > 0 else 0
        eta = (self.total - self.current) / rate if rate > 0 else 0

        progress = self.current / self.total * 100
        print(f"\r{self.prefix}: {progress:.1f}% [{self.current}/{self.total}] "
              f"ETA: {eta:.0f}s {message}", end="")

        if self.current >= self.total:
            print()  # New line at completion

    def reset(self) -> None:
        """Reset the logger."""
        self.current = 0
        self.start_time = datetime.now()
