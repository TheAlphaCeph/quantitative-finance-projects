"""
GAN-based Order Book Anomaly Detection

A deep learning framework for detecting anomalies in limit order book data
using Generative Adversarial Networks (GANs) with GRU-based architectures.

Main Components:
    - models: Generator and Discriminator neural network architectures
    - trainer: GAN training with moment matching and early stopping
    - detector: Anomaly detection using trained discriminator
    - microstructure: Market microstructure variable calculations
    - synthetic: Synthetic order book generation and quality assessment
    - data_loader: Data loading and preprocessing utilities
    - config: Configuration classes for all components
    - utils: Visualization and helper functions

Example Usage:
    >>> from gan_orderbook import GANTrainer, AnomalyDetector
    >>> from gan_orderbook import load_order_book_data, create_data_loaders
    >>>
    >>> # Load and preprocess data
    >>> data = load_order_book_data('data/', '0050', months, columns)
    >>> train_loader, val_loader = create_data_loaders(data['normalized'])
    >>>
    >>> # Train GAN
    >>> trainer = GANTrainer()
    >>> results = trainer.train(train_loader, val_loader, epochs=200)
    >>>
    >>> # Detect anomalies
    >>> detector = AnomalyDetector(trainer.discriminator)
    >>> abnormal_idx, normal_idx = detector.detect(test_data, threshold=0.5)

Author: Abhay Kanwar
"""

__version__ = "1.0.0"
__author__ = "Abhay Kanwar"

# Core models
from .models import (
    Generator,
    Discriminator,
    create_gan,
    count_parameters,
    save_models,
    load_models
)

# Training
from .trainer import (
    GANTrainer,
    set_seed,
    get_loss_verge,
    compute_moment_loss,
    compute_derivative_loss
)

# Anomaly Detection
from .detector import (
    AnomalyDetector,
    analyze_detection_results
)

# Microstructure Analysis
from .microstructure import (
    compute_trade_returns,
    compute_mid_returns,
    compute_trade_size,
    compute_spread,
    compute_spread_diff,
    compute_ob_pressure_1,
    compute_ob_pressure_5,
    compute_daily_statistics,
    compute_all_variables,
    run_ks_tests,
    compare_statistics
)

# Synthetic Data Generation
from .synthetic import (
    SyntheticGenerator,
    assess_quality,
    compare_distributions,
    compute_temporal_statistics,
    generate_quality_report
)

# Data Loading
from .data_loader import (
    OrderBookDataset,
    load_raw_data,
    prepare_minutely_data,
    create_daily_sequences,
    normalize_data,
    load_order_book_data,
    create_data_loaders
)

# Configuration
from .config import (
    DataConfig,
    TrainingConfig,
    DetectionConfig
)

# Utilities
from .utils import (
    setup_plotting_style,
    plot_training_losses,
    plot_score_distribution,
    plot_anomaly_comparison,
    plot_order_book_snapshot,
    format_results_table,
    save_results,
    load_results,
    print_summary,
    calculate_detection_metrics,
    get_feature_names,
    validate_order_book,
    ProgressLogger
)

__all__ = [
    # Models
    'Generator',
    'Discriminator',
    'create_gan',
    'count_parameters',
    'save_models',
    'load_models',

    # Training
    'GANTrainer',
    'set_seed',
    'get_loss_verge',
    'compute_moment_loss',
    'compute_derivative_loss',

    # Detection
    'AnomalyDetector',
    'analyze_detection_results',

    # Microstructure
    'compute_trade_returns',
    'compute_mid_returns',
    'compute_trade_size',
    'compute_spread',
    'compute_spread_diff',
    'compute_ob_pressure_1',
    'compute_ob_pressure_5',
    'compute_daily_statistics',
    'compute_all_variables',
    'run_ks_tests',
    'compare_statistics',

    # Synthetic
    'SyntheticGenerator',
    'assess_quality',
    'compare_distributions',
    'compute_temporal_statistics',
    'generate_quality_report',

    # Data
    'OrderBookDataset',
    'load_raw_data',
    'prepare_minutely_data',
    'create_daily_sequences',
    'normalize_data',
    'load_order_book_data',
    'create_data_loaders',

    # Config
    'DataConfig',
    'TrainingConfig',
    'DetectionConfig',

    # Utils
    'setup_plotting_style',
    'plot_training_losses',
    'plot_score_distribution',
    'plot_anomaly_comparison',
    'plot_order_book_snapshot',
    'format_results_table',
    'save_results',
    'load_results',
    'print_summary',
    'calculate_detection_metrics',
    'get_feature_names',
    'validate_order_book',
    'ProgressLogger',
]
