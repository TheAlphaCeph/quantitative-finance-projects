# GAN-Based Anomaly Detection for Limit Order Book Data

A deep learning framework for detecting anomalies in high-frequency limit order book (LOB) data using Generative Adversarial Networks (GANs).

## Overview

This project implements a GAN-based approach for unsupervised anomaly detection in financial order book data:

1. **Generator** learns to synthesize realistic order book sequences
2. **Discriminator** learns to distinguish real from synthetic data
3. **Anomaly Detection**: Real data scoring low on the discriminator represents market anomalies

The unsupervised approach requires no labeled anomaly data - the model learns what "normal" looks like.

### Securities Analyzed

| Ticker | Name | Type |
|--------|------|------|
| 0050 | Taiwan 50 ETF | Index ETF |
| 0056 | High Dividend ETF | Dividend ETF |
| 2330 | TSMC | Individual Stock |

**Training Period:** Q4 2023 (Oct-Dec)
**Testing Period:** Q1 2024 (Jan-Mar)

## Project Structure

```
GAN_OrderBook_Anomaly/
├── data/                       # Order book data (18 CSV.gz files)
├── models/                     # Trained model weights (.pth, .npy)
├── notebooks/
│   └── complete_analysis.ipynb # Full end-to-end analysis notebook
├── outputs/
│   ├── figures/               # Training curves, score distributions
│   └── results/               # K-S tests, quality metrics
├── src/                       # Source code modules
│   ├── models.py             # Generator & Discriminator architectures
│   ├── trainer.py            # GAN training loop
│   ├── detector.py           # Anomaly detection
│   ├── microstructure.py     # Market microstructure calculations
│   ├── synthetic.py          # Synthetic data generation
│   └── ...
└── requirements.txt
```

## Installation

```bash
# Clone the repository
git clone https://github.com/abhaykanwar/GAN_OrderBook_Anomaly.git
cd GAN_OrderBook_Anomaly

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the complete analysis notebook:

```bash
cd notebooks
jupyter notebook complete_analysis.ipynb
```

Or use the Python modules directly:

```python
from src import GANTrainer, AnomalyDetector, load_order_book_data

# Load and preprocess data
train_data = load_order_book_data('data/', '0050', train_months, columns)

# Train GAN
trainer = GANTrainer()
results = trainer.train(train_loader, val_loader, epochs=200)

# Detect anomalies
detector = AnomalyDetector(trainer.discriminator)
anomalies, normal = detector.detect(test_data, threshold=0.5)
```

## Model Architecture

### Generator
- 4 GRU layers with interleaved linear transformations
- LeakyReLU activations (0.01)
- Input: (batch, 265, 20) noise → Output: (batch, 265, 20) synthetic LOB

### Discriminator
- 4 GRU layers with dropout (0.15)
- Sigmoid output for real/fake classification
- Input: (batch, 265, 20) LOB → Output: (batch, 1) probability

### Training
- Generator LR: 0.00375, Discriminator LR: 0.001
- Adam optimizer (β₁=0.99, β₂=0.999)
- Gradient clipping: G=0.3, D=0.1
- Moment matching loss (mean, variance, skewness)
- Early stopping based on loss verge

## Key Results

### Training Performance
| Stock | Epochs | Final G Loss | Final D Loss |
|-------|--------|--------------|--------------|
| 0050 | 131 | 0.47 | 0.25 |
| 0056 | 54 | 0.85 | 0.26 |
| 2330 | 66 | 0.21 | 0.31 |

### Anomaly Detection
- Fixed threshold (0.5): 0% anomalies detected
- Indicates market stability between Q4 2023 and Q1 2024
- Percentile-based approach (bottom 10%) enables statistical comparison

### Statistical Analysis (K-S Tests)
Significant differences found in:
- Trade price returns (0050: p=0.0009)
- Order book pressure (2330: p=0.023)

### Synthetic Data Quality
| Metric | 0050 | 0056 | 2330 |
|--------|------|------|------|
| Arbitrage Violations | 32.75% | 22.64% | 12.75% |
| BV1 Ratio | 0.97 | 0.91 | 0.93 |

**Conclusion:** Synthetic data captures volume scales but exhibits critical no-arbitrage violations - not suitable for trading without post-processing.

## Data Format

5-level limit order book with 20 features per minutely snapshot:

| Feature | Description |
|---------|-------------|
| SP1-SP5 | Ask prices (levels 1-5) |
| BP1-BP5 | Bid prices (levels 1-5) |
| SV1-SV5 | Ask volumes (levels 1-5) |
| BV1-BV5 | Bid volumes (levels 1-5) |

**Temporal Resolution:** 265 minutely observations per trading day

## Requirements

- Python 3.8+
- PyTorch 2.0+
- pandas, numpy, scipy
- matplotlib, seaborn

## Author

**Abhay Kanwar**

## Acknowledgments

- Data: Taiwan Stock Exchange order book snapshots
