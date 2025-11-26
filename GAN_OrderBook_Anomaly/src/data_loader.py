"""
Data loading and preprocessing for order book data.

This module handles loading raw order book CSV files, preprocessing
(cleaning, normalization, resampling), and creating PyTorch datasets.
"""

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Dict
import warnings

from .config import DataConfig


class OrderBookDataset(Dataset):
    """
    PyTorch Dataset for order book sequences.

    Each sample is a complete trading day (265 minutely observations)
    with 20 features (5-level prices and volumes).

    Attributes:
        data: Tensor of shape (n_days, 265, 20)
        normalized: Whether data is normalized
    """

    def __init__(self, data: np.ndarray):
        """
        Initialize dataset.

        Args:
            data: Order book data of shape (n_days, seq_len, n_features)
        """
        self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


def load_raw_data(data_dir: str, stock: str, months: List[str],
                  columns: List[str]) -> pd.DataFrame:
    """
    Load raw order book data from CSV files.

    Args:
        data_dir: Directory containing data files
        stock: Stock ticker (e.g., "0050")
        months: List of month strings (e.g., ["202310", "202311"])
        columns: Column names to use

    Returns:
        DataFrame with concatenated data from all months
    """
    dfs = []

    for month in months:
        # Construct filename: {stock}_md_{month}_{month}.csv.gz
        year_month = month[:6]
        filepath = os.path.join(data_dir, f"{stock}_md_{year_month}_{year_month}.csv.gz")

        if os.path.exists(filepath):
            df = pd.read_csv(filepath, compression='gzip', usecols=columns)
            dfs.append(df)
            print(f"  Loaded {os.path.basename(filepath)}: {len(df)} rows")
        else:
            warnings.warn(f"File not found: {filepath}")

    if not dfs:
        raise FileNotFoundError(f"No data files found for {stock}")

    return pd.concat(dfs, ignore_index=True)


def prepare_minutely_data(df: pd.DataFrame, trading_days: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Prepare minutely order book snapshots from raw tick data.

    Processing steps:
    1. Clean invalid observations (negative prices, zero spreads)
    2. Scale prices (divide by 100) and volumes (multiply by 1000)
    3. Forward-fill missing values
    4. Resample to minutely frequency

    Args:
        df: Raw order book DataFrame
        trading_days: Optional list of valid trading days

    Returns:
        Minutely aggregated DataFrame
    """
    if df.empty:
        return None

    df = df.copy()

    # Calculate cumulative value for forward-filling
    df['bfValue'] = df['lastPx'] * df['size']
    df['bfValue'] = df['bfValue'].ffill()
    df['cumValue'] = df.groupby('date')['bfValue'].cumsum()

    # Filter invalid observations
    df = df[df['SP1'] > 0]  # Valid ask price
    df = df[df['BP1'] > 0]  # Valid bid price
    df = df[df['SP1'] - df['BP1'] > 0]  # Positive spread (no-arbitrage)

    # Scale prices and volumes
    for i in range(1, 6):
        df[f'SP{i}'] = df[f'SP{i}'] / 100  # Prices in TWD
        df[f'BP{i}'] = df[f'BP{i}'] / 100
        df[f'SV{i}'] = df[f'SV{i}'] * 1000  # Volumes in shares
        df[f'BV{i}'] = df[f'BV{i}'] * 1000

    df['lastPx'] = df['lastPx'] / 100
    df['size'] = df['size'] * 1000
    df['volume'] = df['volume'] * 1000

    # Forward-fill prices, zero-fill sizes
    df['lastPx'] = df.groupby('date')['lastPx'].ffill()
    df['size'] = df.groupby('date')['size'].transform(lambda x: x.fillna(0))

    # Clean up temporary columns
    df = df.drop(columns=['bfValue', 'cumValue'], errors='ignore')

    # Create datetime index
    df['dt_index'] = pd.to_datetime(
        df['date'].astype(str) + ' ' + df['time'].astype(str),
        format="%Y-%m-%d %H%M%S%f"
    )

    # Remove duplicates (keep last observation per timestamp)
    df = df[~df['dt_index'].duplicated(keep='last')]

    # Resample to minutely frequency
    bin_size = '1min'
    agg_dict = {col: 'last' for col in df.columns if col not in ['dt_index', 'date', 'time']}

    df_minutely = df.groupby(
        pd.Grouper(key='dt_index', freq=bin_size, closed='right', label='right')
    ).agg(agg_dict)

    # Reset index and restore date column
    df_minutely = df_minutely.reset_index()
    df_minutely['date'] = df_minutely['dt_index'].dt.date

    # Forward-fill any remaining NaNs
    df_minutely = df_minutely.ffill()

    return df_minutely


def create_daily_sequences(minutely_data: pd.DataFrame,
                           minutes_per_day: int = 265) -> np.ndarray:
    """
    Convert minutely data to daily sequences.

    Only keeps complete trading days (exactly 265 minutes).

    Args:
        minutely_data: Minutely DataFrame
        minutes_per_day: Expected observations per day

    Returns:
        Array of shape (n_days, minutes_per_day, n_features)
    """
    sequences = []

    for date, day_data in minutely_data.groupby('date'):
        if len(day_data) == minutes_per_day:
            # Extract features (columns 5: onwards are order book features)
            features = day_data.values
            sequences.append(features)

    if not sequences:
        raise ValueError("No complete trading days found")

    return np.array(sequences)


def normalize_data(data: np.ndarray,
                   feature_start: int = 5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Normalize order book features.

    Processing:
    1. Log-transform volumes (last 10 features)
    2. Z-score normalization per day

    Args:
        data: Array of shape (n_days, seq_len, n_features)
        feature_start: Index where order book features start

    Returns:
        Tuple of (normalized_data, means, stds)
    """
    # Extract order book features
    X = data[:, :, feature_start:].astype(float)

    # Log-transform volumes (last 10 columns: SV1-5, BV1-5)
    X[:, :, -10:] = np.log(1 + X[:, :, -10:])

    # Calculate per-day statistics
    X_mean = X.mean(axis=1)  # Shape: (n_days, n_features)
    X_std = X.std(axis=1)    # Shape: (n_days, n_features)

    # Z-score normalization (scale by 2*std for more stable training)
    X_normalized = np.zeros_like(X)
    for i in range(len(X)):
        X_normalized[i] = (X[i] - X_mean[i]) / (2 * X_std[i] + 1e-10)

    # Handle any remaining NaN/Inf
    X_normalized = np.nan_to_num(X_normalized, nan=0, posinf=0, neginf=0)

    return X_normalized, X_mean, X_std


def load_order_book_data(data_dir: str, stock: str,
                         months: List[str], columns: List[str],
                         normalize: bool = True) -> Dict:
    """
    Complete pipeline to load and preprocess order book data.

    Args:
        data_dir: Data directory
        stock: Stock ticker
        months: List of months to load
        columns: Column names
        normalize: Whether to normalize features

    Returns:
        Dictionary with processed data and metadata
    """
    print(f"\nLoading data for {stock}...")

    # Load raw data
    df = load_raw_data(data_dir, stock, months, columns)

    # Prepare minutely data
    minutely = prepare_minutely_data(df)

    # Create daily sequences
    sequences = create_daily_sequences(minutely)
    print(f"  Created {len(sequences)} complete trading days")

    result = {
        'raw_sequences': sequences,
        'stock': stock,
        'n_days': len(sequences)
    }

    # Normalize if requested
    if normalize:
        X_norm, X_mean, X_std = normalize_data(sequences)
        result['normalized'] = X_norm
        result['mean'] = X_mean
        result['std'] = X_std

    return result


def create_data_loaders(data: np.ndarray, batch_size: int = 50,
                        train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation data loaders.

    Args:
        data: Normalized data array
        batch_size: Batch size
        train_ratio: Fraction of data for training

    Returns:
        Tuple of (train_loader, val_loader)
    """
    dataset = OrderBookDataset(data)

    # Split into train/validation
    n_train = int(len(dataset) * train_ratio)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val]
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
