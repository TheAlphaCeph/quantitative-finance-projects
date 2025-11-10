"""
Data utilities for loading orderbook data from files
No proprietary data source dependencies - works with standard file formats
"""

import pandas as pd
import numpy as np
from typing import Optional


def load_nbbo_data(filepath: str) -> pd.DataFrame:
    """
    Load NBBO data from CSV or Parquet file

    Expected columns:
    - timestamp (datetime index)
    - best_bid_price, best_ask_price
    - best_bid_size, best_ask_size

    Parameters:
    -----------
    filepath : str
        Path to data file (.csv or .parquet)

    Returns:
    --------
    pd.DataFrame
        NBBO data with datetime index
    """
    if filepath.endswith('.parquet'):
        df = pd.read_parquet(filepath)
    else:
        df = pd.read_csv(filepath, parse_dates=['timestamp'])

    if 'timestamp' in df.columns:
        df.set_index('timestamp', inplace=True)

    required_cols = ['best_bid_price', 'best_ask_price', 'best_bid_size', 'best_ask_size']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def load_ofi_data(filepath: str) -> Optional[pd.Series]:
    """
    Load pre-computed OFI data from file

    Parameters:
    -----------
    filepath : str
        Path to OFI data file

    Returns:
    --------
    pd.Series or None
        OFI time series with datetime index
    """
    try:
        if filepath.endswith('.parquet'):
            df = pd.read_parquet(filepath)
        else:
            df = pd.read_csv(filepath, parse_dates=['timestamp'])

        if 'timestamp' in df.columns:
            df.set_index('timestamp', inplace=True)

        # Assume first column is OFI
        ofi = df.iloc[:, 0] if len(df.columns) > 0 else df['ofi']
        return ofi
    except Exception as e:
        print(f"Warning: Could not load OFI data: {e}")
        return None


def generate_synthetic_nbbo(
    n_seconds: int = 3600,
    start_price: float = 100.0,
    volatility: float = 0.02,
    spread_bps: float = 5.0,
    start_time: str = '2024-01-01 09:30:00',
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate synthetic NBBO data for testing and demonstration

    Parameters:
    -----------
    n_seconds : int
        Number of seconds to simulate
    start_price : float
        Starting mid price
    volatility : float
        Daily volatility (annualized)
    spread_bps : float
        Bid-ask spread in basis points
    start_time : str
        Start timestamp
    seed : int
        Random seed for reproducibility

    Returns:
    --------
    pd.DataFrame
        Synthetic NBBO data
    """
    np.random.seed(seed)

    # Generate timestamps
    timestamps = pd.date_range(start_time, periods=n_seconds, freq='1S')

    # Generate price process (GBM)
    dt = 1.0 / (252 * 6.5 * 3600)  # 1 second in trading year
    drift = 0.0  # No drift for realistic simulation
    returns = np.random.randn(n_seconds) * volatility * np.sqrt(dt)
    mid_prices = start_price * np.exp(np.cumsum(returns))

    # Add microstructure noise
    noise = np.random.randn(n_seconds) * (volatility * np.sqrt(dt) * 0.1)
    mid_prices += noise

    # Generate spread
    spread = mid_prices * (spread_bps / 10000)
    half_spread = spread / 2

    bid_prices = mid_prices - half_spread
    ask_prices = mid_prices + half_spread

    # Generate sizes (Poisson-like distribution)
    bid_sizes = np.random.poisson(500, n_seconds) + 100
    ask_sizes = np.random.poisson(500, n_seconds) + 100

    # Add some imbalance dynamics
    imbalance = np.cumsum(np.random.randn(n_seconds) * 0.01)
    bid_sizes = (bid_sizes * (1 + imbalance)).astype(int)
    ask_sizes = (ask_sizes * (1 - imbalance)).astype(int)
    bid_sizes = np.maximum(bid_sizes, 100)
    ask_sizes = np.maximum(ask_sizes, 100)

    return pd.DataFrame({
        'best_bid_price': bid_prices,
        'best_ask_price': ask_prices,
        'best_bid_size': bid_sizes,
        'best_ask_size': ask_sizes
    }, index=timestamps)


def generate_synthetic_ofi(
    nbbo_df: pd.DataFrame,
    intensity: float = 1.0,
    persistence: float = 0.95
) -> pd.Series:
    """
    Generate synthetic OFI from NBBO data

    Parameters:
    -----------
    nbbo_df : pd.DataFrame
        NBBO data
    intensity : float
        OFI intensity scaling factor
    persistence : float
        AR(1) persistence parameter

    Returns:
    --------
    pd.Series
        Synthetic OFI time series
    """
    n = len(nbbo_df)

    # Generate autocorrelated OFI process
    innovations = np.random.randn(n) * intensity
    ofi = np.zeros(n)
    ofi[0] = innovations[0]

    for i in range(1, n):
        ofi[i] = persistence * ofi[i-1] + innovations[i]

    # Add correlation with price changes
    mid_price = (nbbo_df['best_bid_price'] + nbbo_df['best_ask_price']) / 2
    price_changes = mid_price.diff().fillna(0)
    ofi += price_changes.values * 100  # Amplify correlation

    return pd.Series(ofi, index=nbbo_df.index, name='ofi')


def validate_nbbo_data(df: pd.DataFrame) -> bool:
    """
    Validate NBBO data format and consistency

    Parameters:
    -----------
    df : pd.DataFrame
        NBBO data to validate

    Returns:
    --------
    bool
        True if valid, raises ValueError otherwise
    """
    # Check required columns
    required = ['best_bid_price', 'best_ask_price', 'best_bid_size', 'best_ask_size']
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Check bid < ask
    if (df['best_bid_price'] >= df['best_ask_price']).any():
        raise ValueError("Bid price must be less than ask price")

    # Check positive sizes
    if (df['best_bid_size'] <= 0).any() or (df['best_ask_size'] <= 0).any():
        raise ValueError("Sizes must be positive")

    # Check for NaN
    if df.isnull().any().any():
        raise ValueError("Data contains NaN values")

    return True
