"""
Numba-optimized core functions for high-performance calculations
"""

import numpy as np
from numba import njit


@njit
def gamma_weight(distance: float, shape: float = 2.0) -> float:
    """
    Gamma-based weighting function for microprice deviation

    Parameters:
    -----------
    distance : float
        Normalized distance of microprice from mid (0 to 1)
    shape : float
        Shape parameter controlling decay speed

    Returns:
    --------
    float
        Weight in [0, 1]
    """
    # Use min/max instead of np.clip for scalar values in numba
    distance = max(0.0, min(distance, 1.0))
    x = shape * distance
    weight = np.exp(-x) * (1.0 + x)
    return max(0.0, min(weight, 1.0))


@njit
def compute_weighted_microprice(
    bid_prices: np.ndarray,
    bid_sizes: np.ndarray,
    ask_prices: np.ndarray,
    ask_sizes: np.ndarray,
    gamma_shape: float = 2.0
) -> tuple:
    """
    Compute gamma-weighted microprice for each snapshot

    Returns: (microprice, weighted_deviation, gamma_weights)
    """
    n = len(bid_prices)
    microprice = np.empty(n, dtype=np.float64)
    weighted_dev = np.empty(n, dtype=np.float64)
    weights = np.empty(n, dtype=np.float64)

    for i in range(n):
        bp, ap = bid_prices[i], ask_prices[i]
        bs, az = bid_sizes[i], ask_sizes[i]

        mid = 0.5 * (bp + ap)
        spread = ap - bp

        # Calculate gamma weights based on distance from mid
        if spread > 1e-9 and bs + az > 0:
            # Calculate preliminary distances
            dist_bid = 0.0  # Bid is at mid - 0.5*spread
            dist_ask = 1.0  # Ask is at mid + 0.5*spread

            # Apply gamma weighting to volumes
            bid_weight = gamma_weight(dist_bid, gamma_shape)
            ask_weight = gamma_weight(dist_ask, gamma_shape)

            # Gamma-weighted microprice
            weighted_bid_vol = bs * bid_weight
            weighted_ask_vol = az * ask_weight

            if weighted_bid_vol + weighted_ask_vol > 0:
                micro = (weighted_bid_vol * ap + weighted_ask_vol * bp) / (weighted_bid_vol + weighted_ask_vol)
            else:
                micro = mid

            weights[i] = 0.5 * (bid_weight + ask_weight)
        elif bs + az > 0:
            # Standard volume-weighted if spread too small
            micro = (bs * ap + az * bp) / (bs + az)
            weights[i] = 1.0
        else:
            micro = mid
            weights[i] = 1.0

        microprice[i] = micro
        deviation = micro - mid
        weighted_dev[i] = deviation * weights[i]

    return microprice, weighted_dev, weights


@njit
def integrate_ema(
    values: np.ndarray,
    timestamps: np.ndarray,
    half_life: float
) -> np.ndarray:
    """
    Exponential moving average integration with time-aware decay

    Parameters:
    -----------
    values : np.ndarray
        Signal values
    timestamps : np.ndarray
        Timestamps in nanoseconds
    half_life : float
        Half-life in seconds

    Returns:
    --------
    np.ndarray
        EMA-integrated values
    """
    n = len(values)
    if n == 0:
        return np.empty(0, dtype=np.float64)

    integrated = np.empty(n, dtype=np.float64)
    integrated[0] = values[0]
    decay_factor = np.log(2.0) / half_life

    for i in range(1, n):
        dt = (timestamps[i] - timestamps[i-1]) / 1e9
        if dt < 0:
            # Negative time delta indicates out-of-order data - treat as zero passage
            dt = 0.0
        elif dt > 3600:
            # Cap at 1 hour to prevent underflow
            dt = 3600.0

        decay = np.exp(-decay_factor * dt)
        integrated[i] = decay * integrated[i-1] + (1.0 - decay) * values[i]

    return integrated
