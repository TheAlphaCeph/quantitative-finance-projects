"""
Kalman Filter for Dynamic Hedge Ratios
======================================

Adaptive hedge ratio estimation using Kalman filtering.
Much more responsive than rolling OLS regression.

Author: Abhay Kanwar
Date: 2025-11-03
"""

import numpy as np
from pykalman import KalmanFilter


class KalmanHedgeRatio:
    """
    Kalman Filter for estimating time-varying hedge ratios in pair trading.

    The state is [beta, alpha] where:
    - beta: hedge ratio (slope)
    - alpha: intercept

    Observation model: price_base = beta * price_quote + alpha + noise
    """

    def __init__(self, delta=1e-4, vw=1e-3):
        """
        Initialize Kalman Filter for hedge ratio estimation.

        Parameters
        ----------
        delta : float
            Process noise (how much beta can change per step)
            Higher = more adaptive, lower = more stable
        vw : float
            Observation noise (measurement uncertainty)
        """
        self.delta = delta
        self.vw = vw

        # State: [beta, alpha]
        self.state_mean = np.array([1.0, 0.0])
        self.state_cov = np.eye(2)

        # Transition: random walk (beta and alpha can drift)
        self.transition_matrix = np.eye(2)
        self.transition_cov = delta * np.eye(2)

    def update(self, price_base, price_quote):
        """
        Update hedge ratio with new price observation.

        Parameters
        ----------
        price_base : float
            Base asset price (e.g., BTC)
        price_quote : float
            Quote asset price (e.g., ETH)

        Returns
        -------
        beta : float
            Current hedge ratio estimate
        """
        # Observation matrix: [price_quote, 1] for y = beta*x + alpha
        obs_matrix = np.array([[price_quote, 1.0]])

        # Predict step
        state_mean_pred = self.transition_matrix @ self.state_mean
        state_cov_pred = (
            self.transition_matrix @ self.state_cov @ self.transition_matrix.T
            + self.transition_cov
        )

        # Update step
        obs_cov = obs_matrix @ state_cov_pred @ obs_matrix.T + self.vw
        kalman_gain = state_cov_pred @ obs_matrix.T / obs_cov

        innovation = price_base - (obs_matrix @ state_mean_pred)[0]
        self.state_mean = state_mean_pred + (kalman_gain * innovation).flatten()
        self.state_cov = (
            state_cov_pred - kalman_gain @ obs_matrix @ state_cov_pred
        )

        beta = self.state_mean[0]
        return beta

    def get_spread(self, price_base, price_quote):
        """
        Calculate spread with dynamically updated hedge ratio.

        Parameters
        ----------
        price_base : float
            Base asset price
        price_quote : float
            Quote asset price

        Returns
        -------
        spread : float
            Log spread
        beta : float
            Current hedge ratio
        """
        beta = self.update(price_base, price_quote)

        # Log spread
        spread = np.log(price_base) - beta * np.log(price_quote)

        return spread, beta

    def reset(self):
        """Reset filter to initial state."""
        self.state_mean = np.array([1.0, 0.0])
        self.state_cov = np.eye(2)


class KalmanHedgeRatioVectorized:
    """
    Vectorized version for batch processing of price series.
    Uses simple online update for better compatibility.
    """

    def __init__(self, delta=1e-4, vw=1e-3):
        """
        Initialize Kalman Filter.

        Parameters
        ----------
        delta : float
            Process noise
        vw : float
            Observation noise
        """
        self.delta = delta
        self.vw = vw

    def fit(self, prices_base, prices_quote):
        """
        Fit Kalman filter to entire price series using online updates.

        Parameters
        ----------
        prices_base : array-like
            Base asset prices
        prices_quote : array-like
            Quote asset prices

        Returns
        -------
        betas : np.ndarray
            Time series of hedge ratios
        spreads : np.ndarray
            Time series of log spreads
        """
        prices_base = np.asarray(prices_base)
        prices_quote = np.asarray(prices_quote)

        n = len(prices_base)

        # Initialize state
        state_mean = np.array([1.0, 0.0])  # [beta, alpha]
        state_cov = np.eye(2)

        # Transition matrix and covariance
        F = np.eye(2)
        Q = self.delta * np.eye(2)

        betas = np.zeros(n)

        # Online Kalman filter updates
        for i in range(n):
            # Predict
            state_mean = F @ state_mean
            state_cov = F @ state_cov @ F.T + Q

            # Observation matrix for this step
            H = np.array([[prices_quote[i], 1.0]])

            # Update
            y = prices_base[i] - (H @ state_mean)[0]  # Innovation
            S = H @ state_cov @ H.T + self.vw  # Innovation covariance
            K = state_cov @ H.T / S  # Kalman gain

            state_mean = state_mean + (K * y).flatten()
            state_cov = state_cov - K @ H @ state_cov

            betas[i] = state_mean[0]

        # Calculate spreads
        spreads = np.log(prices_base) - betas * np.log(prices_quote)

        return betas, spreads


def test_kalman_vs_ols():
    """
    Test Kalman Filter vs Rolling OLS on synthetic data.
    """
    import pandas as pd
    from sklearn.linear_model import LinearRegression

    # Generate synthetic pair trading data
    np.random.seed(42)
    n = 1000

    # True beta varies over time
    true_beta = 1.0 + 0.3 * np.sin(np.arange(n) / 100)

    price_quote = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.01))
    price_base = true_beta * price_quote + np.random.randn(n) * 5

    # Kalman Filter
    kalman = KalmanHedgeRatioVectorized(delta=1e-3, vw=5.0)
    betas_kalman, spreads_kalman = kalman.fit(price_base, price_quote)

    # Rolling OLS (200 period window)
    betas_ols = np.full(n, np.nan)
    window = 200

    for i in range(window, n):
        X = price_quote[i-window:i].reshape(-1, 1)
        y = price_base[i-window:i]
        model = LinearRegression().fit(X, y)
        betas_ols[i] = model.coef_[0]

    # Compare tracking error
    kalman_error = np.abs(betas_kalman - true_beta)
    ols_error = np.abs(betas_ols - true_beta)

    print("Kalman Filter vs Rolling OLS Comparison")
    print("=" * 50)
    print(f"Kalman MAE: {np.nanmean(kalman_error):.4f}")
    print(f"OLS MAE:    {np.nanmean(ols_error):.4f}")
    print(f"Improvement: {(1 - np.nanmean(kalman_error)/np.nanmean(ols_error)) * 100:.1f}%")

    return betas_kalman, betas_ols, true_beta


if __name__ == "__main__":
    # Run test
    test_kalman_vs_ols()
