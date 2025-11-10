"""
Backtesting engine for strategy evaluation
"""

import pandas as pd
import numpy as np
from typing import Optional


class BacktestEngine:
    """Execute strategy backtest with realistic transaction costs"""

    def __init__(self, slippage_bps: float = 5.0):
        """
        Parameters:
        -----------
        slippage_bps : float
            Slippage cost in basis points (default: 5 bps)
        """
        self.slippage_bps = slippage_bps
        self.slippage_fraction = slippage_bps / 10000

    def run(
        self,
        signals: pd.DataFrame,
        price_series: pd.Series,
        holding_period: int = 60
    ) -> pd.DataFrame:
        """
        Execute backtest based on trading signals

        Parameters:
        -----------
        signals : pd.DataFrame
            Signals with 'position' column (-1, 0, +1)
        price_series : pd.Series
            Price time series (e.g., mid price)
        holding_period : int
            Holding period in seconds

        Returns:
        --------
        pd.DataFrame
            Trade results with columns:
            - entry_time, exit_time
            - position
            - entry_price, exit_price
            - raw_return, net_return
        """
        results = []

        # Align prices to signal timestamps
        prices = price_series.reindex(signals.index, method='ffill')

        for t, signal_row in signals.iterrows():
            position = signal_row['position']

            if position == 0:
                continue  # No trade

            entry_price = prices.loc[t]
            if pd.isna(entry_price):
                continue

            # Determine exit time
            exit_time = t + pd.Timedelta(seconds=holding_period)

            # Find nearest price at exit
            exit_idx = prices.index.searchsorted(exit_time)
            if exit_idx >= len(prices):
                exit_idx = len(prices) - 1

            exit_price = prices.iloc[exit_idx]
            if pd.isna(exit_price):
                exit_price = prices.iloc[-1]

            # Calculate returns
            raw_return = (exit_price - entry_price) / entry_price * position

            # Apply transaction costs (entry + exit)
            transaction_cost = 2 * self.slippage_fraction
            net_return = raw_return - transaction_cost

            results.append({
                'entry_time': t,
                'exit_time': prices.index[exit_idx] if exit_idx < len(prices) else prices.index[-1],
                'position': position,
                'entry_price': float(entry_price),
                'exit_price': float(exit_price),
                'raw_return': raw_return,
                'net_return': net_return
            })

        trades_df = pd.DataFrame(results)

        if not trades_df.empty:
            trades_df = trades_df.sort_values('entry_time').reset_index(drop=True)

        return trades_df

    def monte_carlo_simulation(
        self,
        trade_returns: np.ndarray,
        num_paths: int = 10000,
        seed: int = 42
    ) -> dict:
        """
        Perform Monte Carlo simulation by bootstrapping trade returns

        Parameters:
        -----------
        trade_returns : np.ndarray
            Array of individual trade returns
        num_paths : int
            Number of random paths to simulate
        seed : int
            Random seed for reproducibility

        Returns:
        --------
        dict
            Statistics from simulated distribution
        """
        if len(trade_returns) == 0:
            return {}

        rng = np.random.RandomState(seed)
        final_returns = []

        for _ in range(num_paths):
            # Bootstrap sample with replacement
            sample = rng.choice(trade_returns, size=len(trade_returns), replace=True)
            cum_return = np.prod(1 + sample) - 1
            final_returns.append(cum_return)

        final_returns = np.array(final_returns)

        return {
            'median_final_return': float(np.median(final_returns)),
            'mean_final_return': float(np.mean(final_returns)),
            'std_final_return': float(np.std(final_returns)),
            'pct_5_return': float(np.percentile(final_returns, 5)),
            'pct_95_return': float(np.percentile(final_returns, 95))
        }
