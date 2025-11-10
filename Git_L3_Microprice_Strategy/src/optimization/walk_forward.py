"""
Walk-forward optimization for strategy parameters
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from ..backtesting.backtest_engine import BacktestEngine
from ..backtesting.performance_metrics import PerformanceMetrics


class WalkForwardOptimizer:
    """Walk-forward parameter optimization with out-of-sample validation"""

    def __init__(
        self,
        train_period_days: int = 30,
        test_period_days: int = 10,
        min_trades: int = 100
    ):
        """
        Parameters:
        -----------
        train_period_days : int
            Training window size in days
        test_period_days : int
            Testing window size in days
        min_trades : int
            Minimum trades required for valid optimization
        """
        self.train_period_days = train_period_days
        self.test_period_days = test_period_days
        self.min_trades = min_trades

    def optimize_gamma(
        self,
        features: pd.DataFrame,
        price_series: pd.Series,
        gamma_range: np.ndarray = None,
        holding_period: int = 60,
        slippage_bps: float = 5.0
    ) -> Dict:
        """
        Optimize gamma shape parameter via walk-forward analysis

        Parameters:
        -----------
        features : pd.DataFrame
            Complete feature set over entire period
        price_series : pd.Series
            Price series for backtesting
        gamma_range : np.ndarray
            Range of gamma values to test
        holding_period : int
            Holding period in seconds
        slippage_bps : float
            Slippage in basis points

        Returns:
        --------
        Dict
            Optimization results with best gamma and performance
        """
        if gamma_range is None:
            gamma_range = np.arange(0.5, 5.5, 0.5)

        # Split data into walk-forward windows
        windows = self._create_windows(features.index)

        if len(windows) == 0:
            return {'success': False, 'message': 'Insufficient data for walk-forward'}

        all_oos_results = []
        gamma_selections = []

        for train_start, train_end, test_start, test_end in windows:
            # Training phase: find best gamma
            train_features = features.loc[train_start:train_end]
            train_prices = price_series.loc[train_start:train_end]

            best_gamma, best_sharpe = self._optimize_on_window(
                train_features,
                train_prices,
                gamma_range,
                holding_period,
                slippage_bps
            )

            if best_gamma is None:
                continue

            gamma_selections.append({
                'train_start': train_start,
                'train_end': train_end,
                'test_start': test_start,
                'test_end': test_end,
                'selected_gamma': best_gamma,
                'train_sharpe': best_sharpe
            })

            # Testing phase: apply best gamma out-of-sample
            test_features = features.loc[test_start:test_end]
            test_prices = price_series.loc[test_start:test_end]

            oos_trades = self._backtest_with_gamma(
                test_features,
                test_prices,
                best_gamma,
                holding_period,
                slippage_bps
            )

            if not oos_trades.empty:
                all_oos_results.append(oos_trades)

        if not all_oos_results:
            return {'success': False, 'message': 'No valid out-of-sample results'}

        # Combine all out-of-sample results
        combined_oos = pd.concat(all_oos_results, ignore_index=True)
        combined_oos = combined_oos.sort_values('entry_time').reset_index(drop=True)

        # Compute performance on combined OOS results
        oos_metrics = PerformanceMetrics.compute(combined_oos)

        # Average selected gamma
        avg_gamma = np.mean([g['selected_gamma'] for g in gamma_selections])

        return {
            'success': True,
            'optimal_gamma': avg_gamma,
            'oos_metrics': oos_metrics,
            'oos_trades': combined_oos,
            'gamma_selections': pd.DataFrame(gamma_selections),
            'num_windows': len(windows)
        }

    def _create_windows(self, index: pd.DatetimeIndex) -> List[Tuple]:
        """Create walk-forward train/test windows"""
        windows = []
        start_date = index.min()
        end_date = index.max()

        current = start_date
        while current < end_date:
            train_start = current
            train_end = train_start + pd.Timedelta(days=self.train_period_days)

            if train_end >= end_date:
                break

            test_start = train_end
            test_end = test_start + pd.Timedelta(days=self.test_period_days)

            if test_end > end_date:
                test_end = end_date

            windows.append((train_start, train_end, test_start, test_end))

            # Move to next window
            current = test_start

        return windows

    def _optimize_on_window(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        gamma_range: np.ndarray,
        holding_period: int,
        slippage_bps: float
    ) -> Tuple[float, float]:
        """Find best gamma on training window"""
        best_gamma = None
        best_sharpe = -np.inf

        for gamma in gamma_range:
            trades = self._backtest_with_gamma(
                features,
                prices,
                gamma,
                holding_period,
                slippage_bps
            )

            if trades.empty or len(trades) < self.min_trades:
                continue

            metrics = PerformanceMetrics.compute(trades)
            sharpe = metrics.get('sharpe_ratio', -np.inf)

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_gamma = gamma

        return best_gamma, best_sharpe

    def _backtest_with_gamma(
        self,
        features: pd.DataFrame,
        prices: pd.Series,
        gamma: float,
        holding_period: int,
        slippage_bps: float
    ) -> pd.DataFrame:
        """Run backtest with specific gamma parameter"""
        # Recompute signals with this gamma (simplified - assumes features already computed)
        # In practice, you'd need to recompute microprice features with new gamma
        # For now, generate signals from existing features
        signals = self._generate_signals(features, gamma)

        # Run backtest
        engine = BacktestEngine(slippage_bps=slippage_bps)
        trades = engine.run(signals, prices, holding_period=holding_period)

        return trades

    def _generate_signals(
        self,
        features: pd.DataFrame,
        gamma: float,
        threshold: float = 0.5
    ) -> pd.DataFrame:
        """
        Generate trading signals from features

        Note: This is a simplified version. In production, you would
        recompute microprice features with the new gamma value.
        """
        signals = pd.DataFrame(index=features.index)

        # Standardize features
        feat_norm = features.copy()
        for col in feat_norm.columns:
            std = feat_norm[col].std()
            if std and std > 0:
                feat_norm[col] = (feat_norm[col] - feat_norm[col].mean()) / std

        # Composite signal (adjust weights as needed)
        if 'integrated_pressure' in feat_norm.columns:
            pressure_signal = feat_norm['integrated_pressure'].fillna(0)
        else:
            pressure_signal = 0

        # OFI signal
        ofi_cols = [c for c in feat_norm.columns
                   if c.startswith('ofi_') and c.endswith('_norm')]
        if ofi_cols:
            ofi_signal = feat_norm[ofi_cols].mean(axis=1)
        else:
            ofi_signal = 0

        # Volume signal
        if 'volume_imbalance' in feat_norm.columns:
            volume_signal = feat_norm['volume_imbalance'].fillna(0)
        else:
            volume_signal = 0

        # Weighted composite
        composite = 0.5 * pressure_signal + 0.3 * ofi_signal + 0.2 * volume_signal

        # Generate positions
        signals['position'] = 0
        signals.loc[composite > threshold, 'position'] = 1
        signals.loc[composite < -threshold, 'position'] = -1

        return signals

    def grid_search(
        self,
        features: pd.DataFrame,
        price_series: pd.Series,
        param_grid: Dict,
        holding_period: int = 60,
        slippage_bps: float = 5.0
    ) -> pd.DataFrame:
        """
        Grid search over multiple parameters

        Parameters:
        -----------
        features : pd.DataFrame
            Feature dataframe
        price_series : pd.Series
            Price series
        param_grid : Dict
            Dictionary of parameter ranges
            Example: {'gamma': [1.0, 2.0, 3.0], 'threshold': [0.3, 0.5, 0.7]}
        holding_period : int
            Holding period in seconds
        slippage_bps : float
            Slippage in basis points

        Returns:
        --------
        pd.DataFrame
            Grid search results sorted by Sharpe ratio
        """
        from itertools import product

        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))

        results = []
        engine = BacktestEngine(slippage_bps=slippage_bps)

        for combo in combinations:
            params = dict(zip(param_names, combo))

            # Generate signals with these parameters
            signals = self._generate_signals_with_params(features, params)

            # Run backtest
            trades = engine.run(signals, price_series, holding_period=holding_period)

            if trades.empty:
                continue

            # Compute metrics
            metrics = PerformanceMetrics.compute(trades)

            result = {**params, **metrics}
            results.append(result)

        results_df = pd.DataFrame(results)

        if not results_df.empty:
            results_df = results_df.sort_values('sharpe_ratio', ascending=False)

        return results_df

    def _generate_signals_with_params(
        self,
        features: pd.DataFrame,
        params: Dict
    ) -> pd.DataFrame:
        """Generate signals with specific parameter values"""
        gamma = params.get('gamma', 2.0)
        threshold = params.get('threshold', 0.5)

        return self._generate_signals(features, gamma, threshold)
