"""
Parameter Optimization

Walk-forward grid search with proper train/test separation to avoid
data leakage and overfitting.

DECISION CRITERIA:
    1. Expanding window: Train on all data up to test year
    2. Annual test periods: One year out-of-sample each iteration
    3. Grid search: Test all parameter combinations on train set
    4. Evaluate: Apply optimal parameters to test set
    5. Aggregate: Combine out-of-sample results across all test years

    CRITICAL: No lookahead bias. Test data never used in parameter selection.

INPUTS:
    - Historical data (prices, sentiment)
    - Parameter grid to search
    - Performance metric to optimize (e.g., Sharpe ratio)

OUTPUTS:
    - Optimal parameters
    - Out-of-sample performance by year
    - Parameter stability over time

ASSUMPTIONS:
    - Minimum 3 years of training data required
    - Parameters stable within 1-year test windows
    - Annual rebalancing of parameter selection

REFERENCES:
    Pardo, R. (2008). "The Evaluation and Optimization of Trading Strategies."
    John Wiley & Sons.
"""

from typing import Dict, List, Tuple, Callable, Any
import numpy as np
import pandas as pd
from itertools import product
from loguru import logger


class ParameterOptimizer:
    """
    Walk-forward parameter optimization.
    
    Parameters
    ----------
    parameter_grid : dict
        Dictionary of parameter names to list of values
    min_train_years : int, optional
        Minimum training period in years (default: 3)
    test_period_years : int, optional
        Test period length in years (default: 1)
    
    Examples
    --------
    >>> optimizer = ParameterOptimizer(
    ...     parameter_grid={
    ...         'price_weight': [0.30, 0.40, 0.50],
    ...         'sentiment_weight': [0.25, 0.35, 0.45],
    ...         'frog_weight': [0.15, 0.25, 0.35]
    ...     },
    ...     min_train_years=3
    ... )
    >>> 
    >>> results = optimizer.optimize(
    ...     data=historical_data,
    ...     strategy_func=build_strategy,
    ...     metric_func=calculate_sharpe
    ... )
    """
    
    def __init__(
        self,
        parameter_grid: Dict[str, List[Any]],
        min_train_years: int = 3,
        test_period_years: int = 1
    ):
        self.parameter_grid = parameter_grid
        self.min_train_years = min_train_years
        self.test_period_years = test_period_years
        
        # Generate all parameter combinations
        param_names = list(parameter_grid.keys())
        param_values = list(parameter_grid.values())
        self.param_combinations = [
            dict(zip(param_names, combo))
            for combo in product(*param_values)
        ]
        
        logger.info(
            f"ParameterOptimizer initialized: "
            f"{len(self.param_combinations)} combinations, "
            f"min_train={min_train_years}y, test={test_period_years}y"
        )
    
    def optimize(
        self,
        data: pd.DataFrame,
        strategy_func: Callable,
        metric_func: Callable,
        start_year: int = None,
        end_year: int = None
    ) -> Dict[str, Any]:
        """
        Run walk-forward optimization.
        
        Parameters
        ----------
        data : pd.DataFrame
            Historical data with datetime index
        strategy_func : callable
            Function that takes (data, **params) and returns strategy returns
        metric_func : callable
            Function that takes returns and calculates performance metric
        start_year : int, optional
            First test year (default: auto-detect from data)
        end_year : int, optional
            Last test year (default: auto-detect from data)
        
        Returns
        -------
        dict
            Optimization results including:
            - optimal_params: Best parameters across all periods
            - yearly_results: Out-of-sample performance by year
            - param_stability: How parameters changed over time
        
        Notes
        -----
        CRITICAL FIX: Proper train/test split with no lookahead bias.
        Training data: all years < test_year
        Test data: only test_year
        
        This ensures parameters are selected WITHOUT knowledge of future data.
        """
        logger.info("Starting walk-forward optimization")
        
        # Determine test years
        if start_year is None:
            start_year = data.index.year.min() + self.min_train_years
        if end_year is None:
            end_year = data.index.year.max()
        
        test_years = range(start_year, end_year + 1)
        
        yearly_results = []
        optimal_params_by_year = []
        
        for test_year in test_years:
            logger.info(f"Optimizing for test year {test_year}")
            
            # CRITICAL: Proper train/test split
            train_data = data[data.index.year < test_year]
            test_data = data[data.index.year == test_year]
            
            if len(train_data) < self.min_train_years * 252:
                logger.warning(
                    f"Insufficient training data for {test_year}, skipping"
                )
                continue
            
            # Grid search on training data
            best_metric = -np.inf
            best_params = None
            
            for params in self.param_combinations:
                try:
                    # Run strategy on training data
                    train_returns = strategy_func(train_data, **params)
                    
                    # Calculate metric
                    metric_value = metric_func(train_returns)
                    
                    if metric_value > best_metric:
                        best_metric = metric_value
                        best_params = params.copy()
                
                except Exception as e:
                    logger.warning(
                        f"Parameter combination failed: {params}, error: {e}"
                    )
                    continue
            
            if best_params is None:
                logger.error(f"No valid parameters found for {test_year}")
                continue
            
            # Evaluate on test data with best parameters
            try:
                test_returns = strategy_func(test_data, **best_params)
                test_metric = metric_func(test_returns)
                
                yearly_results.append({
                    'year': test_year,
                    'train_metric': best_metric,
                    'test_metric': test_metric,
                    'optimal_params': best_params
                })
                
                optimal_params_by_year.append(best_params)
                
                logger.info(
                    f"Year {test_year}: train_metric={best_metric:.3f}, "
                    f"test_metric={test_metric:.3f}"
                )
            
            except Exception as e:
                logger.error(f"Test evaluation failed for {test_year}: {e}")
                continue
        
        # Aggregate results
        if not yearly_results:
            raise ValueError("Optimization failed: no valid results")
        
        # Find most common optimal parameters (mode)
        optimal_params = self._find_mode_params(optimal_params_by_year)
        
        # Calculate average out-of-sample performance
        avg_test_metric = np.mean([r['test_metric'] for r in yearly_results])
        
        logger.info(
            f"Optimization complete: avg_test_metric={avg_test_metric:.3f}"
        )
        
        return {
            'optimal_params': optimal_params,
            'avg_test_metric': avg_test_metric,
            'yearly_results': yearly_results,
            'param_stability': self._calculate_stability(optimal_params_by_year)
        }
    
    def _find_mode_params(
        self,
        param_history: List[Dict]
    ) -> Dict[str, Any]:
        """
        Find most common parameter values across years.
        
        Parameters
        ----------
        param_history : List[dict]
            List of parameter dictionaries from each year
        
        Returns
        -------
        dict
            Mode (most frequent) value for each parameter
        """
        mode_params = {}
        
        for key in param_history[0].keys():
            values = [p[key] for p in param_history]
            
            # Find mode
            unique_values, counts = np.unique(values, return_counts=True)
            mode_idx = counts.argmax()
            mode_params[key] = unique_values[mode_idx]
        
        return mode_params
    
    def _calculate_stability(
        self,
        param_history: List[Dict]
    ) -> Dict[str, float]:
        """
        Calculate how stable parameters are over time.
        
        Parameters
        ----------
        param_history : List[dict]
            List of optimal parameters from each year
        
        Returns
        -------
        dict
            Stability metrics for each parameter (0=unstable, 1=stable)
        """
        stability = {}
        
        for key in param_history[0].keys():
            values = [p[key] for p in param_history]
            
            # Count how often mode appears
            unique_values, counts = np.unique(values, return_counts=True)
            mode_frequency = counts.max() / len(values)
            
            stability[key] = mode_frequency
        
        return stability
