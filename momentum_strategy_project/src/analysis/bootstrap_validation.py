"""
Bootstrap Validation

Statistical significance testing using block bootstrap to account
for autocorrelation in returns.

DECISION CRITERIA:
    1. Resample returns in blocks to preserve autocorrelation structure
    2. Calculate metrics on each bootstrap sample (1000 iterations)
    3. Construct 95% confidence intervals from bootstrap distribution
    
    Block length = 21 days (approximately 1 month) chosen to capture
    typical autocorrelation structure in daily equity returns.

INPUTS:
    - returns: Strategy returns time series
    - block_length: Size of blocks for resampling (default: 21)
    - n_iterations: Number of bootstrap samples (default: 1000)
    - random_state: Seed for reproducibility (default: 66)

OUTPUTS:
    - Point estimates for each metric
    - 95% confidence intervals [2.5%, 97.5%]
    - p-values for hypothesis tests

ASSUMPTIONS:
    - Returns exhibit autocorrelation within 21-day windows
    - Block length captures typical dependence structure
    - Parallel execution uses independent random states
    - n_jobs=-1 uses all available CPU cores

REFERENCES:
    Politis, D. N., & Romano, J. P. (1994). "The Stationary Bootstrap."
    Journal of the American Statistical Association, 89(428), 1303-1313.
"""

from typing import Dict, Callable, Optional
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from loguru import logger


class BootstrapValidation:
    """
    Block bootstrap validation for strategy metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns
    block_length : int, optional
        Bootstrap block size (default: 21 days)
    n_iterations : int, optional
        Number of bootstrap samples (default: 1000)
    random_state : int, optional
        Random seed for reproducibility (default: 66)
    n_jobs : int, optional
        Number of parallel jobs (default: -1 for all cores)
    
    Examples
    --------
    >>> validator = BootstrapValidation(
    ...     returns=strategy_returns,
    ...     block_length=21,
    ...     n_iterations=1000,
    ...     random_state=66
    ... )
    >>> 
    >>> sharpe_ci = validator.bootstrap_metric(
    ...     metric_func=lambda r: r.mean() / r.std() * np.sqrt(252)
    ... )
    >>> print(f"Sharpe 95% CI: [{sharpe_ci['ci_lower']:.2f}, {sharpe_ci['ci_upper']:.2f}]")
    """
    
    def __init__(
        self,
        returns: pd.Series,
        block_length: int = 21,
        n_iterations: int = 1000,
        random_state: int = 66,
        n_jobs: int = -1
    ):
        self.returns = returns.values  # Convert to numpy for speed
        self.block_length = block_length
        self.n_iterations = n_iterations
        self.random_state = random_state
        self.n_jobs = n_jobs if n_jobs > 0 else -1
        
        # Set global random seed
        np.random.seed(random_state)
        
        logger.info(
            f"BootstrapValidation initialized: "
            f"n={len(returns)}, block_length={block_length}, "
            f"iterations={n_iterations}, seed={random_state}"
        )
    
    def _block_bootstrap_sample(self, iteration: int) -> np.ndarray:
        """
        Generate one block bootstrap sample.
        
        Parameters
        ----------
        iteration : int
            Iteration number (used for seeding)
        
        Returns
        -------
        np.ndarray
            Bootstrapped return series
        """
        # Create independent random state for this iteration
        rng = np.random.RandomState(self.random_state + iteration)
        
        n = len(self.returns)
        n_blocks = int(np.ceil(n / self.block_length))
        
        # Sample random starting points for blocks
        bootstrap_sample = []
        
        for _ in range(n_blocks):
            start_idx = rng.randint(0, n - self.block_length + 1)
            block = self.returns[start_idx:start_idx + self.block_length]
            bootstrap_sample.extend(block)
        
        # Trim to original length
        bootstrap_sample = bootstrap_sample[:n]
        
        return np.array(bootstrap_sample)
    
    def bootstrap_metric(
        self,
        metric_func: Callable[[np.ndarray], float],
        confidence_level: float = 0.95
    ) -> Dict[str, float]:
        """
        Bootstrap confidence interval for a metric.
        
        Parameters
        ----------
        metric_func : callable
            Function that takes returns array and returns scalar metric
        confidence_level : float, optional
            Confidence level (default: 0.95)
        
        Returns
        -------
        dict
            Point estimate, CI bounds, and distribution
        
        Examples
        --------
        >>> # Sharpe ratio
        >>> sharpe_func = lambda r: r.mean() / r.std() * np.sqrt(252)
        >>> results = validator.bootstrap_metric(sharpe_func)
        >>> 
        >>> # Alpha (requires benchmark)
        >>> alpha_func = lambda r: (np.prod(1 + r) ** (252 / len(r)) - 1) - benchmark_cagr
        >>> results = validator.bootstrap_metric(alpha_func)
        """
        logger.info(f"Running bootstrap with {self.n_iterations} iterations")
        
        # Original metric value
        original_metric = metric_func(self.returns)
        
        # Bootstrap distribution
        bootstrap_metrics = []
        
        # Use parallel execution with proper random state management
        with ProcessPoolExecutor(max_workers=None) as executor:
            bootstrap_samples = executor.map(
                self._block_bootstrap_sample,
                range(self.n_iterations)
            )
            
            for sample in bootstrap_samples:
                try:
                    metric_value = metric_func(sample)
                    bootstrap_metrics.append(metric_value)
                except Exception as e:
                    logger.warning(f"Bootstrap iteration failed: {e}")
                    continue
        
        bootstrap_metrics = np.array(bootstrap_metrics)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_metrics, alpha / 2 * 100)
        ci_upper = np.percentile(bootstrap_metrics, (1 - alpha / 2) * 100)
        
        # P-value (two-tailed test against zero)
        p_value = np.mean(
            (bootstrap_metrics <= 0) | (bootstrap_metrics >= 2 * original_metric)
        )
        
        logger.info(
            f"Bootstrap complete: "
            f"estimate={original_metric:.4f}, "
            f"CI=[{ci_lower:.4f}, {ci_upper:.4f}]"
        )
        
        return {
            'estimate': original_metric,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'std_error': bootstrap_metrics.std(),
            'p_value': p_value,
            'distribution': bootstrap_metrics
        }
    
    def validate_sharpe_ratio(self) -> Dict[str, float]:
        """
        Bootstrap Sharpe ratio.
        
        Returns
        -------
        dict
            Sharpe ratio with 95% confidence interval
        """
        sharpe_func = lambda r: (r.mean() / r.std()) * np.sqrt(252)
        return self.bootstrap_metric(sharpe_func)
    
    def validate_alpha(
        self,
        benchmark_returns: np.ndarray
    ) -> Dict[str, float]:
        """
        Bootstrap alpha calculation.
        
        Parameters
        ----------
        benchmark_returns : np.ndarray
            Benchmark returns (same length as strategy)
        
        Returns
        -------
        dict
            Alpha with 95% confidence interval
        """
        # Calculate benchmark CAGR
        n_years = len(benchmark_returns) / 252
        benchmark_cagr = (np.prod(1 + benchmark_returns)) ** (1 / n_years) - 1
        
        def alpha_func(r):
            years = len(r) / 252
            strategy_cagr = (np.prod(1 + r)) ** (1 / years) - 1
            return strategy_cagr - benchmark_cagr
        
        return self.bootstrap_metric(alpha_func)
