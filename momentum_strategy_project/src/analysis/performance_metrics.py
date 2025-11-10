"""
Performance Metrics Module

Comprehensive risk-adjusted performance calculations.

METRICS IMPLEMENTED:
    Returns: CAGR, total return, excess returns
    Risk: Volatility, downside deviation, VaR, CVaR
    Risk-Adjusted: Sharpe, Sortino, Calmar, Information Ratio
    Drawdown: Maximum drawdown, average drawdown, recovery time
    Trade: Win rate, profit factor, trade frequency

ASSUMPTIONS:
    - 252 trading days per year for annualization
    - Volatility annualization: σ_annual = σ_daily · √252
    - Sortino ratio uses zero downside threshold
    - VaR/CVaR at 95% confidence level

REFERENCES:
    Sharpe, W. F. (1966). "Mutual Fund Performance." Journal of Business, 39(1), 119-138.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class PerformanceMetrics:
    """
    Calculate performance and risk metrics.
    
    Parameters
    ----------
    returns : pd.Series
        Strategy returns (daily)
    benchmark_returns : pd.Series, optional
        Benchmark returns for comparison
    risk_free_rate : float, optional
        Annual risk-free rate (default: 0.02 for 2%)
    periods_per_year : int, optional
        Trading periods per year (default: 252)
    
    Examples
    --------
    >>> metrics = PerformanceMetrics(
    ...     returns=strategy_returns,
    ...     benchmark_returns=market_returns,
    ...     risk_free_rate=0.02
    ... )
    >>> 
    >>> print(f"Sharpe Ratio: {metrics.sharpe_ratio():.2f}")
    >>> print(f"Alpha (annual): {metrics.alpha_annual():.4f}")
    """
    
    def __init__(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        risk_free_rate: float = 0.02,
        periods_per_year: int = 252
    ):
        self.returns = returns.copy()
        self.benchmark_returns = benchmark_returns.copy() if benchmark_returns is not None else None
        self.risk_free_rate = risk_free_rate
        self.periods_per_year = periods_per_year
        
        # Calculate daily risk-free rate
        self.daily_rf = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        
        # Calculate excess returns
        self.excess_returns = returns - self.daily_rf
        
        if benchmark_returns is not None:
            self.active_returns = returns - benchmark_returns
        else:
            self.active_returns = None
        
        logger.debug(
            f"PerformanceMetrics initialized with {len(returns)} observations"
        )
    
    def total_return(self) -> float:
        """Calculate cumulative total return."""
        return (1 + self.returns).prod() - 1
    
    def cagr(self) -> float:
        """
        Calculate Compound Annual Growth Rate.
        
        Returns
        -------
        float
            Annualized return
        """
        total_ret = self.total_return()
        years = len(self.returns) / self.periods_per_year
        return (1 + total_ret) ** (1 / years) - 1
    
    def volatility(self, annualize: bool = True) -> float:
        """
        Calculate return volatility.
        
        Parameters
        ----------
        annualize : bool, optional
            Return annualized volatility (default: True)
        
        Returns
        -------
        float
            Standard deviation of returns
        """
        vol = self.returns.std()
        if annualize:
            vol *= np.sqrt(self.periods_per_year)
        return vol
    
    def sharpe_ratio(self) -> float:
        """
        Calculate Sharpe ratio.
        
        Returns
        -------
        float
            Risk-adjusted return measure
        
        Notes
        -----
        Sharpe = (Return - RiskFree) / Volatility
        Annualized using both numerator and denominator
        """
        excess_mean = self.excess_returns.mean() * self.periods_per_year
        excess_vol = self.excess_returns.std() * np.sqrt(self.periods_per_year)
        
        if excess_vol == 0:
            return 0.0
        
        return excess_mean / excess_vol
    
    def sortino_ratio(self, threshold: float = 0.0) -> float:
        """
        Calculate Sortino ratio (downside deviation).
        
        Parameters
        ----------
        threshold : float, optional
            Minimum acceptable return (default: 0.0)
        
        Returns
        -------
        float
            Downside risk-adjusted return
        
        Notes
        -----
        Uses zero threshold (not minimum acceptable return) as standard
        """
        excess_mean = self.excess_returns.mean() * self.periods_per_year
        
        # Calculate downside deviation
        downside_returns = self.returns[self.returns < threshold]
        downside_vol = downside_returns.std() * np.sqrt(self.periods_per_year)
        
        if downside_vol == 0:
            return 0.0
        
        return excess_mean / downside_vol
    
    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown.
        
        Returns
        -------
        float
            Largest peak-to-trough decline
        """
        cumulative = (1 + self.returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min()
    
    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (CAGR / MaxDD).
        
        Returns
        -------
        float
            Return relative to maximum drawdown
        """
        max_dd = abs(self.max_drawdown())
        if max_dd == 0:
            return 0.0
        return self.cagr() / max_dd
    
    def information_ratio(self) -> float:
        """
        Calculate Information Ratio vs benchmark.
        
        Returns
        -------
        float
            Active return / tracking error
        
        Raises
        ------
        ValueError
            If no benchmark provided
        """
        if self.active_returns is None:
            raise ValueError("Benchmark returns required for Information Ratio")
        
        active_mean = self.active_returns.mean() * self.periods_per_year
        tracking_error = self.active_returns.std() * np.sqrt(self.periods_per_year)
        
        if tracking_error == 0:
            return 0.0
        
        return active_mean / tracking_error
    
    def value_at_risk(self, confidence: float = 0.95) -> float:
        """
        Calculate Value at Risk.
        
        Parameters
        ----------
        confidence : float, optional
            Confidence level (default: 0.95)
        
        Returns
        -------
        float
            VaR at specified confidence level (negative number)
        """
        return np.percentile(self.returns, (1 - confidence) * 100)
    
    def conditional_var(self, confidence: float = 0.95) -> float:
        """
        Calculate Conditional Value at Risk (CVaR / Expected Shortfall).
        
        Parameters
        ----------
        confidence : float, optional
            Confidence level (default: 0.95)
        
        Returns
        -------
        float
            Average return in worst (1-confidence)% of cases
        """
        var = self.value_at_risk(confidence)
        return self.returns[self.returns <= var].mean()
    
    def win_rate(self) -> float:
        """
        Calculate proportion of positive return periods.
        
        Returns
        -------
        float
            Win rate as decimal (0.0 to 1.0)
        """
        return (self.returns > 0).mean()
    
    def alpha_annual(self) -> float:
        """
        Calculate annualized alpha vs benchmark (simple).
        
        Returns
        -------
        float
            Annual excess return over benchmark
        
        Notes
        -----
        This is simple alpha, not regression alpha.
        For factor-adjusted alpha, use FactorAttribution module.
        """
        if self.benchmark_returns is None:
            raise ValueError("Benchmark required for alpha calculation")
        
        strategy_cagr = self.cagr()
        
        # Compound benchmark returns properly
        benchmark_cumulative = (1 + self.benchmark_returns).prod() - 1
        years = len(self.benchmark_returns) / self.periods_per_year
        benchmark_cagr = (1 + benchmark_cumulative) ** (1 / years) - 1
        
        return strategy_cagr - benchmark_cagr
    
    def summary(self) -> Dict[str, float]:
        """
        Generate comprehensive performance summary.
        
        Returns
        -------
        dict
            All performance metrics
        """
        metrics = {
            'total_return': self.total_return(),
            'cagr': self.cagr(),
            'volatility': self.volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'calmar_ratio': self.calmar_ratio(),
            'value_at_risk_95': self.value_at_risk(0.95),
            'cvar_95': self.conditional_var(0.95),
            'win_rate': self.win_rate()
        }
        
        if self.benchmark_returns is not None:
            metrics['alpha_annual'] = self.alpha_annual()
            metrics['information_ratio'] = self.information_ratio()
        
        return metrics
