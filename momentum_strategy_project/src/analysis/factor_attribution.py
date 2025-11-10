"""
Factor Attribution Analysis

Orthogonalizes strategy returns against Fama-French factors to prove
alpha is independent of common risk exposures.

DECISION CRITERIA:
    Regress excess returns on factor returns:
    R_t - R_f = α + β_MKT·MKT_t + β_SMB·SMB_t + β_HML·HML_t + ε_t
    
    Where:
    - α (alpha) is factor-neutral return
    - β coefficients are factor exposures
    - ε is idiosyncratic return

INPUTS:
    - strategy_returns: Daily strategy returns
    - factor_returns: DataFrame with MKT-RF, SMB, HML, RF columns
    - Use OLS regression with Newey-West standard errors

OUTPUTS:
    - Alpha (annualized)
    - Factor betas
    - R-squared
    - t-statistics

ASSUMPTIONS:
    - Risk-free rate subtracted from both strategy and market returns
    - Newey-West with lag=21 for autocorrelation/heteroskedasticity
    - Annual alpha calculated: α_annual = α_daily · 252

REFERENCES:
    Fama, E. F., & French, K. R. (2015). "A Five-Factor Asset Pricing Model."
    Journal of Financial Economics, 116(1), 1-22.
"""

from typing import Dict, Optional
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.linear_model import OLS
from loguru import logger


class FactorAttribution:
    """
    Perform Fama-French factor attribution.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Daily strategy returns
    factor_returns : pd.DataFrame
        Daily factor returns with columns: mktrf, smb, hml, rf
    periods_per_year : int, optional
        Trading periods for annualization (default: 252)
    
    Examples
    --------
    >>> attribution = FactorAttribution(
    ...     strategy_returns=strat_returns,
    ...     factor_returns=ff_factors
    ... )
    >>> 
    >>> results = attribution.run_regression()
    >>> print(f"Alpha: {results['alpha_annual']:.4f}")
    >>> print(f"Market Beta: {results['beta_mkt']:.2f}")
    """
    
    def __init__(
        self,
        strategy_returns: pd.Series,
        factor_returns: pd.DataFrame,
        periods_per_year: int = 252
    ):
        self.strategy_returns = strategy_returns.copy()
        self.factor_returns = factor_returns.copy()
        self.periods_per_year = periods_per_year
        
        # Align dates
        common_dates = self.strategy_returns.index.intersection(
            self.factor_returns.index
        )
        self.strategy_returns = self.strategy_returns.loc[common_dates]
        self.factor_returns = self.factor_returns.loc[common_dates]
        
        logger.info(
            f"FactorAttribution initialized with {len(common_dates)} observations"
        )
    
    def run_regression(
        self,
        use_newey_west: bool = True,
        lags: int = 21
    ) -> Dict[str, float]:
        """
        Run factor regression.
        
        Parameters
        ----------
        use_newey_west : bool, optional
            Use Newey-West standard errors (default: True)
        lags : int, optional
            Lag length for Newey-West (default: 21 days)
        
        Returns
        -------
        dict
            Regression results including alpha, betas, t-stats, R-squared
        
        Notes
        -----
        CRITICAL: Subtracts risk-free rate from strategy returns before regression.
        This ensures proper excess return calculation.
        """
        logger.info("Running Fama-French factor regression")
        
        # Calculate excess returns for regression
        excess_returns = self.strategy_returns - self.factor_returns['rf']
        
        # Prepare regression
        y = excess_returns.values
        X = self.factor_returns[['mktrf', 'smb', 'hml']].values
        X = sm.add_constant(X)  # Add intercept
        
        # Run OLS
        model = OLS(y, X)
        
        if use_newey_west:
            # Newey-West for autocorrelation and heteroskedasticity
            results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lags})
        else:
            results = model.fit()
        
        # Extract coefficients
        alpha_daily = results.params[0]
        beta_mkt = results.params[1]
        beta_smb = results.params[2]
        beta_hml = results.params[3]
        
        # Extract t-statistics
        t_alpha = results.tvalues[0]
        t_mkt = results.tvalues[1]
        t_smb = results.tvalues[2]
        t_hml = results.tvalues[3]
        
        # Calculate p-values
        p_alpha = results.pvalues[0]
        
        # Annualize alpha
        alpha_annual = alpha_daily * self.periods_per_year
        
        # R-squared
        r_squared = results.rsquared
        
        logger.info(
            f"Regression complete: "
            f"alpha={alpha_annual:.4f} (t={t_alpha:.2f}), "
            f"R²={r_squared:.3f}"
        )
        
        return {
            'alpha_daily': alpha_daily,
            'alpha_annual': alpha_annual,
            'beta_mkt': beta_mkt,
            'beta_smb': beta_smb,
            'beta_hml': beta_hml,
            't_alpha': t_alpha,
            't_mkt': t_mkt,
            't_smb': t_smb,
            't_hml': t_hml,
            'p_alpha': p_alpha,
            'r_squared': r_squared,
            'n_obs': len(y)
        }
    
    def rolling_regression(
        self,
        window: int = 252
    ) -> pd.DataFrame:
        """
        Run rolling factor regressions.
        
        Parameters
        ----------
        window : int, optional
            Rolling window size in days (default: 252)
        
        Returns
        -------
        pd.DataFrame
            Time series of rolling alphas and betas
        """
        logger.info(f"Running rolling regression with {window}-day window")
        
        results_list = []
        
        for i in range(window, len(self.strategy_returns)):
            window_returns = self.strategy_returns.iloc[i - window:i]
            window_factors = self.factor_returns.iloc[i - window:i]
            
            # Calculate excess returns
            excess_returns = window_returns - window_factors['rf']
            
            # Regression
            y = excess_returns.values
            X = window_factors[['mktrf', 'smb', 'hml']].values
            X = sm.add_constant(X)
            
            model = OLS(y, X)
            fit = model.fit()
            
            results_list.append({
                'date': self.strategy_returns.index[i],
                'alpha': fit.params[0] * self.periods_per_year,
                'beta_mkt': fit.params[1],
                'beta_smb': fit.params[2],
                'beta_hml': fit.params[3],
                'r_squared': fit.rsquared
            })
        
        results_df = pd.DataFrame(results_list)
        results_df = results_df.set_index('date')
        
        logger.info(f"Rolling regression complete: {len(results_df)} periods")
        
        return results_df
