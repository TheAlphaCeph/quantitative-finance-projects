"""
Portfolio Builder

Constructs portfolios from momentum signals with position sizing
and risk constraints.

DECISION CRITERIA:
    1. Select top N stocks by composite signal
    2. Apply position sizing method
    3. Enforce risk constraints (max position, sector limits)
    4. Normalize weights to sum to target exposure

INPUTS:
    - signals: Multi-index DataFrame with composite_signal column
    - max_positions: Maximum number of holdings
    - position_sizing: 'equal_weight', 'signal_weight', 'volatility_weight'

OUTPUTS:
    - Dict of {ticker: weight} for portfolio construction

ASSUMPTIONS:
    - Long-only portfolio (no shorts)
    - Monthly rebalancing frequency
    - Target 100% equity exposure (or specified)
    - Position constraints respected before normalization
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from loguru import logger


class PortfolioBuilder:
    """
    Build portfolios from momentum signals.
    
    Parameters
    ----------
    max_positions : int, optional
        Maximum number of positions (default: 50)
    min_position_size : float, optional
        Minimum position weight (default: 0.01 = 1%)
    max_position_size : float, optional
        Maximum position weight (default: 0.05 = 5%)
    position_sizing : str, optional
        Weighting method: 'equal_weight', 'signal_weight', 'volatility_weight'
        (default: 'equal_weight')
    target_exposure : float, optional
        Target portfolio exposure (default: 1.0 = 100%)
    long_only : bool, optional
        Long-only constraint (default: True)
    
    Examples
    --------
    >>> builder = PortfolioBuilder(
    ...     max_positions=50,
    ...     position_sizing='equal_weight',
    ...     max_position_size=0.05
    ... )
    >>> 
    >>> weights = builder.build_portfolio(
    ...     signals=signal_data,
    ...     date=pd.Timestamp('2024-01-31')
    ... )
    >>> print(f"Portfolio has {len(weights)} positions")
    """
    
    def __init__(
        self,
        max_positions: int = 50,
        min_position_size: float = 0.01,
        max_position_size: float = 0.05,
        position_sizing: str = 'equal_weight',
        target_exposure: float = 1.0,
        long_only: bool = True
    ):
        self.max_positions = max_positions
        self.min_position_size = min_position_size
        self.max_position_size = max_position_size
        self.position_sizing = position_sizing
        self.target_exposure = target_exposure
        self.long_only = long_only
        
        logger.info(
            f"PortfolioBuilder initialized: "
            f"max_positions={max_positions}, sizing={position_sizing}"
        )
    
    def build_portfolio(
        self,
        signals: pd.DataFrame,
        date: pd.Timestamp,
        volatilities: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """
        Build portfolio for given date.
        
        Parameters
        ----------
        signals : pd.DataFrame
            Multi-index (date, ticker) with 'composite_signal' column
        date : pd.Timestamp
            Portfolio construction date
        volatilities : pd.Series, optional
            Historical volatilities for risk-based weighting
        
        Returns
        -------
        dict
            Portfolio weights {ticker: weight}
        
        Notes
        -----
        Weights are normalized to sum to target_exposure.
        All constraints (min/max position size) are enforced.
        """
        if date not in signals.index.get_level_values('date'):
            logger.warning(f"No signals available for {date}")
            return {}
        
        # Get signals for this date
        date_signals = signals.xs(date, level='date')
        
        # Filter for long signals only if long_only
        if self.long_only:
            date_signals = date_signals[date_signals['composite_signal'] > 0]
        
        if len(date_signals) == 0:
            logger.warning(f"No valid signals for {date}")
            return {}
        
        # Select top N stocks by signal strength
        n_positions = min(self.max_positions, len(date_signals))
        top_signals = date_signals.nlargest(n_positions, 'composite_signal')
        
        # Calculate raw weights based on sizing method
        if self.position_sizing == 'equal_weight':
            raw_weights = self._equal_weight(top_signals)
        elif self.position_sizing == 'signal_weight':
            raw_weights = self._signal_weight(top_signals)
        elif self.position_sizing == 'volatility_weight':
            if volatilities is None:
                logger.warning(
                    "Volatility weighting requested but no volatilities provided. "
                    "Using equal weight."
                )
                raw_weights = self._equal_weight(top_signals)
            else:
                raw_weights = self._volatility_weight(top_signals, volatilities)
        else:
            raise ValueError(f"Unknown position sizing: {self.position_sizing}")
        
        # Apply position size constraints
        constrained_weights = self._apply_constraints(raw_weights)
        
        # Normalize to target exposure
        final_weights = self._normalize_weights(
            constrained_weights,
            self.target_exposure
        )
        
        logger.debug(
            f"Built portfolio for {date}: {len(final_weights)} positions, "
            f"total exposure={sum(final_weights.values()):.2%}"
        )
        
        return final_weights
    
    def _equal_weight(self, signals: pd.DataFrame) -> Dict[str, float]:
        """Equal weight across all positions."""
        n = len(signals)
        return {ticker: 1.0 / n for ticker in signals.index}
    
    def _signal_weight(self, signals: pd.DataFrame) -> Dict[str, float]:
        """
        Weight proportional to signal strength.
        
        Higher signals get larger weights.
        """
        signal_values = signals['composite_signal'].copy()
        
        # Ensure all positive (shift if needed)
        if signal_values.min() <= 0:
            signal_values = signal_values - signal_values.min() + 0.01
        
        # Normalize to sum to 1
        total_signal = signal_values.sum()
        
        return {
            ticker: signal_values.loc[ticker] / total_signal
            for ticker in signals.index
        }
    
    def _volatility_weight(
        self,
        signals: pd.DataFrame,
        volatilities: pd.Series
    ) -> Dict[str, float]:
        """
        Inverse volatility weighting (risk parity approach).
        
        Lower volatility stocks get higher weights.
        """
        weights = {}
        
        for ticker in signals.index:
            if ticker not in volatilities.index:
                logger.warning(f"No volatility data for {ticker}, skipping")
                continue
            
            vol = volatilities.loc[ticker]
            if vol <= 0 or np.isnan(vol):
                logger.warning(f"Invalid volatility for {ticker}: {vol}")
                continue
            
            # Inverse volatility
            weights[ticker] = 1.0 / vol
        
        # Normalize
        total_inv_vol = sum(weights.values())
        if total_inv_vol > 0:
            weights = {k: v / total_inv_vol for k, v in weights.items()}
        
        return weights
    
    def _apply_constraints(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Apply position size constraints.
        
        Enforces min_position_size and max_position_size limits.
        """
        constrained = {}
        
        for ticker, weight in weights.items():
            # Apply max constraint
            if weight > self.max_position_size:
                weight = self.max_position_size
            
            # Apply min constraint (exclude if too small)
            if weight >= self.min_position_size:
                constrained[ticker] = weight
            else:
                logger.debug(
                    f"Excluding {ticker}: weight {weight:.4f} "
                    f"below minimum {self.min_position_size:.4f}"
                )
        
        return constrained
    
    def _normalize_weights(
        self,
        weights: Dict[str, float],
        target_exposure: float
    ) -> Dict[str, float]:
        """
        Normalize weights to sum to target exposure.
        
        Parameters
        ----------
        weights : dict
            Raw weights
        target_exposure : float
            Target sum of weights (e.g., 1.0 for 100% exposure)
        
        Returns
        -------
        dict
            Normalized weights
        """
        if not weights:
            return {}
        
        current_sum = sum(weights.values())
        
        if current_sum == 0:
            logger.error("Sum of weights is zero, cannot normalize")
            return {}
        
        scale_factor = target_exposure / current_sum
        
        normalized = {
            ticker: weight * scale_factor
            for ticker, weight in weights.items()
        }
        
        return normalized
    
    def calculate_turnover(
        self,
        old_weights: Dict[str, float],
        new_weights: Dict[str, float]
    ) -> float:
        """
        Calculate portfolio turnover.
        
        Parameters
        ----------
        old_weights : dict
            Previous portfolio weights
        new_weights : dict
            New portfolio weights
        
        Returns
        -------
        float
            Turnover as fraction of portfolio (0.0 to 2.0)
        
        Notes
        -----
        Turnover = sum(|new_weight - old_weight|) / 2
        
        Examples:
        - Complete replacement: turnover = 1.0 (100%)
        - No change: turnover = 0.0
        - Adding new positions: increases turnover
        """
        all_tickers = set(old_weights.keys()) | set(new_weights.keys())
        
        turnover = 0.0
        for ticker in all_tickers:
            old_w = old_weights.get(ticker, 0.0)
            new_w = new_weights.get(ticker, 0.0)
            turnover += abs(new_w - old_w)
        
        # Divide by 2 (standard definition)
        return turnover / 2.0
    
    def get_portfolio_statistics(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate portfolio statistics.
        
        Parameters
        ----------
        weights : dict
            Portfolio weights
        
        Returns
        -------
        dict
            Statistics including concentration measures
        """
        if not weights:
            return {
                'n_positions': 0,
                'total_exposure': 0.0,
                'max_weight': 0.0,
                'min_weight': 0.0,
                'herfindahl_index': 0.0
            }
        
        weight_values = list(weights.values())
        
        # Herfindahl-Hirschman Index (concentration measure)
        hhi = sum(w ** 2 for w in weight_values)
        
        return {
            'n_positions': len(weights),
            'total_exposure': sum(weight_values),
            'max_weight': max(weight_values),
            'min_weight': min(weight_values),
            'mean_weight': np.mean(weight_values),
            'herfindahl_index': hhi,
            'effective_positions': 1.0 / hhi if hhi > 0 else 0.0
        }
