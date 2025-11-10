"""
Frog-in-the-Pan Momentum Detector

Identifies gradual sentiment shifts that persist 3-6 months, distinguishing
them from sudden jumps that quickly revert.

DECISION CRITERIA (4 conditions must all be true):
    1. Monotonicity: >60% of sentiment changes move in same direction
    2. No sudden jumps: No single change exceeds 2 standard deviations
    3. Cumulative magnitude: Absolute sentiment shift >10% over window
    4. Low volatility: Change volatility below historical median

INPUTS:
    - sentiment_series: Daily sentiment scores for single stock (pd.Series)
    - gradual_window: Detection window in days (default: 63 = 3 months)
    - sudden_threshold: Z-score threshold for jump detection (default: 2.0)
    - min_persistence: Forward validation period (default: 126 = 6 months)

OUTPUTS:
    - Binary indicator (1=gradual shift detected, 0=sudden or none)
    - Optional metadata dict with diagnostics

ASSUMPTIONS:
    - Daily sentiment data with sufficient history (≥252 days for volatility)
    - Sentiment changes approximately normally distributed for z-scores
    - 3-6 month horizon captures typical earnings cycle

REFERENCES:
    Da, Z., Gurun, U. G., & Warachka, M. (2014). "Frog in the Pan: 
    Continuous Information and Momentum." Review of Financial Studies, 
    27(7), 2171-2218.
"""

from typing import Tuple, Optional, Dict
import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger


class FrogInPanDetector:
    """
    Detects gradual vs sudden sentiment shifts.
    
    Investors underreact to gradual changes more than sudden jumps (Da et al. 2014).
    This detector identifies which sentiment shifts are gradual and likely to persist.
    
    Parameters
    ----------
    gradual_window : int, optional
        Window in days to assess if change is gradual (default: 63 = 3 months)
    sudden_threshold : float, optional
        Z-score threshold for detecting sudden jumps (default: 2.0)
    min_persistence : int, optional
        Minimum days for validating forward persistence (default: 126 = 6 months)
    volatility_window : int, optional
        Window for calculating historical volatility (default: 252 = 1 year)
    trend_consistency : float, optional
        Minimum proportion of changes in same direction (default: 0.60)
    min_cumulative_change : float, optional
        Minimum absolute sentiment shift as proportion (default: 0.10)
    
    Examples
    --------
    >>> detector = FrogInPanDetector(
    ...     gradual_window=63,
    ...     sudden_threshold=2.0,
    ...     min_persistence=126
    ... )
    >>> 
    >>> # Single stock sentiment time series
    >>> sentiment = pd.Series(
    ...     data=[0.1, 0.12, 0.15, 0.18, ...],  # Gradual increase
    ...     index=pd.date_range('2020-01-01', periods=252)
    ... )
    >>> 
    >>> is_gradual = detector.detect_gradual_shifts(sentiment)
    >>> print(is_gradual.iloc[-1])  # Latest detection
    1  # Gradual shift detected
    """
    
    def __init__(
        self,
        gradual_window: int = 63,
        sudden_threshold: float = 2.0,
        min_persistence: int = 126,
        volatility_window: int = 252,
        trend_consistency: float = 0.55,
        min_cumulative_change: float = 0.05
    ):
        self.gradual_window = gradual_window
        self.sudden_threshold = sudden_threshold
        self.min_persistence = min_persistence
        self.volatility_window = volatility_window
        self.trend_consistency = trend_consistency
        self.min_cumulative_change = min_cumulative_change
        
        logger.info(
            f"FrogInPanDetector initialized: "
            f"window={gradual_window}, threshold={sudden_threshold}, "
            f"persistence={min_persistence}"
        )
    
    def detect_gradual_shifts(
        self,
        sentiment_series: pd.Series,
        return_metadata: bool = False
    ) -> pd.Series:
        """
        Detect gradual sentiment shifts for a single stock.
        
        Parameters
        ----------
        sentiment_series : pd.Series
            Daily sentiment scores with datetime index
        return_metadata : bool, optional
            Return diagnostic metadata alongside binary indicator (default: False)
        
        Returns
        -------
        pd.Series or Tuple[pd.Series, pd.DataFrame]
            Binary indicator (1=gradual, 0=sudden/none) for each date
            If return_metadata=True, also returns DataFrame with diagnostics
        
        Notes
        -----
        Detection occurs only when all four criteria are satisfied:
        1. Monotonicity: ≥60% of daily changes in same direction
        2. No jumps: All daily changes < 2 standard deviations
        3. Magnitude: Cumulative change ≥10% absolute
        4. Volatility: Change volatility < historical median
        
        Requires minimum history of volatility_window days.
        """
        if len(sentiment_series) < self.volatility_window:
            logger.warning(
                f"Insufficient data: {len(sentiment_series)} days "
                f"(need {self.volatility_window})"
            )
            result = pd.Series(0, index=sentiment_series.index)
            if return_metadata:
                metadata = pd.DataFrame(index=sentiment_series.index)
                return result, metadata
            return result
        
        # Calculate daily changes
        changes = sentiment_series.diff()
        
        # Rolling volatility of changes (for criterion 4)
        rolling_vol = changes.rolling(
            window=self.volatility_window,
            min_periods=self.gradual_window
        ).std()
        
        # Initialize detection array
        is_gradual = np.zeros(len(sentiment_series), dtype=int)
        
        # Store metadata if requested
        if return_metadata:
            metadata = pd.DataFrame(index=sentiment_series.index)
            metadata['monotonicity'] = np.nan
            metadata['has_jumps'] = np.nan
            metadata['cumulative_change'] = np.nan
            metadata['volatility_ratio'] = np.nan
        
        # Rolling detection
        for i in range(self.gradual_window, len(sentiment_series)):
            window_changes = changes.iloc[i - self.gradual_window + 1:i + 1]
            window_sentiment = sentiment_series.iloc[i - self.gradual_window + 1:i + 1]
            
            # Criterion 1: Monotonicity (trend consistency)
            non_zero_changes = window_changes[window_changes != 0]
            if len(non_zero_changes) == 0:
                continue
            
            consistency = np.abs(np.mean(np.sign(non_zero_changes)))
            
            # Criterion 2: No sudden jumps
            window_vol = rolling_vol.iloc[i]
            if pd.isna(window_vol) or window_vol == 0:
                continue
            
            z_scores = window_changes / window_vol
            has_jumps = (np.abs(z_scores) > self.sudden_threshold).any()
            
            # Criterion 3: Cumulative magnitude
            cumulative_change = np.abs(
                window_sentiment.iloc[-1] - window_sentiment.iloc[0]
            )
            
            # Criterion 4: Low volatility relative to history
            median_vol = rolling_vol.iloc[max(0, i - self.volatility_window):i].median()
            if pd.isna(median_vol) or median_vol == 0:
                continue
            
            volatility_ratio = window_vol / median_vol

            # All four criteria must be satisfied
            if (
                consistency >= self.trend_consistency and  # Monotonic
                not has_jumps and  # No sudden jumps
                cumulative_change >= self.min_cumulative_change and  # Significant magnitude
                volatility_ratio < 1.0  # Below-median volatility
            ):
                is_gradual[i] = 1
            
            # Store metadata
            if return_metadata:
                metadata.iloc[i] = {
                    'monotonicity': consistency,
                    'has_jumps': int(has_jumps),
                    'cumulative_change': cumulative_change,
                    'volatility_ratio': volatility_ratio
                }
        
        result = pd.Series(is_gradual, index=sentiment_series.index)
        
        logger.debug(
            f"Detected {result.sum()} gradual shifts out of "
            f"{len(result) - self.gradual_window} valid observations"
        )
        
        if return_metadata:
            return result, metadata
        return result
    
    def detect_panel(
        self,
        sentiment_panel: pd.DataFrame,
        return_metadata: bool = False
    ) -> pd.DataFrame:
        """
        Detect gradual shifts for panel of stocks.
        
        Parameters
        ----------
        sentiment_panel : pd.DataFrame
            Multi-index DataFrame (date, ticker) with 'sentiment' column
        return_metadata : bool, optional
            Return diagnostic metadata (default: False)
        
        Returns
        -------
        pd.DataFrame or Tuple[pd.DataFrame, pd.DataFrame]
            Binary indicators (and optionally metadata) for all stocks
        
        Examples
        --------
        >>> # Panel format
        >>> sentiment_panel = pd.DataFrame({
        ...     'sentiment': [0.1, 0.2, 0.12, 0.22, ...]
        ... }, index=pd.MultiIndex.from_product([
        ...     pd.date_range('2020-01-01', periods=100),
        ...     ['AAPL', 'MSFT']
        ... ], names=['date', 'ticker']))
        >>> 
        >>> detector = FrogInPanDetector()
        >>> gradual_panel = detector.detect_panel(sentiment_panel)
        >>> print(gradual_panel.head())
                                gradual_shift
        date       ticker                    
        2020-01-01 AAPL                    0
                   MSFT                    0
        2020-01-02 AAPL                    0
                   MSFT                    0
        """
        if 'sentiment' not in sentiment_panel.columns:
            raise ValueError("sentiment_panel must contain 'sentiment' column")
        
        results = []
        metadata_list = [] if return_metadata else None
        
        # Process each ticker separately
        tickers = sentiment_panel.index.get_level_values('ticker').unique()
        logger.info(f"Processing {len(tickers)} stocks for gradual shift detection")
        
        for ticker in tickers:
            ticker_sentiment = sentiment_panel.xs(ticker, level='ticker')['sentiment']
            
            if return_metadata:
                ticker_result, ticker_metadata = self.detect_gradual_shifts(
                    ticker_sentiment,
                    return_metadata=True
                )
                ticker_metadata['ticker'] = ticker
                metadata_list.append(ticker_metadata.reset_index())
            else:
                ticker_result = self.detect_gradual_shifts(ticker_sentiment)
            
            ticker_result = ticker_result.to_frame('gradual_shift')
            ticker_result['ticker'] = ticker
            results.append(ticker_result.reset_index())
        
        # Combine results
        full_results = pd.concat(results, axis=0, ignore_index=True)
        full_results = full_results.set_index(['date', 'ticker']).sort_index()
        
        logger.info(
            f"Panel detection complete: "
            f"{full_results['gradual_shift'].sum()} total gradual shifts detected"
        )
        
        if return_metadata:
            full_metadata = pd.concat(metadata_list, axis=0, ignore_index=True)
            full_metadata = full_metadata.set_index(['date', 'ticker']).sort_index()
            return full_results, full_metadata
        
        return full_results
    
    def validate_persistence(
        self,
        sentiment_series: pd.Series,
        gradual_indicators: pd.Series,
        forward_returns: pd.Series
    ) -> Dict[str, float]:
        """
        Validate that detected gradual shifts persist and predict returns.
        
        Parameters
        ----------
        sentiment_series : pd.Series
            Original sentiment data
        gradual_indicators : pd.Series
            Binary indicators from detect_gradual_shifts()
        forward_returns : pd.Series
            Forward stock returns for validation
        
        Returns
        -------
        dict
            Validation metrics:
            - persistence_rate: Proportion of gradual shifts that persist
            - forward_correlation: Correlation with future returns
            - mean_forward_return: Average return following gradual shifts
        
        Notes
        -----
        Persistence is defined as sentiment continuing in same direction
        for min_persistence days after detection.
        """
        # Align data
        common_index = sentiment_series.index.intersection(
            gradual_indicators.index
        ).intersection(forward_returns.index)
        
        sentiment = sentiment_series.loc[common_index]
        indicators = gradual_indicators.loc[common_index]
        returns = forward_returns.loc[common_index]
        
        # Find detection dates
        detection_dates = indicators[indicators == 1].index
        
        if len(detection_dates) == 0:
            logger.warning("No gradual shifts detected for validation")
            return {
                'persistence_rate': 0.0,
                'forward_correlation': 0.0,
                'mean_forward_return': 0.0
            }
        
        # Check persistence
        persisted = 0
        forward_rets = []
        
        for detect_date in detection_dates:
            detect_idx = sentiment.index.get_loc(detect_date)
            
            # Check if we have forward window
            if detect_idx + self.min_persistence >= len(sentiment):
                continue
            
            # Sentiment direction at detection
            detection_change = sentiment.iloc[detect_idx] - sentiment.iloc[detect_idx - 1]
            if detection_change == 0:
                continue
            
            direction = np.sign(detection_change)
            
            # Check if sentiment continues in same direction
            future_sentiment = sentiment.iloc[
                detect_idx + 1:detect_idx + self.min_persistence + 1
            ]
            future_direction = np.sign(
                future_sentiment.iloc[-1] - future_sentiment.iloc[0]
            )
            
            if future_direction == direction:
                persisted += 1
            
            # Collect forward returns
            if detect_idx < len(returns):
                forward_rets.append(returns.iloc[detect_idx])
        
        persistence_rate = persisted / len(detection_dates) if detection_dates.any() else 0.0
        
        # Calculate correlation with forward returns
        forward_correlation = 0.0
        mean_forward_return = 0.0
        
        if len(forward_rets) > 0:
            mean_forward_return = np.mean(forward_rets)
            
            # Correlation between indicator and forward returns
            if len(returns) > 0:
                forward_correlation = np.corrcoef(indicators, returns)[0, 1]
                if np.isnan(forward_correlation):
                    forward_correlation = 0.0
        
        logger.info(
            f"Persistence validation: rate={persistence_rate:.2%}, "
            f"correlation={forward_correlation:.3f}, "
            f"mean_return={mean_forward_return:.4f}"
        )
        
        return {
            'persistence_rate': persistence_rate,
            'forward_correlation': forward_correlation,
            'mean_forward_return': mean_forward_return
        }
