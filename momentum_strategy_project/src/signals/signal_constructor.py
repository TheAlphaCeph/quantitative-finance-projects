"""
Composite Signal Constructor

Combines three momentum sources into unified signal:
1. Price momentum (traditional 12-1 month)
2. Sentiment momentum (from NLP transcripts)
3. Frog-in-the-pan score (gradual shift indicator)

DECISION CRITERIA:
    Signal = α·Price_Mom + β·Sent_Mom + γ·Frog_Score
    
    Where α, β, γ are learned weights from optimization:
    - α = 0.40 (price momentum weight)
    - β = 0.35 (sentiment momentum weight)
    - γ = 0.25 (frog-in-the-pan weight)

INPUTS:
    - prices: Multi-index DataFrame (date, ticker) with close prices
    - sentiment: Multi-index DataFrame (date, ticker) with sentiment scores
    - frog_detector: Initialized FrogInPanDetector instance

OUTPUTS:
    - Composite signal scores (cross-sectionally normalized)

ASSUMPTIONS:
    - Monthly rebalancing frequency
    - Prices and sentiment aligned on same dates
    - Missing data handled via forward-fill with max 5-day gap

REFERENCES:
    Jegadeesh, N., & Titman, S. (1993). "Returns to Buying Winners and 
    Selling Losers." Journal of Finance, 48(1), 65-91.
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd
from loguru import logger

from .frog_in_pan_detector import FrogInPanDetector


class SignalConstructor:
    """
    Constructs composite momentum signals.
    
    Weighted combination of price momentum, sentiment momentum, and
    gradual shift detection (frog-in-the-pan).
    
    Parameters
    ----------
    price_weight : float, optional
        Weight for price momentum component (default: 0.40)
    sentiment_weight : float, optional
        Weight for sentiment momentum component (default: 0.35)
    frog_weight : float, optional
        Weight for frog-in-the-pan component (default: 0.25)
    price_lookback : int, optional
        Lookback period for price momentum in days (default: 252)
    price_skip : int, optional
        Skip period to avoid reversal (default: 21)
    sentiment_lookback : int, optional
        Lookback for sentiment momentum (default: 126)
    normalize_cross_section : bool, optional
        Apply cross-sectional normalization (default: True)
    
    Examples
    --------
    >>> constructor = SignalConstructor(
    ...     price_weight=0.40,
    ...     sentiment_weight=0.35,
    ...     frog_weight=0.25
    ... )
    >>> 
    >>> signals = constructor.construct_signals(
    ...     prices=price_data,
    ...     sentiment=sentiment_data,
    ...     frog_detector=detector
    ... )
    """
    
    def __init__(
        self,
        price_weight: float = 0.40,
        sentiment_weight: float = 0.35,
        frog_weight: float = 0.25,
        price_lookback: int = 252,
        price_skip: int = 21,
        sentiment_lookback: int = 126,
        normalize_cross_section: bool = True
    ):
        # Validate weights sum to 1
        total_weight = price_weight + sentiment_weight + frog_weight
        if not np.isclose(total_weight, 1.0):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.3f}"
            )
        
        self.price_weight = price_weight
        self.sentiment_weight = sentiment_weight
        self.frog_weight = frog_weight
        self.price_lookback = price_lookback
        self.price_skip = price_skip
        self.sentiment_lookback = sentiment_lookback
        self.normalize_cross_section = normalize_cross_section
        
        logger.info(
            f"SignalConstructor initialized: "
            f"weights=[{price_weight:.2f}, {sentiment_weight:.2f}, {frog_weight:.2f}]"
        )
    
    def calculate_price_momentum(
        self,
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate traditional 12-1 month price momentum.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Multi-index (date, ticker) with 'close' prices
        
        Returns
        -------
        pd.DataFrame
            Price momentum scores
        """
        logger.debug("Calculating price momentum")
        
        # Unstack to get panel format
        price_panel = prices['close'].unstack('ticker')
        
        # Calculate returns over lookback, skipping recent period
        past_prices = price_panel.shift(self.price_skip)
        old_prices = price_panel.shift(self.price_lookback)
        
        momentum = (past_prices - old_prices) / old_prices
        
        # Stack back to multi-index
        momentum = momentum.stack('ticker')
        
        return momentum.to_frame('price_momentum')
    
    def calculate_sentiment_momentum(
        self,
        sentiment: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate sentiment momentum (change over lookback period).
        
        Parameters
        ----------
        sentiment : pd.DataFrame
            Multi-index (date, ticker) with 'sentiment' scores
        
        Returns
        -------
        pd.DataFrame
            Sentiment momentum scores
        """
        logger.debug("Calculating sentiment momentum")
        
        # Unstack for panel operations
        sent_panel = sentiment['sentiment'].unstack('ticker')
        
        # Calculate change over lookback
        current_sent = sent_panel
        past_sent = sent_panel.shift(self.sentiment_lookback)
        
        sent_momentum = current_sent - past_sent
        
        # Stack back
        sent_momentum = sent_momentum.stack('ticker')
        
        return sent_momentum.to_frame('sentiment_momentum')
    
    def construct_signals(
        self,
        prices: pd.DataFrame,
        sentiment: pd.DataFrame,
        frog_detector: FrogInPanDetector
    ) -> pd.DataFrame:
        """
        Construct composite momentum signals.
        
        Parameters
        ----------
        prices : pd.DataFrame
            Multi-index (date, ticker) with 'close' column
        sentiment : pd.DataFrame
            Multi-index (date, ticker) with 'sentiment' column
        frog_detector : FrogInPanDetector
            Initialized detector instance
        
        Returns
        -------
        pd.DataFrame
            Composite signals with columns:
            - price_momentum
            - sentiment_momentum
            - frog_score
            - composite_signal (weighted combination)
        """
        logger.info("Constructing composite signals")
        
        # Calculate price momentum
        price_mom = self.calculate_price_momentum(prices)
        
        # Calculate sentiment momentum
        sent_mom = self.calculate_sentiment_momentum(sentiment)
        
        # Detect frog-in-the-pan gradual shifts
        frog_scores = frog_detector.detect_panel(sentiment)
        
        # Align all components
        common_index = price_mom.index.intersection(
            sent_mom.index
        ).intersection(frog_scores.index)
        
        price_mom = price_mom.loc[common_index]
        sent_mom = sent_mom.loc[common_index]
        frog_scores = frog_scores.loc[common_index]
        
        # Combine into single DataFrame
        signals = pd.DataFrame({
            'price_momentum': price_mom['price_momentum'],
            'sentiment_momentum': sent_mom['sentiment_momentum'],
            'frog_score': frog_scores['gradual_shift']
        })
        
        # Handle missing values (forward fill with 5-day limit)
        signals = signals.groupby(level='ticker').ffill(limit=5)
        
        # Cross-sectional normalization (rank-based)
        if self.normalize_cross_section:
            for col in ['price_momentum', 'sentiment_momentum', 'frog_score']:
                signals[col] = signals.groupby(level='date')[col].transform(
                    lambda x: (x.rank() - 1) / (len(x) - 1) if len(x) > 1 else 0.5
                )
        
        # Construct composite signal
        signals['composite_signal'] = (
            self.price_weight * signals['price_momentum'] +
            self.sentiment_weight * signals['sentiment_momentum'] +
            self.frog_weight * signals['frog_score']
        )
        
        logger.info(
            f"Generated signals for {len(signals)} observations "
            f"({signals.index.get_level_values('date').nunique()} dates, "
            f"{signals.index.get_level_values('ticker').nunique()} tickers)"
        )
        
        return signals
