"""
External System Interfaces

This module provides connections to external data sources:
- SentimentAPI: NLP sentiment analysis
- CRSPLoader: CRSP/Compustat data via WRDS

These interfaces abstract away data acquisition details from the strategy logic.
"""

from .sentiment_api import SentimentAPI
from .crsp_loader import CRSPLoader

__all__ = ['SentimentAPI', 'CRSPLoader']
