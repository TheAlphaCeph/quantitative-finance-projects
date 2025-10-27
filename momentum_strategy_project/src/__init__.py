"""
Momentum Strategy Module

Main Components:
- interfaces: External system connections (SentimentAPI, CRSPLoader)
- signals: Core algorithms (FrogInPanDetector, SignalConstructor)
- analysis: Performance metrics and validation
- optimization: Parameter tuning
- strategy: Portfolio construction and backtesting
"""

__version__ = '1.0.0'
__author__ = 'Abhay Kanwar'

from . import interfaces
from . import signals
from . import analysis
from . import optimization
from . import strategy

__all__ = [
    'interfaces',
    'signals',
    'analysis',
    'optimization',
    'strategy'
]
