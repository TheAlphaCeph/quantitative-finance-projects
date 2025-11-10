"""
Signal Generation Modules

Core algorithms for momentum strategy:
- FrogInPanDetector: Distinguishes gradual vs sudden sentiment shifts
- SignalConstructor: Combines price, sentiment, and frog-in-pan scores
"""

from .frog_in_pan_detector import FrogInPanDetector
from .signal_constructor import SignalConstructor

__all__ = ['FrogInPanDetector', 'SignalConstructor']
