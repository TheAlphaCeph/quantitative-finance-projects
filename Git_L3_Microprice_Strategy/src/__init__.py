"""
L3 Microprice Strategy - High-frequency trading strategy based on gamma-weighted microprice

A production-quality implementation of microprice-based HFT strategy with:
- Gamma-weighted microprice computation
- Multi-timeframe Order Flow Imbalance (OFI)
- Walk-forward parameter optimization
- Comprehensive backtesting framework

This is a portfolio demonstration - no proprietary data dependencies.
"""

__version__ = "1.0.0"
__author__ = "Abhay"

from .strategy.gamma_microprice_strategy import GammaMicropriceStrategy
from .backtesting.backtest_engine import BacktestEngine
from .backtesting.performance_metrics import PerformanceMetrics

__all__ = [
    'GammaMicropriceStrategy',
    'BacktestEngine',
    'PerformanceMetrics'
]
