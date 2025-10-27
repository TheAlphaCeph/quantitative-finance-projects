"""
Unit tests for FrogInPanDetector
"""

import pytest
import numpy as np
import pandas as pd
from src.signals import FrogInPanDetector


def test_detector_initialization():
    """Test detector initializes with correct parameters."""
    detector = FrogInPanDetector(
        gradual_window=63,
        sudden_threshold=2.0
    )
    assert detector.gradual_window == 63
    assert detector.sudden_threshold == 2.0


def test_gradual_shift_detection():
    """Test detector identifies gradual shifts correctly."""
    # Create synthetic gradual shift
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    sentiment = pd.Series(
        np.linspace(0, 0.2, 300),  # Gradual increase
        index=dates
    )
    
    detector = FrogInPanDetector()
    result = detector.detect_gradual_shifts(sentiment)
    
    # Should detect gradual shift in later periods
    assert result.sum() > 0
    assert len(result) == len(sentiment)


def test_sudden_jump_rejection():
    """Test detector rejects sudden jumps."""
    dates = pd.date_range('2020-01-01', periods=300, freq='D')
    sentiment = pd.Series(0.0, index=dates)
    sentiment.iloc[150] = 0.5  # Sudden jump
    
    detector = FrogInPanDetector()
    result = detector.detect_gradual_shifts(sentiment)
    
    # Should not detect gradual shift due to jump
    assert result.iloc[150:200].sum() == 0
