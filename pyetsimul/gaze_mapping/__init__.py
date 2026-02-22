"""Gaze tracking algorithm implementations.

Provides different gaze tracking algorithms:
- polynomial: Polynomial-based gaze tracking
- homography_normalization: Homography normalization-based gaze tracking
- stampe1993: Stampe (1993) biquadratic polynomial with corner correction

Each algorithm implements the EyeTracker interface for calibration and gaze estimation.
"""

from .homography_normalization import HomographyNormalizationGazeModel
from .polynomial import PolynomialGazeModel
from .stampe1993 import Stampe1993GazeModel

__all__ = ["HomographyNormalizationGazeModel", "PolynomialGazeModel", "Stampe1993GazeModel"]
