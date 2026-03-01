"""Gaze tracking algorithm implementations.

Provides different gaze tracking algorithms:
- polynomial: Polynomial-based gaze tracking
- homography_normalization: Homography normalization-based gaze tracking
- stampe1993: Stampe (1993) biquadratic polynomial with corner correction
- eyelink1000plus: EyeLink 1000 Plus with HREF preprocessing + Stampe (1993)

Each algorithm implements the EyeTracker interface for calibration and gaze estimation.
"""

from .eyelink1000plus import EyeLink1000PlusGazeModel
from .homography_normalization import HomographyNormalizationGazeModel
from .polynomial import PolynomialGazeModel
from .stampe1993 import Stampe1993GazeModel

__all__ = [
    "EyeLink1000PlusGazeModel",
    "HomographyNormalizationGazeModel",
    "PolynomialGazeModel",
    "Stampe1993GazeModel",
]
