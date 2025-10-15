"""Gaze tracking algorithm implementations.

Provides different gaze tracking algorithms:
- polynomial: Polynomial-based gaze tracking
- homography_normalization: Homography normalization-based gaze tracking

Each algorithm implements the EyeTracker interface for calibration and gaze estimation.
"""

from .homography_normalization import HomographyNormalizationGazeModel
from .polynomial import PolynomialGazeModel

__all__ = ["HomographyNormalizationGazeModel", "PolynomialGazeModel"]
