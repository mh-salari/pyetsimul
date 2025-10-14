"""Gaze tracking algorithm implementations.

Provides different gaze tracking algorithms:
- polynomial: Polynomial-based gaze tracking
- homography: Homography-based gaze tracking

Each algorithm implements the EyeTracker interface for calibration and gaze estimation.
"""

from .homography import HomographyGazeModel
from .polynomial import PolynomialGazeModel

__all__ = ["HomographyGazeModel", "PolynomialGazeModel"]
