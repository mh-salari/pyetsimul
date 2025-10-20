"""Polynomial gaze model eye tracking algorithm.

Provides PolynomialGazeModel for polynomial-based gaze tracking.
Implements pupil-corneal reflection vector mapping to screen coordinates.
"""

from .polynomial_descriptor import PolynomialDescriptor
from .polynomial_features import PolynomialFeatures
from .polynomial_gaze_model import PolynomialGazeModel
from .polynomial_state import PolynomialGazeModelState

__all__ = [
    "PolynomialDescriptor",
    "PolynomialFeatures",
    "PolynomialGazeModel",
    "PolynomialGazeModelState",
]
