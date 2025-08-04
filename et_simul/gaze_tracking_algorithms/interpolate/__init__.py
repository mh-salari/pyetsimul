"""Polynomial interpolation eye tracking algorithm.

Provides InterpolationTracker for polynomial-based gaze tracking.
Implements pupil-corneal reflection vector mapping to screen coordinates.
"""

from .interpolation_tracker import InterpolationTracker

__all__ = ["InterpolationTracker"]
