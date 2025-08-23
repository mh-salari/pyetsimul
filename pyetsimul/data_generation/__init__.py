"""Data generation for PyEtSimul eye tracking simulation.

This package provides tools for generating synthetic eye tracking data using
GazeMovement experimental designs.
"""

from .gaze_movement import GazeMovementExperiment
from .eye_movement import EyeMovementExperiment

__all__ = ["GazeMovementExperiment", "EyeMovementExperiment"]
