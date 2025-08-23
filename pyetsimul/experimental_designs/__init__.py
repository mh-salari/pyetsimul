"""Experimental designs for PyEtSimul eye tracking simulation.

This package provides standardized experimental designs for both data generation
and evaluation. Designs specify the structure and parameters of experiments.
"""

from .base import ExperimentalDesignBase
from .gaze_movement import GazeMovement
from .eye_movement import EyeMovement

__all__ = ["ExperimentalDesignBase", "GazeMovement", "EyeMovement"]
