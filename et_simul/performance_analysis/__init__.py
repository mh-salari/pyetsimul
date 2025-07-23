"""Performance analysis module for eye tracking systems.

This module provides functions to analyze eye tracker accuracy and robustness
across different conditions:

- gaze_point_analysis: Analyzes accuracy across different gaze target points
- observer_position_analysis: Analyzes robustness to observer position changes
"""

from .gaze_point_analysis import accuracy_over_gaze_points
from .observer_position_analysis import accuracy_over_observer_positions

__all__ = [
    'accuracy_over_gaze_points',
    'accuracy_over_observer_positions'
]