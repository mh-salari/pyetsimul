"""Performance analysis API for eye tracking systems.

Exports functions to analyze accuracy and robustness across gaze points, observer positions, and calibration quality.
"""

from .gaze_movement import accuracy_over_gaze_points
from .eye_movement import accuracy_over_eye_positions
from .calibration_analysis import accuracy_at_calibration_points

__all__ = ["accuracy_over_gaze_points", "accuracy_over_eye_positions", "accuracy_at_calibration_points"]
