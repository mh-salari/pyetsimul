"""Performance analysis API for eye tracking systems.

Exports functions to analyze accuracy and robustness across gaze points, observer positions, and calibration quality.
"""

from .gaze_point_analysis import accuracy_over_gaze_points
from .observer_position_analysis import accuracy_over_observer_positions
from .calibration_analysis import accuracy_at_calibration_points

__all__ = ["accuracy_over_gaze_points", "accuracy_over_observer_positions", "accuracy_at_calibration_points"]
