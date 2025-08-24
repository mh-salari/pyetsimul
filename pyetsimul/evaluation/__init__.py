"""Performance analysis API for eye tracking systems.

Exports functions to analyze accuracy and robustness across gaze points, observer positions, and calibration quality.
"""

from .calibration_analysis import accuracy_at_calibration_points

__all__ = ["accuracy_at_calibration_points"]
