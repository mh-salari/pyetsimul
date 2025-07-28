"""Interpolation eye tracker.

Pupil-CR tracker with biquadratic interpolation.
"""

from et_simul.core import EyeTracker
from .polynomials import get_polynomial
from .calibration import calibrate
from .prediction import predict, PredictionResult
import numpy as np


class InterpolationTracker(EyeTracker):
    """Pupil-CR eye tracker using biquadratic interpolation.

    This eye tracker uses the pupil-corneal reflection vector and maps it
    to screen coordinates using biquadratic interpolation.
    """

    @classmethod
    def setup(cls, cameras, lights, calib_points, polynomial, use_refraction=True):
        """Create interpolation eye tracker setup.

        Args:
            cameras: List of Camera objects
            lights: List of Light objects
            calib_points: Calibration points list
            polynomial: Polynomial type to use
            use_refraction: Whether to use refraction model (default True)

        Returns:
            InterpolationTracker: Configured interpolation eye tracker
        """
        tracker = cls(
            cameras=cameras, lights=lights, calib_points=np.array(calib_points).T, use_refraction=use_refraction
        )
        tracker.polynomial_name = polynomial
        tracker.polynomial_func = get_polynomial(polynomial)
        return tracker

    def calibrate(self, calib_data):
        """Calibration function for pupil-CR interpolation.

        Automatically detects 1D vs 2D polynomials based on feature shape.

        Args:
            calib_data: Calibration data from run_calibration
        """
        calibrate(self, calib_data)

    def predict_gaze(self, camimg) -> PredictionResult:
        """Predict gaze from pupil-corneal reflection vector.

        Uses the calibrated polynomial model to predict screen gaze coordinates
        from the pupil-CR vector in the camera image.

        Args:
            camimg: Camera image data containing pupil and corneal reflection positions

        Returns:
            PredictionResult: Detailed prediction result with all intermediate values.
        """
        return predict(self, camimg)
