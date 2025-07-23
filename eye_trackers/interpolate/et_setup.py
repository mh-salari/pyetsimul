"""Interpolation eye tracker.

Pupil-CR tracker with biquadratic interpolation.
"""

import numpy as np
from et_simul.core import Camera, Light, EyeTracker
from .polynomials import get_polynomial
from .calibration import calibrate_eye_tracker
from .prediction import predict_gaze_position


class InterpolationTracker(EyeTracker):
    """Pupil-CR eye tracker using biquadratic interpolation.

    This eye tracker uses the pupil-corneal reflection vector and maps it
    to screen coordinates using biquadratic interpolation.
    """

    @classmethod
    def setup(cls, polynomial="cerrolaza_2008"):
        """Create interpolation eye tracker setup.

        Args:
            polynomial: Polynomial type to use (default: 'cerrolaza_2008')

        Returns:
            InterpolationTracker: Configured interpolation eye tracker
        """
        # Lines 27-32: Create the camera
        cam = Camera(err=0.0, err_type="gaussian")
        cam.trans[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        cam.rest_trans = cam.trans.copy()
        cam.point_at(np.array([0, 550e-3, 350e-3, 1]))

        # Lines 35-36: Create lights
        light = Light(position=np.array([200e-3, 0, 350e-3, 1]))

        # Lines 40-49: Calibration points
        calib_points = np.array(
            [
                [-200e-3, 50e-3],
                [0, 50e-3],
                [200e-3, 50e-3],
                [-200e-3, 200e-3],
                [0, 200e-3],
                [200e-3, 200e-3],
                [-200e-3, 350e-3],
                [0, 350e-3],
                [200e-3, 350e-3],
            ]
        ).T

        tracker = cls(cameras=[cam], lights=[light], calib_points=calib_points)
        tracker.polynomial_name = polynomial
        tracker.polynomial_func = get_polynomial(polynomial)
        return tracker

    def calibrate(self, calib_data):
        """Calibration function for pupil-CR interpolation.

        Automatically detects 1D vs 2D polynomials based on feature shape.

        Args:
            calib_data: Calibration data from run_calibration
        """
        calibrate_eye_tracker(self, calib_data)

    def predict_gaze(self, camimg):
        """Predict gaze position from pupil-corneal reflection vector.

        Uses the calibrated polynomial model to predict screen gaze coordinates
        from the pupil-CR vector in the camera image.

        Args:
            camimg: Camera image data containing pupil and corneal reflection positions

        Returns:
            2D gaze position [x, y] on screen or None if prediction fails
        """
        return predict_gaze_position(self, camimg)
