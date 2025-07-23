"""Hennessey eye tracker.

Eye tracker that uses Hennessey et al.'s method.
"""

import numpy as np
from et_simul.core import Camera, Light, EyeTracker
from .calibrate import hennessey_calib
from .evaluate import hennessey_eval_main


class HennesseyTracker(EyeTracker):
    """Eye tracker that uses Hennessey et al.'s method.

    Based on [1]. The eye tracker uses a single camera and dual light sources
    with various recalibration methods and pupil algorithms.

    [1] Craig Hennessey, Borna Noureddin, Peter Lawrence. A Single Camera
        Eye-Gaze Tracking System with Free Head Motion. ETRA 2006.
    """

    @classmethod
    def setup(cls, params=None):
        """Create an eye tracker that uses Hennessey et al.'s method.

         This function is based on the original MATLAB implementation from the
        et_simul project — © 2008 Martin Böhme, University of Lübeck.
        Python port © 2025 Mohammadhossein Salari.
        Licensed under the GNU GPL v3.0 or later.

        Args:
            params: Dictionary containing one or more of the following fields;
                   default values are used for fields that are not specified
                   or if no 'params' argument is passed in.

                   - 'recalib_type' is the type of recalibration to use. Valid values are
                     'angle', 'henn_angle', 'henn3d', 'hennessey', and 'screen'. The
                     recalibration types are described in detail in the documentation for
                     the recalibration routines recalib_<r>_calib() (where <r> is the name of
                     the recalibration type).

                   - 'pupil_alg' is the algorithm to use for estimating the pupil center.
                     Valid values are: 'hennessey' (uses the original algorithm from the
                     paper, where individual rays to pupil contour points are traced back
                     into the eye) and 'pupil_center' (where the pupil center is determined
                     by fitting an ellipse to the pupil in the image and a single ray to
                     this pupil center is traced back into the ray)

                   - 'err' and 'err_type' are the amount and type of camera error, specified
                     as described in camera_make().

                   - 'parameter_err' is the amount of relative error in the eye model
                     parameters assumed by the algorithm relative to their true values in
                     the simulated eye.

        Returns:
            HennesseyTracker: Configured Hennessey eye tracker
        """
        # If no parameters passed, use defaults
        if params is None:
            params = {}

        # Default parameters
        default_params = {
            "recalib_type": "hennessey",
            "pupil_alg": "hennessey",
            "err": 0.0,
            "err_type": "uniform",
            "parameter_err": 0.0,
        }

        # Obtain parameter values by overwriting those fields in the default
        # parameters that were also specified in 'params'
        merged_params = default_params.copy()
        merged_params.update(params)

        # Create camera
        cam = Camera(err=merged_params["err"], err_type=merged_params["err_type"])
        cam.trans[:3, :3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        cam.rest_trans = cam.trans.copy()
        cam.point_at(np.array([0, 550e-3, 350e-3, 1]))

        # Create lights (on vertical edge of monitor)
        light1 = Light(position=np.array([200e-3, 0, 50e-3, 1]))
        light2 = Light(position=np.array([200e-3, 0, 300e-3, 1]))

        if True:
            # Nine-point calibration pattern
            calib_points = np.array(
                [
                    [0, 200e-3],  # Point 1
                    [-150e-3, 50e-3],  # Point 2
                    [-150e-3, 350e-3],  # Point 3
                    [150e-3, 50e-3],  # Point 4
                    [150e-3, 350e-3],  # Point 5
                    [-100e-3, 150e-3],  # Point 6
                    [-100e-3, 250e-3],  # Point 7
                    [100e-3, 150e-3],  # Point 8
                    [100e-3, 250e-3],  # Point 9
                ]
            ).T
        else:
            # Calibration points are just the corners of the screen
            calib_points = np.array(
                [
                    [-200e-3, 50e-3],  # Point 1
                    [200e-3, 50e-3],  # Point 2
                    [-200e-3, 350e-3],  # Point 3
                    [200e-3, 350e-3],  # Point 4
                ]
            ).T

        # Create tracker instance
        tracker = cls(cameras=[cam], lights=[light1, light2], calib_points=calib_points)

        # Set state parameters
        tracker.state["recalib_type"] = merged_params["recalib_type"]
        tracker.state["pupil_alg"] = merged_params["pupil_alg"]
        tracker.state["parameter_err"] = merged_params["parameter_err"]

        return tracker

    def calibrate(self, calib_data):
        """Calibration function for Hennessey method.

        Based on hennessey_calib.m
        """

        hennessey_calib(self, calib_data)

    def predict_gaze(self, camimg):
        """Predict gaze position for Hennessey method.

        Based on hennessey_eval.m and hennessey_eval_base.m
        """
        return hennessey_eval_main(self, camimg)
