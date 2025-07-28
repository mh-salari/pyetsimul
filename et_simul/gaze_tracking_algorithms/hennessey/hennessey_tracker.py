"""Hennessey eye tracker.

Eye tracker that uses Hennessey et al.'s method.
"""

from et_simul.core import EyeTracker
from .calibrate import hennessey_calib
from .prediction import predict, PredictionResult


class HennesseyTracker(EyeTracker):
    """Eye tracker that uses Hennessey et al.'s method.

    Based on [1]. The eye tracker uses a single camera and dual light sources
    with various recalibration methods and pupil algorithms.

    [1] Craig Hennessey, Borna Noureddin, Peter Lawrence. A Single Camera
        Eye-Gaze Tracking System with Free Head Motion. ETRA 2006.
    """

    @classmethod
    def setup(cls, cameras, lights, calib_points, params=None):
        """Create an eye tracker that uses Hennessey et al.'s method.

        Args:
            cameras: List of Camera objects
            lights: List of Light objects
            calib_points: Calibration point coordinates (numpy array format)
            params: Dictionary containing optional parameters:
                   - 'recalib_type': Type of recalibration ('hennessey', 'angle', etc.)
                   - 'pupil_alg': Pupil center algorithm ('hennessey', 'pupil_center')
                   - 'parameter_err': Relative error in eye model parameters

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
            "parameter_err": 0.0,
        }

        # Merge parameters
        merged_params = default_params.copy()
        merged_params.update(params)

        # Create tracker instance with provided components
        tracker = cls(cameras=cameras, lights=lights, calib_points=calib_points)

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

    def predict_gaze(self, camimg) -> PredictionResult:
        """Predict gaze position for Hennessey method.

        Based on hennessey_eval.m and hennessey_eval_base.m

        Returns:
            PredictionResult: Detailed prediction result with all intermediate values
        """
        return predict(self, camimg)
