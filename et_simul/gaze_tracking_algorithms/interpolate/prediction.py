"""Gaze prediction module for interpolation eye tracker.

Predicts gaze coordinates from pupil-corneal reflection vectors using
calibrated polynomial regression models.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from ...types import Point3D, Point4D


@dataclass
class PredictionResult:
    """Detailed prediction result containing all intermediate values.

    Attributes:
        gaze_point: Predicted gaze position [x, y] or None if prediction failed
        pc: Pupil center coordinates in camera image
        cr: Corneal reflection coordinates in camera image
        pcr_vector: PC-CR difference vector used for prediction
        polynomial_name: Name of polynomial used for prediction
        feature_vector: Computed polynomial features
        prediction_successful: Whether prediction was successful
    """

    gaze_point: Optional[Point3D] = None
    pc: Optional[Point3D] = None
    cr: Optional[Point3D] = None
    pcr_vector: Optional[Point3D] = None
    polynomial_name: Optional[str] = None
    feature_vector: Optional[Point4D] = None
    prediction_successful: bool = False

    @property
    def gaze(self) -> Optional[Point3D]:
        """Convenience property to access gaze point."""
        return self.gaze_point

    def __bool__(self) -> bool:
        """Return True if prediction was successful."""
        return self.prediction_successful


def predict(eye_tracker, camimg) -> PredictionResult:
    """Predict gaze from pupil-corneal reflection vector.

    Uses the calibrated polynomial model to predict screen gaze coordinates
    from the pupil-CR vector in the camera image.

    Args:
        eye_tracker: Calibrated InterpolationTracker instance
        camimg: Camera image data containing pupil and corneal reflection positions

    Returns:
        PredictionResult: Detailed prediction result with all intermediate values.
    """
    result = PredictionResult()

    # Extract PC and CR from camera image
    result.pc = camimg[0]["pc"]
    result.cr = camimg[0]["cr"][0] if camimg[0]["cr"] else None
    result.polynomial_name = getattr(eye_tracker, "polynomial_name", "unknown")

    if result.pc is not None and result.cr is not None:
        result.pcr_vector = result.pc - result.cr

        # Test polynomial shape to determine type
        test_features = eye_tracker.polynomial_func(result.pcr_vector[0], result.pcr_vector[1])
        result.feature_vector = test_features

        if test_features.ndim == 2:
            result.gaze_point = _predict_2d(eye_tracker, result.pcr_vector)
        else:
            result.gaze_point = _predict_1d(eye_tracker, result.pcr_vector)

        result.prediction_successful = result.gaze_point is not None

    return result


def _predict_1d(eye_tracker, pcr):
    """Predict with 1D polynomial (shared features for x and y)."""
    feature_vector = eye_tracker.polynomial_func(pcr[0], pcr[1])
    gaze = eye_tracker.state["A"] @ feature_vector
    return gaze


def _predict_2d(eye_tracker, pcr):
    """Predict with 2D polynomial (separate features for x and y)."""
    feature_matrix = eye_tracker.polynomial_func(pcr[0], pcr[1])

    # Extract separate feature vectors for x and y coordinates
    x_features = feature_matrix[0, :]  # Features for x coordinate
    y_features = feature_matrix[1, :]  # Features for y coordinate

    # Apply separate calibration matrices
    gaze_x = eye_tracker.state["A_x"] @ x_features
    gaze_y = eye_tracker.state["A_y"] @ y_features

    return np.array([gaze_x[0], gaze_y[0]])
