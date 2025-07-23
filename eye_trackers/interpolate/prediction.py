"""Gaze prediction module for interpolation eye tracker.

Predicts gaze coordinates from pupil-corneal reflection vectors using
calibrated polynomial regression models.
"""

import numpy as np


def predict_gaze_position(eye_tracker, camimg):
    """Predict gaze position from pupil-corneal reflection vector.

    Uses the calibrated polynomial model to predict screen gaze coordinates
    from the pupil-CR vector in the camera image.

    Args:
        eye_tracker: Calibrated InterpolationTracker instance
        camimg: Camera image data containing pupil and corneal reflection positions

    Returns:
        2D gaze position [x, y] on screen or None if prediction fails
    """
    pc = camimg[0]["pc"]
    cr = camimg[0]["cr"][0]

    if pc is not None and cr is not None:
        pcr = pc - cr

        # Test polynomial shape to determine type
        test_features = eye_tracker.polynomial_func(pcr[0], pcr[1])

        if test_features.ndim == 2:
            return _predict_2d(eye_tracker, pcr)
        else:
            return _predict_1d(eye_tracker, pcr)
    else:
        return None


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
