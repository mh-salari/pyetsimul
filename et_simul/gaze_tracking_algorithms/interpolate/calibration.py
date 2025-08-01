"""Calibration module for interpolation eye tracker.

Handles polynomial regression calibration for both 1D and 2D polynomial models.
"""

import numpy as np
from typing import List, Dict, Any


def calibrate(eye_tracker, calib_data: List[Dict[str, Any]]) -> None:
    """Calibrate the interpolation eye tracker using polynomial regression.

    Automatically detects 1D vs 2D polynomials based on feature shape and
    performs appropriate least squares calibration.

    Args:
        eye_tracker: InterpolationTracker instance
        calib_data: Calibration data from et_calib
    """
    # Test polynomial shape with dummy data to determine type
    test_features = eye_tracker.polynomial_func(0.0, 0.0)

    if test_features.ndim == 2:
        _calibrate_2d(eye_tracker, calib_data)
    else:
        _calibrate_1d(eye_tracker, calib_data)


def _calibrate_1d(eye_tracker, calib_data: List[Dict[str, Any]]) -> None:
    """Calibrate with 1D polynomial (shared features for x and y)."""

    # Get feature size from first valid calibration point
    feature_size = None
    for i in range(eye_tracker.calib_points.shape[1]):
        pc = calib_data[i]["camimg"][0]["pc"]
        cr = calib_data[i]["camimg"][0]["cr"][0]
        if pc is not None and cr is not None:
            pcr = pc - cr
            test_features = eye_tracker.polynomial_func(pcr[0], pcr[1])
            feature_size = len(test_features)
            break

    if feature_size is None:
        raise ValueError("No valid calibration data found")

    X = np.zeros((feature_size, eye_tracker.calib_points.shape[1]))

    for i in range(eye_tracker.calib_points.shape[1]):
        pc = calib_data[i]["camimg"][0]["pc"]
        cr = calib_data[i]["camimg"][0]["cr"][0]

        if pc is not None and cr is not None:
            pcr = pc - cr
            feature_vec = eye_tracker.polynomial_func(pcr[0], pcr[1])
            X[:, i] = feature_vec

    # Single calibration matrix for both x and y coordinates
    # Solves: calib_points = A @ X using least squares
    eye_tracker.state["A"] = eye_tracker.calib_points @ np.linalg.pinv(X)


def _calibrate_2d(eye_tracker, calib_data: List[Dict[str, Any]]) -> None:
    """Calibrate with 2D polynomial (separate features for x and y)."""
    # Get feature matrix size from first valid calibration point
    feature_matrix = None
    for i in range(eye_tracker.calib_points.shape[1]):
        pc = calib_data[i]["camimg"][0]["pc"]
        cr = calib_data[i]["camimg"][0]["cr"][0]
        if pc is not None and cr is not None:
            pcr = pc - cr
            feature_matrix = eye_tracker.polynomial_func(pcr[0], pcr[1])
            break

    if feature_matrix is None:
        raise ValueError("No valid calibration data found")

    # Separate feature matrices for x and y
    num_coords, feature_size = feature_matrix.shape  # 2 coordinates, N features each
    X_x = np.zeros((feature_size, eye_tracker.calib_points.shape[1]))
    X_y = np.zeros((feature_size, eye_tracker.calib_points.shape[1]))

    for i in range(eye_tracker.calib_points.shape[1]):
        pc = calib_data[i]["camimg"][0]["pc"]
        cr = calib_data[i]["camimg"][0]["cr"][0]

        if pc is not None and cr is not None:
            pcr = pc - cr
            feature_matrix = eye_tracker.polynomial_func(pcr[0], pcr[1])
            X_x[:, i] = feature_matrix[0, :]  # Features for x coordinate
            X_y[:, i] = feature_matrix[1, :]  # Features for y coordinate

    # Separate calibration matrices for x and y coordinates
    # Solve separate least squares problems for each coordinate
    eye_tracker.state["A_x"] = eye_tracker.calib_points[0:1, :] @ np.linalg.pinv(X_x)  # x coordinate only
    eye_tracker.state["A_y"] = eye_tracker.calib_points[1:2, :] @ np.linalg.pinv(X_y)  # y coordinate only
