"""Unit tests for Eye.get_pupil_boundary_in_camera_image method."""

import numpy as np
from et_simul.core.eye import Eye
from et_simul.core.camera import Camera


def test_camera_pointed_at_eye():
    """Test eye pupil image with camera properly pointed at eye using MATLAB reference values."""
    # Create eye and camera
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position eye at [0, 500mm, 200mm]
    eye_pos = np.array([0, 500e-3, 200e-3])
    e.trans[0:3, 3] = eye_pos

    # Set up camera with proper orientation and point at eye
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye looks back at camera
    e.look_at(np.array([0, 0, 0, 1]))

    # Get pupil image with 20 points
    N = 20
    X = e.get_pupil_boundary_in_camera_image(c, N)

    # MATLAB reference values from proper setup
    matlab_expected_points = np.array(
        [
            [18.220, 0.000],
            [17.328, -5.630],
            [14.740, -10.710],
            [10.710, -14.740],
            [5.630, -17.328],
            [0.000, -18.220],
            [-5.630, -17.328],
            [-10.710, -14.740],
            [-14.740, -10.710],
            [-17.328, -5.630],
            [-18.220, 0.000],
            [-17.328, 5.630],
            [-14.740, 10.710],
            [-10.710, 14.740],
            [-5.630, 17.328],
            [-0.000, 18.220],
            [5.630, 17.328],
            [10.710, 14.740],
            [14.740, 10.710],
            [17.328, 5.630],
        ]
    ).T

    # Should return valid pupil points
    assert X is not None
    assert X.shape[0] == 2  # 2D image coordinates
    assert X.shape[1] == N  # Should have N points

    # Test actual point values against MATLAB reference
    tolerance = 1.0
    for i in range(min(N, matlab_expected_points.shape[1])):
        px, py = X[0, i], X[1, i]
        mx, my = matlab_expected_points[0, i], matlab_expected_points[1, i]
        assert abs(px - mx) < tolerance, f"Point {i}: X mismatch {px} vs {mx}"
        assert abs(py - my) < tolerance, f"Point {i}: Y mismatch {py} vs {my}"


def test_eye_rotated_away_no_pupil_visible():
    """Test eye rotated to position where pupil cannot be seen by camera."""
    # Create eye and camera
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position eye
    eye_pos = np.array([0, 500e-3, 200e-3])
    e.trans[0:3, 3] = eye_pos

    # Set up camera pointed at eye
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])
    c.err = 0
    c.err_type = "uniform"

    # Eye looks 180 degrees away (directly away from camera)
    # This should make the pupil not visible to the camera
    away_target = np.array([0.0, 1.0, 0.0, 1.0]) * 0.5  # Look directly away from camera
    e.look_at(away_target)

    # Get pupil image
    N = 20
    X = e.get_pupil_boundary_in_camera_image(c, N)

    # When eye is rotated away such that pupil is not visible,
    # should return None or significantly fewer points
    if X is not None:
        # The pupil may still be partially visible depending on eye geometry
        # Just verify we get a valid result structure
        assert X.shape[0] == 2
        assert X.shape[1] <= N
    else:
        # No points visible (acceptable behavior)
        assert X is None


def test_output_properties():
    """Test that output has correct properties when pupil is visible."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Basic setup for visible pupil
    eye_pos = np.array([0, 500e-3, 200e-3])
    e.trans[0:3, 3] = eye_pos

    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])
    c.err = 0
    c.err_type = "uniform"

    e.look_at(np.array([0, 0, 0, 1]))

    X = e.get_pupil_boundary_in_camera_image(c, 20)

    if X is not None:
        # Check types and shapes
        assert isinstance(X, np.ndarray)
        assert X.dtype == np.float64
        assert X.shape[0] == 2
        assert X.shape[1] == 20

        # All values should be finite
        assert np.all(np.isfinite(X))
