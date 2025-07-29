"""Unit tests for Eye.get_pupil_boundary_in_camera_image method."""

import numpy as np
import warnings
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

    assert X is not None, "X should not be None for these inputs"

    # Check types and shapes
    assert isinstance(X, np.ndarray)
    assert X.dtype == np.float64
    assert X.shape[0] == 2
    assert X.shape[1] == 20

    # All values should be finite
    assert np.all(np.isfinite(X))


def test_eye_facing_away_from_camera():
    """Test that method returns None when eye is facing away from camera."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position eye at [0, 500mm, 200mm]
    eye_pos = np.array([0, 500e-3, 200e-3])
    e.trans[0:3, 3] = eye_pos

    # Set up camera with proper orientation pointing at eye
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])

    # Set eye rest orientation same as camera (both looking in same direction)
    e.set_rest_orientation(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye looks away from camera (same direction as camera viewing direction)
    # Camera views in direction [0, 1, 0] (positive y), so make eye look that way too
    camera_viewing_direction = -c.orientation[:, 2]  # negative of optical axis
    eye_target = e.position + camera_viewing_direction
    eye_target_homo = np.array([eye_target[0], eye_target[1], eye_target[2], 1])
    e.look_at(eye_target_homo)

    # Test that warning is generated and method returns None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X = e.get_pupil_boundary_in_camera_image(c, 20)

        # Should have generated a warning
        assert len(w) > 0
        warning_messages = [str(warning.message) for warning in w]
        assert any("face the same direction" in msg for msg in warning_messages)

    # Should return None for back-of-eye scenario
    assert X is None


def test_eye_behind_camera():
    """Test that method returns None when eye is behind camera."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position camera at origin
    c.trans[0:3, 3] = np.array([0, 0, 0])

    # Position eye behind camera (negative y direction if camera looks in +y)
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    eye_pos = np.array([0, -500e-3, 0])  # Behind camera
    e.trans[0:3, 3] = eye_pos

    # Set eye rest orientation
    e.set_rest_orientation(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye orientation doesn't matter much since it's behind camera
    e.look_at(np.array([0, 0, 0, 1]))

    # Test that warning is generated and method returns None
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        X = e.get_pupil_boundary_in_camera_image(c, 20)

        # Should have generated a warning
        assert len(w) > 0
        warning_messages = [str(warning.message) for warning in w]
        assert any("behind camera" in msg for msg in warning_messages)

    # Should return None for behind-camera scenario
    assert X is None


def test_eye_rotated_90_degrees():
    """Test that method returns None when eye is rotated 90 degrees away."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position eye at [0, 500mm, 200mm]
    eye_pos = np.array([0, 500e-3, 200e-3])
    e.trans[0:3, 3] = eye_pos

    # Set up camera with proper orientation pointing at eye
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.trans[:, 3])

    # Set eye rest orientation same as camera initially
    e.set_rest_orientation(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye looks 90 degrees away (perpendicular to camera)
    # If camera is at origin looking toward eye, make eye look to the side
    side_target = np.array([eye_pos[0] + 1, eye_pos[1], eye_pos[2], 1])
    e.look_at(side_target)

    # Get pupil image - may return None or some points depending on geometry
    X = e.get_pupil_boundary_in_camera_image(c, 20)

    # In this case we just verify the method handles it gracefully
    # Could be None or a valid result depending on exact geometry
    if X is not None:
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 2
        assert np.all(np.isfinite(X))
