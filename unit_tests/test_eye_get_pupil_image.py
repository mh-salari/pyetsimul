"""Unit tests for Eye.get_pupil_boundary_in_camera_image method."""

import numpy as np
import pytest
from pyetsimul.core.eye import Eye
from pyetsimul.core.camera import Camera
from pyetsimul.types import Position3D, Point2D


def test_camera_pointed_at_eye():
    """Test eye pupil image with camera properly pointed at eye using MATLAB reference values."""
    # Create eye and camera
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position eye at [0, 500mm, 200mm]
    eye_pos = Position3D(x=0, y=500e-3, z=200e-3)
    e.position = eye_pos

    # Set up camera with proper orientation and point at eye
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.position)

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye looks back at camera
    e.look_at(Position3D(x=0, y=0, z=0))

    # Get pupil image with 20 points
    N = 20
    e.pupil.N = N  # Set pupil resolution
    X, _ = e.get_pupil_in_camera_image(c)

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
    assert isinstance(X, list)  # Should be List[Point2D]
    assert len(X) == N  # Should have N points

    # Test actual point values against MATLAB reference
    tolerance = 1.0
    for i in range(min(N, matlab_expected_points.shape[1])):
        px, py = X[i].x, X[i].y
        mx, my = matlab_expected_points[0, i], matlab_expected_points[1, i]
        assert abs(px - mx) < tolerance, f"Point {i}: X mismatch {px} vs {mx}"
        assert abs(py - my) < tolerance, f"Point {i}: Y mismatch {py} vs {my}"


def test_output_properties():
    """Test that output has correct properties when pupil is visible."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Basic setup for visible pupil
    eye_pos = Position3D(x=0, y=500e-3, z=200e-3)
    e.position = eye_pos

    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.position)
    c.err = 0
    c.err_type = "uniform"

    e.look_at(Position3D(x=0, y=0, z=0))

    e.pupil.N = 20  # Set pupil resolution
    X, _ = e.get_pupil_in_camera_image(c)

    assert X is not None, "X should not be None for these inputs"

    # Check types and shapes
    assert isinstance(X, list)  # Should be List[Point2D]
    assert len(X) == 20
    assert all(isinstance(p, Point2D) for p in X)

    # All values should be finite
    assert all(np.isfinite(p.x) and np.isfinite(p.y) for p in X)


def test_eye_facing_away_from_camera():
    """Test that method returns None and warns when eye is facing away from camera."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position eye at [0, 500mm, 200mm]
    eye_pos = Position3D(x=0, y=500e-3, z=200e-3)
    e.position = eye_pos

    # Set up camera with proper orientation pointing at eye
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.position)

    # Set eye rest orientation same as camera (both looking in same direction)
    e.set_rest_orientation(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye looks away from camera (same direction as camera viewing direction)
    # Camera views in direction [0, 1, 0] (positive y), so make eye look that way too
    camera_viewing_direction = -c.orientation[:, 2]  # negative of optical axis
    eye_target = Position3D(
        x=e.position.x + camera_viewing_direction[0],
        y=e.position.y + camera_viewing_direction[1],
        z=e.position.z + camera_viewing_direction[2],
    )
    e.look_at(eye_target)

    # Test that method returns None and produces expected warning
    e.pupil.N = 20  # Set pupil resolution

    # Expect warning about no refracted pupil points
    with pytest.warns(UserWarning, match="No refracted pupil points could be computed"):
        X, _ = e.get_pupil_in_camera_image(c)

    # Should return None for back-of-eye scenario
    assert X is None, "Should return None when eye faces away from camera"


def test_eye_behind_camera():
    """Test that method returns None and warns when eye is behind camera."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position camera at origin
    c.trans[0:3, 3] = np.array([0, 0, 0])

    # Position eye behind camera (negative y direction if camera looks in +y)
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    eye_pos = Position3D(x=0, y=-500e-3, z=0)  # Behind camera
    e.position = eye_pos

    # Set eye rest orientation
    e.set_rest_orientation(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye orientation doesn't matter much since it's behind camera
    e.look_at(Position3D(x=0, y=0, z=0))

    # Test that method returns None and produces expected warning
    e.pupil.N = 20  # Set pupil resolution

    # Expect warning about no valid pupil points found
    with pytest.warns(UserWarning, match="No valid pupil points found in camera image"):
        X, _ = e.get_pupil_in_camera_image(c)

    # Should return None for behind-camera scenario
    assert X is None, "Should return None when eye is behind camera"


def test_eye_rotated_90_degrees():
    """Test that method returns None when eye is rotated 90 degrees away."""
    e = Eye(fovea_displacement=False)
    c = Camera()

    # Position eye at [0, 500mm, 200mm]
    eye_pos = Position3D(x=0, y=500e-3, z=200e-3)
    e.position = eye_pos

    # Set up camera with proper orientation pointing at eye
    c.trans[0:3, 0:3] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    c.rest_trans = c.trans.copy()
    c.point_at(e.position)

    # Set eye rest orientation same as camera initially
    e.set_rest_orientation(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]))

    # Disable camera error for consistent results
    c.err = 0
    c.err_type = "uniform"

    # Eye looks 90 degrees away (perpendicular to camera)
    # If camera is at origin looking toward eye, make eye look to the side
    side_target = Position3D(x=eye_pos.x + 1, y=eye_pos.y, z=eye_pos.z)
    e.look_at(side_target)

    # Get pupil image - may return None or some points depending on geometry
    e.pupil.N = 20  # Set pupil resolution
    X, _ = e.get_pupil_in_camera_image(c)

    # In this case we just verify the method handles it gracefully
    # Could be None or a valid result depending on exact geometry
    if X is not None:
        assert isinstance(X, list)  # Should be List[Point2D]
        assert all(isinstance(p, Point2D) for p in X)
        assert all(np.isfinite(p.x) and np.isfinite(p.y) for p in X)
