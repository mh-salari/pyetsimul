"""Unit tests for Camera.unproject method."""

import numpy as np
from et_simul.core.camera import Camera


def test_single_point_unprojection():
    """Test basic unprojection of single 2D point with MATLAB reference values."""
    # Create standard camera
    c = Camera()

    # Single 2D point on image plane
    X = np.array([[100.0], [50.0]])  # 2x1 matrix
    d = 200.0  # Distance from camera

    pos = c.unproject(X, d)

    # MATLAB reference values
    expected_pos = np.array(
        [
            [6.9444444444444446],
            [3.4722222222222223],
            [-200.0000000000000000],
            [1.0000000000000000],
        ]
    )

    np.testing.assert_allclose(pos, expected_pos, rtol=1e-15, atol=1e-15)


def test_multiple_points_unprojection():
    """Test unprojection of multiple 2D points with MATLAB reference values."""
    c = Camera()

    # Multiple 2D points on image plane
    X = np.array(
        [
            [0.0, 100.0, -150.0, 200.0],  # x coordinates
            [0.0, 50.0, -75.0, 100.0],  # y coordinates (2x4 matrix)
        ]
    )
    d = 300.0  # Distance from camera

    pos = c.unproject(X, d)

    # MATLAB reference values
    expected_pos = np.array(
        [
            [
                0.0000000000000000,
                10.4166666666666679,
                -15.6250000000000000,
                20.8333333333333357,
            ],
            [
                0.0000000000000000,
                5.2083333333333339,
                -7.8125000000000000,
                10.4166666666666679,
            ],
            [
                -300.0000000000000000,
                -300.0000000000000000,
                -300.0000000000000000,
                -300.0000000000000000,
            ],
            [
                1.0000000000000000,
                1.0000000000000000,
                1.0000000000000000,
                1.0000000000000000,
            ],
        ]
    )

    np.testing.assert_allclose(pos, expected_pos, rtol=1e-15, atol=1e-15)


def test_image_center_unprojection():
    """Test unprojection of point at image center with MATLAB reference values."""
    c = Camera()

    # Point at image center
    X = np.array([[0.0], [0.0]])  # Center of image plane
    d = 150.0

    pos = c.unproject(X, d)

    # MATLAB reference values
    expected_pos = np.array(
        [
            [0.0000000000000000],
            [0.0000000000000000],
            [-150.0000000000000000],
            [1.0000000000000000],
        ]
    )

    np.testing.assert_allclose(pos, expected_pos, rtol=1e-15, atol=1e-15)


def test_different_focal_length():
    """Test unprojection with modified focal length and MATLAB reference values."""
    c = Camera()
    c.focal_length = 1000.0  # Modified focal length

    # 2D point
    X = np.array([[50.0], [-25.0]])
    d = 500.0

    pos = c.unproject(X, d)

    # MATLAB reference values
    expected_pos = np.array(
        [
            [25.0000000000000000],
            [-12.5000000000000000],
            [-500.0000000000000000],
            [1.0000000000000000],
        ]
    )

    np.testing.assert_allclose(pos, expected_pos, rtol=1e-15, atol=1e-15)


def test_large_coordinates():
    """Test unprojection with large coordinate values and MATLAB reference values."""
    c = Camera()

    # Large 2D coordinates
    X = np.array(
        [
            [500.0, -800.0],  # x coordinates
            [300.0, -600.0],  # y coordinates (2x2 matrix)
        ]
    )
    d = 1000.0  # Large distance

    pos = c.unproject(X, d)

    # MATLAB reference values
    expected_pos = np.array(
        [
            [173.6111111111111143, -277.7777777777777715],
            [104.1666666666666714, -208.3333333333333428],
            [-1000.0000000000000000, -1000.0000000000000000],
            [1.0000000000000000, 1.0000000000000000],
        ]
    )

    np.testing.assert_allclose(pos, expected_pos, rtol=1e-15, atol=1e-15)


def test_projection_unprojection_roundtrip():
    """Test that project-unproject roundtrip recovers original points."""
    # Create camera with no noise
    c = Camera()
    c.err = 0  # No noise for exact roundtrip

    # Start with 3D points
    original_3d = np.array(
        [
            [50.0, -30.0],  # x coordinates
            [25.0, -15.0],  # y coordinates
            [-400.0, -600.0],  # z coordinates (negative for in front of camera)
            [1.0, 1.0],  # homogeneous (4x2 matrix)
        ]
    )

    # Project to 2D
    projected_2d, distances, valid = c.project(original_3d)

    # Unproject back using original distances
    unprojected_3d = np.zeros((4, projected_2d.shape[1]))
    for i in range(projected_2d.shape[1]):
        unprojected_3d[:, i : i + 1] = c.unproject(projected_2d[:, i : i + 1], distances[i])

    # Check roundtrip accuracy
    diff_matrix = np.abs(original_3d - unprojected_3d)
    max_diff = np.max(diff_matrix)

    # Roundtrip should be very accurate
    assert max_diff < 1e-10


def test_output_properties():
    """Test that output has correct properties."""
    c = Camera()

    # Single point test
    X = np.array([[100.0], [50.0]])
    d = 200.0

    pos = c.unproject(X, d)

    # Check types and shapes
    assert isinstance(pos, np.ndarray)
    assert pos.shape == (4, 1)
    assert pos.dtype == np.float64

    # Test multiple points
    X_multi = np.array([[0.0, 100.0], [0.0, 50.0]])
    pos_multi = c.unproject(X_multi, d)

    assert pos_multi.shape == (4, 2)
    assert pos_multi.dtype == np.float64

    # Check homogeneous coordinate property (last row should be all 1s)
    np.testing.assert_allclose(pos_multi[3, :], 1.0, rtol=1e-15, atol=1e-15)
