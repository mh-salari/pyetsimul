"""Unit tests for Camera.unproject method."""

import numpy as np

from pyetsimul.core.camera import Camera
from pyetsimul.types import Point2D, Position3D


def test_single_point_unprojection() -> None:
    """Test basic unprojection of single 2D point with MATLAB reference values."""
    # Create standard camera
    c = Camera()

    # Single 2D point on image plane
    x = Point2D(100.0, 50.0)
    d = 200.0  # Distance from camera

    pos = c.unproject(x, d)

    # MATLAB reference values
    expected_pos = np.array([
        [6.9444444444444446],
        [3.4722222222222223],
        [-200.0],
        [1.0],
    ])

    # Convert Position3D result to array for comparison
    pos_array = np.array(pos).reshape(-1, 1)
    np.testing.assert_allclose(pos_array, expected_pos, rtol=1e-15, atol=1e-15)


def test_multiple_points_unprojection() -> None:
    """Test unprojection of multiple 2D points with MATLAB reference values."""
    c = Camera()

    # Multiple 2D points on image plane
    x = [Point2D(0.0, 0.0), Point2D(100.0, 50.0), Point2D(-150.0, -75.0), Point2D(200.0, 100.0)]
    d = 300.0  # Distance from camera

    pos = c.unproject(x, d)

    # MATLAB reference values
    expected_pos = np.array([
        [
            0.0,
            10.4166666666666679,
            -15.6250,
            20.8333333333333357,
        ],
        [
            0.0,
            5.2083333333333339,
            -7.8125,
            10.4166666666666679,
        ],
        [
            -300.0,
            -300.0,
            -300.0,
            -300.0,
        ],
        [
            1.0,
            1.0,
            1.0,
            1.0,
        ],
    ])

    # Convert List[Position3D] result to array for comparison
    pos_array = np.column_stack([np.array(p) for p in pos])
    np.testing.assert_allclose(pos_array, expected_pos, rtol=1e-15, atol=1e-15)


def test_image_center_unprojection() -> None:
    """Test unprojection of point at image center with MATLAB reference values."""
    c = Camera()

    # Point at image center
    x = Point2D(0.0, 0.0)  # Center of image plane
    d = 150.0

    pos = c.unproject(x, d)

    # MATLAB reference values
    expected_pos = np.array([
        [0.0],
        [0.0],
        [-150.0],
        [1.0],
    ])

    # Convert Position3D result to array for comparison (single point)
    pos_array = np.array(pos).reshape(-1, 1)
    np.testing.assert_allclose(pos_array, expected_pos, rtol=1e-15, atol=1e-15)


def test_different_focal_length() -> None:
    """Test unprojection with modified focal length and MATLAB reference values."""
    c = Camera()
    c.camera_matrix.focal_length = 1000.0  # Modified focal length

    # 2D point
    x = Point2D(50.0, -25.0)
    d = 500.0

    pos = c.unproject(x, d)

    # MATLAB reference values
    expected_pos = np.array([
        [25.0],
        [-12.5000],
        [-500.0],
        [1.0],
    ])

    # Convert Position3D result to array for comparison (single point)
    pos_array = np.array(pos).reshape(-1, 1)
    np.testing.assert_allclose(pos_array, expected_pos, rtol=1e-15, atol=1e-15)


def test_large_coordinates() -> None:
    """Test unprojection with large coordinate values and MATLAB reference values."""
    c = Camera()

    # Large 2D coordinates
    x = [Point2D(500.0, 300.0), Point2D(-800.0, -600.0)]
    d = 1000.0  # Large distance

    pos = c.unproject(x, d)

    # MATLAB reference values
    expected_pos = np.array([
        [173.6111111111111143, -277.7777777777777715],
        [104.1666666666666714, -208.3333333333333428],
        [-1000.0, -1000.0],
        [1.0, 1.0],
    ])

    # Convert List[Position3D] result to array for comparison
    pos_array = np.column_stack([np.array(p) for p in pos])
    np.testing.assert_allclose(pos_array, expected_pos, rtol=1e-15, atol=1e-15)


def test_projection_unprojection_roundtrip() -> None:
    """Test that project-unproject roundtrip recovers original points."""
    # Create camera with no noise
    c = Camera()
    c.err = 0  # No noise for exact roundtrip

    # Start with 3D points
    original_3d = [Position3D(50.0, 25.0, -400.0), Position3D(-30.0, -15.0, -600.0)]

    # Project to 2D
    result = c.project(original_3d)
    projected_2d = result.image_points
    distances = result.distances
    # valid = result.valid_mask

    # Unproject back using original distances
    unprojected_3d = []
    for i in range(projected_2d.shape[1]):
        point_2d = Point2D(projected_2d[0, i], projected_2d[1, i])
        result_pos = c.unproject(point_2d, distances[i])
        unprojected_3d.append(result_pos)

    # Check roundtrip accuracy
    original_array = np.column_stack([np.array(p) for p in original_3d])
    unprojected_array = np.column_stack([np.array(p) for p in unprojected_3d])
    diff_matrix = np.abs(original_array - unprojected_array)
    max_diff = np.max(diff_matrix)

    # Roundtrip should be very accurate
    assert max_diff < 1e-10


def test_output_properties() -> None:
    """Test that output has correct properties."""
    c = Camera()

    # Single point test
    x = Point2D(100.0, 50.0)
    d = 200.0

    pos = c.unproject(x, d)

    # Check types and shapes
    assert isinstance(pos, Position3D)
    pos_array = np.array(pos)
    assert pos_array.shape == (4,)
    assert pos_array.dtype == np.float64

    # Test multiple points
    x_multi = [Point2D(0.0, 0.0), Point2D(100.0, 50.0)]
    pos_multi = c.unproject(x_multi, d)

    # Check types and shapes for multiple points
    assert isinstance(pos_multi, list)
    assert len(pos_multi) == 2
    assert all(isinstance(p, Position3D) for p in pos_multi)
    pos_multi_array = np.column_stack([np.array(p) for p in pos_multi])
    assert pos_multi_array.shape == (4, 2)
    assert pos_multi_array.dtype == np.float64

    # Check homogeneous coordinate property (last row should be all 1s)
    np.testing.assert_allclose(pos_multi_array[3, :], 1.0, rtol=1e-15, atol=1e-15)
