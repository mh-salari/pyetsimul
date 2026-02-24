"""Unit tests for Camera.project method."""

import numpy as np

from pyetsimul.core.camera import Camera
from pyetsimul.types import Point2D, Position3D


def test_mixed_points_in_out_bounds() -> None:
    """Test mixed points with some in bounds and some out of bounds."""
    # Create test camera matching parameters
    c = Camera(err_type="uniform", err=0)
    c.camera_matrix.focal_length = 500
    c.camera_matrix.resolution = Point2D(640, 480)

    # Define test points using Position3D
    pos = [Position3D(0, 100000, -200000), Position3D(0, -100000, -200000), Position3D(50000, 50000, -300000)]

    result = c.project(pos)
    x = result.image_points
    dist = result.distances
    valid_condition = result.valid_mask

    # Expected distances
    expected_dist = np.array([200000.0, 200000.0, 300000.0])

    # Check that invalid points are set to NaN (points 1 and 2 are out of bounds)
    assert np.isnan(x[0, 0])
    assert np.isnan(x[1, 0])
    assert np.isnan(x[0, 1])
    assert np.isnan(x[1, 1])

    # Check valid point projection (point 3)
    np.testing.assert_allclose(x[0, 2], 83.333333, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(x[1, 2], 83.333333, rtol=1e-6, atol=1e-6)

    # Check distances
    np.testing.assert_allclose(dist, expected_dist, rtol=1e-6, atol=1e-6)

    # Check valid condition (only point 3 should be valid)
    valid_indices = np.where(valid_condition)[0]
    expected_valid_indices = [2]
    np.testing.assert_array_equal(valid_indices, expected_valid_indices)


def test_all_points_outside_bounds() -> None:
    """Test points that are all outside image bounds."""
    c = Camera(err_type="uniform", err=0)
    c.camera_matrix.focal_length = 500
    c.camera_matrix.resolution = Point2D(640, 480)

    # Define test points that are clearly out of bounds
    pos = [Position3D(1000000, 0, -200000), Position3D(0, 1000000, -200000), Position3D(-1000000, -1000000, -200000)]

    result = c.project(pos)
    x = result.image_points
    dist = result.distances
    valid_condition = result.valid_mask

    expected_dist = np.array([200000.0, 200000.0, 200000.0])

    # Check that all points are set to NaN
    assert np.isnan(x).all()

    # Check distances
    np.testing.assert_allclose(dist, expected_dist, rtol=1e-6, atol=1e-6)

    # Check valid condition (no points should be valid)
    valid_indices = np.where(valid_condition)[0]
    assert len(valid_indices) == 0


def test_boundary_points() -> None:
    """Test points exactly on image boundary."""
    c = Camera(err_type="uniform", err=0)
    c.camera_matrix.focal_length = 500
    c.camera_matrix.resolution = Point2D(640, 480)

    # Calculate boundary points
    boundary_x = c.camera_matrix.resolution.x / 2  # 320
    boundary_y = c.camera_matrix.resolution.y / 2  # 240
    z_depth = -200000

    x_right = boundary_x * z_depth / c.camera_matrix.focal_length
    y_top = boundary_y * z_depth / c.camera_matrix.focal_length
    x_left = -boundary_x * z_depth / c.camera_matrix.focal_length
    y_bottom = -boundary_y * z_depth / c.camera_matrix.focal_length

    pos = [
        Position3D(x_right, 0, z_depth),
        Position3D(0, y_top, z_depth),
        Position3D(x_left, 0, z_depth),
        Position3D(0, y_bottom, z_depth),
    ]

    result = c.project(pos)
    x = result.image_points
    # dist = result.distances
    valid_condition = result.valid_mask

    # Expected projection values
    expected_x = np.array([[-320.0, 0.0, 320.0, 0.0], [0.0, -240.0, 0.0, 240.0]])

    # Check projected points
    np.testing.assert_allclose(x, expected_x, rtol=1e-6, atol=1e-6)

    # Check that all boundary points are valid
    valid_indices = np.where(valid_condition)[0]
    expected_valid_indices = [0, 1, 2, 3]
    np.testing.assert_array_equal(valid_indices, expected_valid_indices)


def test_center_point() -> None:
    """Test point at image center."""
    c = Camera(err_type="uniform", err=0)
    c.camera_matrix.focal_length = 500
    c.camera_matrix.resolution = Point2D(640, 480)

    # Point at center
    pos = Position3D(0, 0, -200000)

    result = c.project(pos)
    x = result.image_points
    dist = result.distances
    valid_condition = result.valid_mask

    # Should project to (0, 0)
    np.testing.assert_allclose(x[0, 0], 0.0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(x[1, 0], 0.0, rtol=1e-12, atol=1e-15)

    # Distance should be 200000
    np.testing.assert_allclose(dist[0], 200000.0, rtol=1e-12, atol=1e-15)

    # Should be valid
    assert valid_condition[0]


def test_output_properties() -> None:
    """Test that output has correct properties."""
    c = Camera(err_type="uniform", err=0)
    c.camera_matrix.focal_length = 500
    c.camera_matrix.resolution = Point2D(640, 480)

    # Simple test points
    pos = [Position3D(0, 0, -200000), Position3D(100000, 0, -300000)]

    result = c.project(pos)
    x = result.image_points
    dist = result.distances
    valid_condition = result.valid_mask

    # Check types and shapes
    assert isinstance(x, np.ndarray)
    assert isinstance(dist, np.ndarray)
    assert isinstance(valid_condition, np.ndarray)
    assert x.shape == (2, 2)
    assert dist.shape == (2,)
    assert valid_condition.shape == (2,)
    assert x.dtype == np.float64
    assert dist.dtype == np.float64
    assert valid_condition.dtype == bool
