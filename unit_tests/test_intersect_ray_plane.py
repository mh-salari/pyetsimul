"""Unit tests for intersect_ray_plane function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_plane


def test_normal_intersection():
    """Test normal ray-plane intersection with MATLAB reference values."""
    # Ray perpendicular to plane
    R0 = np.array([0.0, 0.0, -2.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0])  # Ray direction
    P0 = np.array([0.0, 0.0, 0.0])  # Point on plane
    Pn = np.array([0.0, 0.0, 1.0])  # Plane normal

    x = intersect_ray_plane(R0, Rd, P0, Pn)

    # MATLAB reference values
    expected_x = np.array([0.0000000000000000, 0.0000000000000000, 0.0000000000000000])

    assert x is not None
    np.testing.assert_allclose(x, expected_x, rtol=1e-14, atol=1e-15)


def test_ray_parallel_to_plane():
    """Test ray parallel to plane - should return None."""
    # Ray parallel to plane (no intersection)
    R0 = np.array([0.0, 0.0, 1.0])  # Ray origin
    Rd = np.array([1.0, 0.0, 0.0])  # Ray direction (parallel to plane)
    P0 = np.array([0.0, 0.0, 0.0])  # Point on plane
    Pn = np.array([0.0, 0.0, 1.0])  # Plane normal

    x = intersect_ray_plane(R0, Rd, P0, Pn)

    # Should return None for parallel ray
    assert x is None


def test_ray_starting_on_plane():
    """Test ray starting on plane with MATLAB reference values."""
    # Ray origin on plane surface
    R0 = np.array([0.0, 0.0, 0.0])  # Ray origin (on plane)
    Rd = np.array([0.0, 0.0, 1.0])  # Ray direction
    P0 = np.array([0.0, 0.0, 0.0])  # Point on plane
    Pn = np.array([0.0, 0.0, 1.0])  # Plane normal

    x = intersect_ray_plane(R0, Rd, P0, Pn)

    # MATLAB reference values
    expected_x = np.array([0.0000000000000000, 0.0000000000000000, 0.0000000000000000])

    assert x is not None
    np.testing.assert_allclose(x, expected_x, rtol=1e-14, atol=1e-15)


def test_oblique_intersection():
    """Test oblique intersection with tilted plane and MATLAB reference values."""
    # Angled ray with tilted plane
    R0 = np.array([0.0, 0.0, -2.0])  # Ray origin
    Rd = np.array([1.0, 1.0, 1.0])  # Ray direction (diagonal)
    P0 = np.array([1.0, 1.0, 0.0])  # Point on plane
    Pn = np.array([1.0, 1.0, 1.0])  # Plane normal (diagonal)

    x = intersect_ray_plane(R0, Rd, P0, Pn)

    # MATLAB reference values
    expected_x = np.array([1.3333333333333333, 1.3333333333333333, -0.6666666666666667])

    assert x is not None
    np.testing.assert_allclose(x, expected_x, rtol=1e-14, atol=1e-15)

    # Verify point is on plane: (x-P0)·Pn = 0
    plane_check = np.dot(x - P0, Pn)
    assert abs(plane_check) < 1e-14


def test_backward_intersection():
    """Test ray pointing away from plane (backward intersection) with MATLAB reference values."""
    # Ray pointing away from plane
    R0 = np.array([0.0, 0.0, 1.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0])  # Ray direction (away from plane)
    P0 = np.array([0.0, 0.0, 0.0])  # Point on plane
    Pn = np.array([0.0, 0.0, 1.0])  # Plane normal

    x = intersect_ray_plane(R0, Rd, P0, Pn)

    # MATLAB reference values (backward intersection)
    expected_x = np.array([0.0000000000000000, 0.0000000000000000, 0.0000000000000000])

    assert x is not None
    np.testing.assert_allclose(x, expected_x, rtol=1e-14, atol=1e-15)


def test_xy_plane_intersection():
    """Test intersection with XY plane and MATLAB reference values."""
    # Ray hitting XY plane from above
    R0 = np.array([1.0, 2.0, 3.0])  # Ray origin
    Rd = np.array([0.0, 0.0, -1.0])  # Ray direction (downward)
    P0 = np.array([0.0, 0.0, 0.0])  # Point on plane
    Pn = np.array([0.0, 0.0, 1.0])  # Plane normal

    x = intersect_ray_plane(R0, Rd, P0, Pn)

    # MATLAB reference values
    expected_x = np.array([1.0000000000000000, 2.0000000000000000, 0.0000000000000000])

    assert x is not None
    np.testing.assert_allclose(x, expected_x, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    R0 = np.array([0.0, 0.0, -2.0])
    Rd = np.array([0.0, 0.0, 1.0])
    P0 = np.array([0.0, 0.0, 0.0])
    Pn = np.array([0.0, 0.0, 1.0])

    x = intersect_ray_plane(R0, Rd, P0, Pn)
    assert x is not None, "x should not be None for these inputs"

    # Check types and shapes
    assert isinstance(x, np.ndarray)
    assert x.dtype == np.float64
    assert x.shape == (3,)

    # Point should be on plane: (x-P0)·Pn = 0
    plane_equation = np.dot(x - P0, Pn)
    assert abs(plane_equation) < 1e-14

    # Point should be on ray: x = R0 + t*Rd for some t
    if not np.allclose(Rd, 0):
        non_zero_idx = np.nonzero(Rd)[0][0]
        t = (x[non_zero_idx] - R0[non_zero_idx]) / Rd[non_zero_idx]
        x_on_ray = R0 + t * Rd
        np.testing.assert_allclose(x, x_on_ray, rtol=1e-12)
