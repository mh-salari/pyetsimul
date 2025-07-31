"""Unit tests for intersect_ray_circle function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_circle


def test_normal_intersection():
    """Test ray intersecting circle with MATLAB reference values."""
    # Ray hitting circle from below (returns closest intersection)
    R0 = np.array([0.0, -2.0])  # Ray origin
    Rd = np.array([0.0, 1.0])  # Ray direction
    C0 = np.array([0.0, 0.0])  # Circle center
    Cr = 1.0  # Circle radius

    pos = intersect_ray_circle(R0, Rd, C0, Cr)

    # MATLAB reference values
    expected_pos = np.array([0.0, -1.0])

    assert pos is not None
    np.testing.assert_allclose(pos, expected_pos, rtol=1e-14, atol=1e-15)


def test_tangent_intersection():
    """Test ray tangent to circle with MATLAB reference values."""
    # Ray just touching circle surface
    R0 = np.array([1.0, -2.0])  # Ray origin
    Rd = np.array([0.0, 1.0])  # Ray direction
    C0 = np.array([0.0, 0.0])  # Circle center
    Cr = 1.0  # Circle radius

    pos = intersect_ray_circle(R0, Rd, C0, Cr)

    # MATLAB reference values
    expected_pos = np.array([1.0, 0.0])

    assert pos is not None
    np.testing.assert_allclose(pos, expected_pos, rtol=1e-14, atol=1e-15)


def test_ray_missing_circle():
    """Test ray that misses circle - should return None."""
    # Ray misses circle completely
    R0 = np.array([2.0, -2.0])  # Ray origin
    Rd = np.array([0.0, 1.0])  # Ray direction
    C0 = np.array([0.0, 0.0])  # Circle center
    Cr = 1.0  # Circle radius

    pos = intersect_ray_circle(R0, Rd, C0, Cr)

    # Should return None when ray misses circle
    assert pos is None


def test_ray_inside_circle():
    """Test ray starting inside circle with MATLAB reference values."""
    # Ray origin at circle center
    R0 = np.array([0.0, 0.0])  # Ray origin (at center)
    Rd = np.array([1.0, 0.0])  # Ray direction
    C0 = np.array([0.0, 0.0])  # Circle center
    Cr = 2.0  # Circle radius

    pos = intersect_ray_circle(R0, Rd, C0, Cr)

    # MATLAB reference values (returns the intersection behind the ray origin)
    expected_pos = np.array([-2.0, 0.0])

    assert pos is not None
    np.testing.assert_allclose(pos, expected_pos, rtol=1e-14, atol=1e-15)


def test_non_unit_direction():
    """Test with non-unit ray direction and MATLAB reference values."""
    # Ray direction with length 2.0 (should be normalized)
    R0 = np.array([0.0, -3.0])  # Ray origin
    Rd = np.array([0.0, 2.0])  # Ray direction (length 2.0)
    C0 = np.array([0.0, 0.0])  # Circle center
    Cr = 1.0  # Circle radius

    pos = intersect_ray_circle(R0, Rd, C0, Cr)

    # MATLAB reference values (same as unit direction case)
    expected_pos = np.array([0.0, -1.0])

    assert pos is not None
    np.testing.assert_allclose(pos, expected_pos, rtol=1e-14, atol=1e-15)


def test_diagonal_intersection():
    """Test diagonal ray intersection with MATLAB reference values."""
    # Ray moving diagonally
    R0 = np.array([-2.0, -2.0])  # Ray origin
    Rd = np.array([1.0, 1.0])  # Ray direction (diagonal)
    C0 = np.array([0.0, 0.0])  # Circle center
    Cr = 1.0  # Circle radius

    pos = intersect_ray_circle(R0, Rd, C0, Cr)

    # MATLAB reference values
    expected_pos = np.array([-0.7071067811865472, -0.7071067811865472])

    assert pos is not None
    np.testing.assert_allclose(pos, expected_pos, rtol=1e-14, atol=1e-15)

    # Verify distance from center matches MATLAB precision
    dist_from_center = np.linalg.norm(pos - C0)
    expected_dist = 0.9999999999999996  # MATLAB's exact value
    assert abs(dist_from_center - expected_dist) < 1e-15


def test_output_properties():
    """Test that output has correct properties."""
    R0 = np.array([0.0, -2.0])
    Rd = np.array([0.0, 1.0])
    C0 = np.array([0.0, 0.0])
    Cr = 1.0

    pos = intersect_ray_circle(R0, Rd, C0, Cr)
    assert pos is not None, "pos should not be None for these inputs"

    # Check types and shapes
    assert isinstance(pos, np.ndarray)
    assert pos.dtype == np.float64
    assert pos.shape == (2,)

    # Point should be on circle surface
    dist_to_center = np.linalg.norm(pos - C0)
    assert np.isclose(dist_to_center, Cr, rtol=1e-12)

    # Point should be on ray: pos = R0 + t*Rd for some t
    if not np.allclose(Rd, 0):
        non_zero_idx = np.nonzero(Rd)[0][0]
        t = (pos[non_zero_idx] - R0[non_zero_idx]) / Rd[non_zero_idx]
        pos_on_ray = R0 + t * Rd
        np.testing.assert_allclose(pos, pos_on_ray, rtol=1e-12)
