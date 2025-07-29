"""Unit tests for intersect_ray_sphere function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_sphere


def test_two_intersections():
    """Test ray intersecting sphere with two points and MATLAB reference values."""
    # Ray passes through sphere center
    R0 = np.array([0.0, 0.0, -2.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0])  # Ray direction
    S0 = np.array([0.0, 0.0, 0.0])  # Sphere center
    Sr = 1.0  # Sphere radius

    pos1, pos2 = intersect_ray_sphere(R0, Rd, S0, Sr)

    # MATLAB reference values
    expected_pos1 = np.array([0.0000000000000000, 0.0000000000000000, -1.0000000000000000])
    expected_pos2 = np.array([0.0000000000000000, 0.0000000000000000, 1.0000000000000000])

    assert pos1 is not None
    assert pos2 is not None
    np.testing.assert_allclose(pos1, expected_pos1, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(pos2, expected_pos2, rtol=1e-14, atol=1e-15)


def test_tangent_intersection():
    """Test ray tangent to sphere and MATLAB reference values."""
    # Ray just touches sphere surface
    R0 = np.array([0.0, 1.0, -2.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0])  # Ray direction
    S0 = np.array([0.0, 0.0, 0.0])  # Sphere center
    Sr = 1.0  # Sphere radius

    pos1, pos2 = intersect_ray_sphere(R0, Rd, S0, Sr)

    # MATLAB reference values (both identical for tangent)
    expected_pos = np.array([0.0000000000000000, 1.0000000000000000, 0.0000000000000000])

    assert pos1 is not None
    assert pos2 is not None
    np.testing.assert_allclose(pos1, expected_pos, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(pos2, expected_pos, rtol=1e-14, atol=1e-15)


def test_ray_missing_sphere():
    """Test ray that misses sphere - should return None."""
    # Ray misses sphere completely
    R0 = np.array([0.0, 2.0, -2.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0])  # Ray direction
    S0 = np.array([0.0, 0.0, 0.0])  # Sphere center
    Sr = 1.0  # Sphere radius

    pos1, pos2 = intersect_ray_sphere(R0, Rd, S0, Sr)

    # Should return None for both when ray misses sphere
    assert pos1 is None
    assert pos2 is None


def test_ray_inside_sphere():
    """Test ray starting inside sphere and MATLAB reference values."""
    # Ray origin at sphere center
    R0 = np.array([0.0, 0.0, 0.0])  # Ray origin (at center)
    Rd = np.array([1.0, 0.0, 0.0])  # Ray direction
    S0 = np.array([0.0, 0.0, 0.0])  # Sphere center
    Sr = 2.0  # Sphere radius

    pos1, pos2 = intersect_ray_sphere(R0, Rd, S0, Sr)

    # MATLAB reference values
    expected_pos1 = np.array([-2.0000000000000000, 0.0000000000000000, 0.0000000000000000])
    expected_pos2 = np.array([2.0000000000000000, 0.0000000000000000, 0.0000000000000000])

    assert pos1 is not None
    assert pos2 is not None
    np.testing.assert_allclose(pos1, expected_pos1, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(pos2, expected_pos2, rtol=1e-14, atol=1e-15)


def test_non_unit_direction():
    """Test with non-unit ray direction and MATLAB reference values."""
    # Ray direction with length 2.0 (should be normalized)
    R0 = np.array([0.0, 0.0, -3.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 2.0])  # Ray direction (length 2.0)
    S0 = np.array([0.0, 0.0, 0.0])  # Sphere center
    Sr = 1.0  # Sphere radius

    pos1, pos2 = intersect_ray_sphere(R0, Rd, S0, Sr)

    # MATLAB reference values (same as unit direction case)
    expected_pos1 = np.array([0.0000000000000000, 0.0000000000000000, -1.0000000000000000])
    expected_pos2 = np.array([0.0000000000000000, 0.0000000000000000, 1.0000000000000000])

    assert pos1 is not None
    assert pos2 is not None
    np.testing.assert_allclose(pos1, expected_pos1, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(pos2, expected_pos2, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    R0 = np.array([0.0, 0.0, -2.0])
    Rd = np.array([0.0, 0.0, 1.0])
    S0 = np.array([0.0, 0.0, 0.0])
    Sr = 1.0

    pos1, pos2 = intersect_ray_sphere(R0, Rd, S0, Sr)
    assert pos1 is not None, "pos1 should not be None for these inputs"
    assert pos2 is not None, "pos2 should not be None for these inputs"

    # Check types and shapes
    assert isinstance(pos1, np.ndarray)
    assert isinstance(pos2, np.ndarray)
    assert pos1.dtype == np.float64
    assert pos2.dtype == np.float64
    assert pos1.shape == (3,)
    assert pos2.shape == (3,)

    # Both points should be on sphere surface
    dist1 = np.linalg.norm(pos1 - S0)
    dist2 = np.linalg.norm(pos2 - S0)
    assert np.isclose(dist1, Sr, rtol=1e-12)
    assert np.isclose(dist2, Sr, rtol=1e-12)

    # pos1 should be closer to ray origin than pos2
    dist_to_origin1 = np.linalg.norm(pos1 - R0)
    dist_to_origin2 = np.linalg.norm(pos2 - R0)
    assert dist_to_origin1 <= dist_to_origin2
