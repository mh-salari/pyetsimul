"""Unit tests for intersect_ray_conic function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_sphere, intersect_ray_conic


def test_conic_equals_sphere_on_axis():
    """Test conic with k=0 gives sphere behavior."""
    # On-axis ray
    R0 = np.array([0.0, 0.0, -2.0, 1.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0, 0.0])  # Ray direction
    radius = 1.0

    # For k=0, both should create equivalent spheres at the same location
    sphere_center = np.array([0.0, 0.0, 0.0, 1.0])  # Sphere at origin
    conic_center = np.array([0.0, 0.0, 0.0, 1.0])  # Conic also at origin

    # Sphere function with proper center
    sphere_pos1, sphere_pos2 = intersect_ray_sphere(R0, Rd, sphere_center, radius)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_pos1, conic_pos2 = intersect_ray_conic(R0, Rd, conic_center, r_param, k)

    # Should match the sphere centered at origin
    assert sphere_pos1 is not None and conic_pos1 is not None
    assert sphere_pos2 is not None and conic_pos2 is not None
    np.testing.assert_allclose(sphere_pos1, conic_pos1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(sphere_pos2, conic_pos2, rtol=1e-12, atol=1e-14)


def test_conic_equals_sphere_off_axis():
    """Test conic with k=0 gives sphere behavior for off-axis ray."""
    # Off-axis ray
    R0 = np.array([-1.5, 0.5, -2.0, 1.0])
    Rd = np.array([0.3, -0.1, 1.0, 0.0])
    Rd[:3] = Rd[:3] / np.linalg.norm(Rd[:3])
    radius = 1.0

    # For k=0, both should create equivalent spheres at the same location
    sphere_center = np.array([0.0, 0.0, 0.0, 1.0])  # Sphere at origin
    conic_center = np.array([0.0, 0.0, 0.0, 1.0])  # Conic also at origin

    # Sphere function with proper center
    sphere_pos1, sphere_pos2 = intersect_ray_sphere(R0, Rd, sphere_center, radius)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_pos1, conic_pos2 = intersect_ray_conic(R0, Rd, conic_center, r_param, k)

    # Should match the sphere centered at origin
    assert sphere_pos1 is not None, "sphere_pos1 should not be None"
    assert conic_pos1 is not None, "conic_pos1 should not be None"
    np.testing.assert_allclose(sphere_pos1, conic_pos1, rtol=1e-12, atol=1e-14)

    assert sphere_pos2 is not None, "sphere_pos2 should not be None"
    assert conic_pos2 is not None, "conic_pos2 should not be None"
    np.testing.assert_allclose(sphere_pos2, conic_pos2, rtol=1e-12, atol=1e-14)
