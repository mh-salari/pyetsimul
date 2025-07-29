"""Unit tests for intersect_ray_conic function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_sphere, intersect_ray_conic


def test_conic_equals_sphere_on_axis():
    """Test conic with Q=0 gives sphere behavior (accounting for center shift)."""
    # On-axis ray
    R0 = np.array([0.0, 0.0, -2.0])
    Rd = np.array([0.0, 0.0, 1.0])
    radius = 1.0

    # For Q=0, we want both to create equivalent spheres at the same location
    sphere_center = np.array([0.0, 0.0, 0.0])  # Sphere at origin
    conic_center = np.array([0.0, 0.0, radius])  # Conic offset by +r to create sphere at origin

    # Sphere function with proper center
    sphere_pos1, sphere_pos2 = intersect_ray_sphere(R0, Rd, sphere_center, radius)

    # Conic function with Q=0
    r_apical = radius
    Q = 0.0  # Q=0 represents a perfect sphere
    conic_pos1, conic_pos2 = intersect_ray_conic(R0, Rd, conic_center, r_apical, Q)

    # Should match the sphere centered at origin
    assert sphere_pos1 is not None and conic_pos1 is not None
    assert sphere_pos2 is not None and conic_pos2 is not None
    np.testing.assert_allclose(sphere_pos1, conic_pos1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(sphere_pos2, conic_pos2, rtol=1e-12, atol=1e-14)


def test_conic_equals_sphere_off_axis():
    """Test conic with Q=0 gives sphere behavior for off-axis ray (accounting for center shift)."""
    # Off-axis ray
    R0 = np.array([-1.5, 0.5, -2.0])
    Rd = np.array([0.3, -0.1, 1.0])
    Rd = Rd / np.linalg.norm(Rd)
    radius = 1.0

    # For Q=0, we want both to create equivalent spheres at the same location
    sphere_center = np.array([0.0, 0.0, 0.0])  # Sphere at origin
    conic_center = np.array([0.0, 0.0, radius])  # Conic offset by +r to create sphere at origin

    # Sphere function with proper center
    sphere_pos1, sphere_pos2 = intersect_ray_sphere(R0, Rd, sphere_center, radius)

    # Conic function with Q=0
    r_apical = radius
    Q = 0.0  # Q=0 represents a perfect sphere
    conic_pos1, conic_pos2 = intersect_ray_conic(R0, Rd, conic_center, r_apical, Q)

    # Should match the sphere centered at origin
    assert sphere_pos1 is not None, "sphere_pos1 should not be None"
    assert conic_pos1 is not None, "conic_pos1 should not be None"
    np.testing.assert_allclose(sphere_pos1, conic_pos1, rtol=1e-12, atol=1e-14)

    assert sphere_pos2 is not None, "sphere_pos2 should not be None"
    assert conic_pos2 is not None, "conic_pos2 should not be None"
    np.testing.assert_allclose(sphere_pos2, conic_pos2, rtol=1e-12, atol=1e-14)
