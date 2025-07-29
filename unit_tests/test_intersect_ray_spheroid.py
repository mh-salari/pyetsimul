"""Unit tests for intersect_ray_spheroid function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_sphere, intersect_ray_spheroid


def test_spheroid_equals_sphere_on_axis():
    """Test spheroid with equal axes gives same result as sphere function."""
    # On-axis ray
    R0 = np.array([0.0, 0.0, -2.0])
    Rd = np.array([0.0, 0.0, 1.0])
    S0 = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    # Sphere function
    sphere_pos1, sphere_pos2 = intersect_ray_sphere(R0, Rd, S0, radius)

    # Spheroid function with equal axes (should be identical to sphere)
    a = b = c = radius
    spheroid_pos1, spheroid_pos2 = intersect_ray_spheroid(R0, Rd, S0, a, b, c)

    # Should match exactly
    assert sphere_pos1 is not None and spheroid_pos1 is not None
    assert sphere_pos2 is not None and spheroid_pos2 is not None
    np.testing.assert_allclose(sphere_pos1, spheroid_pos1, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(sphere_pos2, spheroid_pos2, rtol=1e-12, atol=1e-14)


def test_spheroid_equals_sphere_off_axis():
    """Test spheroid with equal axes gives same result as sphere function for off-axis ray."""
    # Off-axis ray
    R0 = np.array([-1.5, 0.5, -2.0])
    Rd = np.array([0.3, -0.1, 1.0])
    Rd = Rd / np.linalg.norm(Rd)
    S0 = np.array([0.0, 0.0, 0.0])
    radius = 1.0

    # Sphere function
    sphere_pos1, sphere_pos2 = intersect_ray_sphere(R0, Rd, S0, radius)

    # Spheroid function with equal axes (should be identical to sphere)
    a = b = c = radius
    spheroid_pos1, spheroid_pos2 = intersect_ray_spheroid(R0, Rd, S0, a, b, c)

    # Should match exactly
    if sphere_pos1 is not None and spheroid_pos1 is not None:
        np.testing.assert_allclose(sphere_pos1, spheroid_pos1, rtol=1e-12, atol=1e-14)
    else:
        assert sphere_pos1 is None and spheroid_pos1 is None

    if sphere_pos2 is not None and spheroid_pos2 is not None:
        np.testing.assert_allclose(sphere_pos2, spheroid_pos2, rtol=1e-12, atol=1e-14)
    else:
        assert sphere_pos2 is None and spheroid_pos2 is None
