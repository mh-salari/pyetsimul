"""Unit tests for reflect_ray_conic function."""

import numpy as np
from et_simul.optics.reflections import reflect_ray_sphere, reflect_ray_conic


def test_conic_equals_sphere_reflection():
    """Test conic reflection with k=0 gives sphere behavior."""
    # Off-axis ray for realistic reflection
    R0 = np.array([-1.0, 0.5, -2.0, 1.0])
    Rd = np.array([0.2, -0.1, 1.0, 0.0])
    Rd[:3] = Rd[:3] / np.linalg.norm(Rd[:3])
    radius = 1.0

    # For k=0, both should create equivalent spheres at the same location
    sphere_center = np.array([0.0, 0.0, 0.0, 1.0])  # Sphere at origin
    conic_center = np.array([0.0, 0.0, 0.0, 1.0])  # Conic also at origin

    # Sphere function with proper center
    sphere_U0, sphere_Ud = reflect_ray_sphere(R0, Rd, sphere_center, radius)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_U0, conic_Ud = reflect_ray_conic(R0, Rd, conic_center, r_param, k)

    # Should match the sphere centered at origin
    assert sphere_U0 is not None, "sphere_U0 should not be None"
    assert conic_U0 is not None, "conic_U0 should not be None"
    np.testing.assert_allclose(sphere_U0, conic_U0, rtol=1e-12, atol=1e-14)

    assert sphere_Ud is not None, "sphere_Ud should not be None"
    assert conic_Ud is not None, "conic_Ud should not be None"
    np.testing.assert_allclose(sphere_Ud, conic_Ud, rtol=1e-12, atol=1e-14)
