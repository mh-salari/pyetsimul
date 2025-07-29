"""Unit tests for refract_ray_spheroid function."""

import numpy as np
from et_simul.optics.refractions import refract_ray_sphere, refract_ray_spheroid


def test_spheroid_equals_sphere_refraction():
    """Test spheroid refraction with equal axes gives same result as sphere function."""
    # Off-axis ray for realistic refraction
    R0 = np.array([-1.0, 0.5, -2.0])
    Rd = np.array([0.2, -0.1, 1.0])
    Rd = Rd / np.linalg.norm(Rd)
    S0 = np.array([0.0, 0.0, 0.0])
    radius = 1.0
    n_outside = 1.0
    n_inside = 1.5

    # Sphere function
    sphere_U0, sphere_Ud = refract_ray_sphere(R0, Rd, S0, radius, n_outside, n_inside)

    # Spheroid function with equal axes (should be identical to sphere)
    a = b = c = radius
    spheroid_U0, spheroid_Ud = refract_ray_spheroid(R0, Rd, S0, a, b, c, n_outside, n_inside)

    # Should match exactly
    if sphere_U0 is not None and spheroid_U0 is not None:
        np.testing.assert_allclose(sphere_U0, spheroid_U0, rtol=1e-12, atol=1e-14)
    else:
        assert sphere_U0 is None and spheroid_U0 is None

    if sphere_Ud is not None and spheroid_Ud is not None:
        np.testing.assert_allclose(sphere_Ud, spheroid_Ud, rtol=1e-12, atol=1e-14)
    else:
        assert sphere_Ud is None and spheroid_Ud is None
