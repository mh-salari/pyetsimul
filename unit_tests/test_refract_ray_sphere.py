"""Unit tests for refract_ray_sphere function."""

import numpy as np
from et_simul.optics.refractions import refract_ray_sphere


def test_basic_refraction():
    """Test basic refraction scenario with MATLAB reference values."""
    # Ray entering sphere (relevant to corneal refraction)
    R0 = np.array([0.0, 0.0, 0.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0])  # Ray direction
    S0 = np.array([0.0, 0.0, 5.0])  # Sphere center
    Sr = 2.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass/cornea

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)

    # MATLAB reference values
    expected_U0 = np.array([0.0000000000000000, 0.0000000000000000, 3.0000000000000000])
    expected_Ud = np.array([0.0000000000000000, 0.0000000000000000, 1.0000000000000000])

    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-12, atol=1e-14)


def test_oblique_incidence():
    """Test oblique incidence refraction with MATLAB reference values."""
    # Diagonal ray - relevant to off-axis eye tracking scenarios
    R0 = np.array([-3.0, 0.0, 0.0])  # Ray origin
    Rd = np.array([1.0, 0.0, 1.0])  # Ray direction (diagonal)
    S0 = np.array([0.0, 0.0, 5.0])  # Sphere center
    Sr = 2.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)

    # MATLAB reference values with many decimals
    expected_U0 = np.array(
        [-0.0000000000000013, 0.0000000000000000, 2.9999999999999987]
    )
    expected_Ud = np.array([0.4714045207910319, 0.0000000000000000, 0.8819171036881968])

    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-12, atol=1e-14)


def test_ray_missing_sphere():
    """Test case where ray misses sphere - should return None."""
    R0 = np.array([0.0, 0.0, 0.0])  # Ray origin
    Rd = np.array([1.0, 0.0, 0.0])  # Ray direction (misses sphere)
    S0 = np.array([0.0, 0.0, 5.0])  # Sphere center
    Sr = 1.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)

    # Should return None for both when ray misses sphere
    assert U0 is None
    assert Ud is None


def test_output_properties():
    """Test that output has correct properties."""
    R0 = np.array([0.0, 0.0, 0.0])
    Rd = np.array([0.0, 0.0, 1.0])
    S0 = np.array([0.0, 0.0, 5.0])
    Sr = 2.0
    n_outside = 1.0
    n_sphere = 1.5

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)

    if U0 is not None and Ud is not None:
        # Check types and shapes
        assert isinstance(U0, np.ndarray)
        assert isinstance(Ud, np.ndarray)
        assert U0.shape == (3,)
        assert Ud.shape == (3,)
        assert U0.dtype == np.float64
        assert Ud.dtype == np.float64

        # Refracted direction should be unit vector
        assert np.isclose(np.linalg.norm(Ud), 1.0, rtol=1e-12)
