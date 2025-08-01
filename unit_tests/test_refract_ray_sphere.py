"""Unit tests for refract_ray_sphere function."""

import numpy as np
from et_simul.optics.refractions import refract_ray_sphere


def test_basic_refraction():
    """Test basic refraction scenario with MATLAB reference values."""
    # Ray entering sphere (relevant to corneal refraction)
    R0 = np.array([0.0, 0.0, 0.0, 1.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0, 0.0])  # Ray direction
    S0 = np.array([0.0, 0.0, 5.0, 1.0])  # Sphere center
    Sr = 2.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass/cornea

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)

    # MATLAB reference values (converted to 4D homogeneous coordinates)
    expected_U0 = np.array([0.0, 0.0, 3.0, 1.0])
    expected_Ud = np.array([0.0, 0.0, 1.0, 0.0])

    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-12, atol=1e-14)


def test_oblique_incidence():
    """Test oblique incidence refraction with MATLAB reference values."""
    # Diagonal ray - relevant to off-axis eye tracking scenarios
    R0 = np.array([-3.0, 0.0, 0.0, 1.0])  # Ray origin
    Rd = np.array([1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2), 0.0])  # Ray direction (diagonal, normalized)
    S0 = np.array([0.0, 0.0, 5.0, 1.0])  # Sphere center
    Sr = 2.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)

    # MATLAB reference values with many decimals (converted to 4D homogeneous)
    # Values recalculated for normalized diagonal ray direction
    expected_U0 = np.array([-1.332268e-15, 0.0, 3.0, 1.0])
    expected_Ud = np.array([0.4714045207910319, 0.0, 0.8819171036881968, 0.0])

    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-12, atol=1e-14)


def test_ray_missing_sphere():
    """Test case where ray misses sphere - should return None."""
    R0 = np.array([0.0, 0.0, 0.0, 1.0])  # Ray origin
    Rd = np.array([1.0, 0.0, 0.0, 0.0])  # Ray direction (misses sphere)
    S0 = np.array([0.0, 0.0, 5.0, 1.0])  # Sphere center
    Sr = 1.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)

    # Should return None for both when ray misses sphere
    assert U0 is None
    assert Ud is None


def test_output_properties():
    """Test that output has correct properties."""
    R0 = np.array([0.0, 0.0, 0.0, 1.0])
    Rd = np.array([0.0, 0.0, 1.0, 0.0])
    S0 = np.array([0.0, 0.0, 5.0, 1.0])
    Sr = 2.0
    n_outside = 1.0
    n_sphere = 1.5

    U0, Ud = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere)
    assert U0 is not None, "U0 should not be None for these inputs"
    assert Ud is not None, "Ud should not be None for these inputs"

    # Check types and shapes
    assert isinstance(U0, np.ndarray)
    assert isinstance(Ud, np.ndarray)
    assert U0.shape == (4,)
    assert Ud.shape == (4,)
    assert U0.dtype == np.float64
    assert Ud.dtype == np.float64

    # Check homogeneous coordinate values
    assert U0[3] == 1.0  # Position coordinate
    assert Ud[3] == 0.0  # Direction coordinate

    # Refracted direction should be unit vector (3D components only)
    assert np.isclose(np.linalg.norm(Ud[:3]), 1.0, rtol=1e-12)
