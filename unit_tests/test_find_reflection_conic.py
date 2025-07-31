"""Unit tests for find_reflection_conic function."""

import numpy as np
from et_simul.optics.reflections import find_reflection_sphere, find_reflection_conic


def test_conic_equals_sphere_reflection():
    """Test conic reflection with k=0 gives sphere behavior."""
    # Realistic eye tracking geometry based on example.py
    L = np.array([200e-3, 0, 0])  # Light position (200mm in x)
    C = np.array([0, 0, -50e-3])  # Camera position (50mm in front, facing -z)
    radius = 7.98e-3  # Realistic corneal radius

    # For k=0, both should create equivalent spheres at the same location
    sphere_center = np.array([0.0, 0.0, 0.0])  # Sphere at origin
    conic_center = np.array([0.0, 0.0, 0.0])  # Conic also at origin

    # Sphere function with proper center
    sphere_U0 = find_reflection_sphere(L, C, sphere_center, radius)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_U0 = find_reflection_conic(L, C, conic_center, r_param, k)

    # Should match exactly since both create identical spheres
    assert sphere_U0 is not None, "Sphere reflection should not be None for this camera position"
    assert conic_U0 is not None, "Conic reflection should not be None for this camera position"
    np.testing.assert_allclose(sphere_U0, conic_U0, rtol=1e-12, atol=1e-14)


def test_conic_output_properties():
    """Test that conic reflection output has correct properties."""
    L = np.array([200e-3, 0, 0])  # Realistic light position
    C = np.array([0, 0, -50e-3])  # Realistic camera position
    S0 = np.array([0.0, 0.0, 0.0])  # Conic center at origin
    r_param = 7.98e-3  # Radius parameter
    k = -0.18  # Typical prolate cornea

    U0 = find_reflection_conic(L, C, S0, r_param, k)
    assert U0 is not None, "U0 should not be None for this camera position"

    # Check types and shapes
    assert isinstance(U0, np.ndarray)
    assert U0.dtype == np.float64
    assert U0.shape == (3,)

    # Should be finite values
    assert np.all(np.isfinite(U0))


def test_prolate_vs_oblate_conic():
    """Test that prolate and oblate conics give different results."""
    L = np.array([200e-3, 0, 0])  # Realistic light position
    C = np.array([0, 0, -50e-3])  # Realistic camera position
    S0 = np.array([0.0, 0.0, 0.0])  # Conic center at origin
    r_param = 7.98e-3  # Radius parameter

    # Prolate conic (typical cornea)
    k_prolate = -0.18
    U0_prolate = find_reflection_conic(L, C, S0, r_param, k_prolate)

    # Oblate conic
    k_oblate = 0.18
    U0_oblate = find_reflection_conic(L, C, S0, r_param, k_oblate)

    # Should give different results (unless by coincidence)
    assert U0_prolate is not None, "U0_prolate should not be None for this camera position"
    assert U0_oblate is not None, "U0_oblate should not be None for this camera position"

    # Results should be different for different asphericity
    assert not np.allclose(U0_prolate, U0_oblate, rtol=1e-10)
