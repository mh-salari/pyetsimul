"""Unit tests for find_refraction_conic function."""

import numpy as np
from et_simul.optics.refractions import find_refraction_sphere, find_refraction_conic


def test_conic_equals_sphere_refraction():
    """Test conic refraction with Q=0 gives sphere behavior (accounting for center shift)."""
    # Realistic eye tracking geometry based on example.py
    radius = 7.98e-3  # Realistic corneal radius
    n_outside = 1.0  # Air
    n_inside = 1.376  # Cornea

    # Object behind cornea (pupil), camera in front
    O = np.array([0.0, 0.0, -6e-3])  # Object 6mm behind origin (pupil)
    C = np.array([0.0, 0.0, 50e-3])  # Camera 50mm in front

    # For Q=0, we want both to create equivalent spheres at the same location
    sphere_center = np.array([0.0, 0.0, 0.0])  # Sphere at origin
    conic_center = np.array([0.0, 0.0, radius])  # Conic offset by +r to create sphere at origin

    # Sphere function with proper center
    sphere_I = find_refraction_sphere(C, O, sphere_center, radius, n_outside, n_inside)

    # Conic function with Q=0
    r_apical = radius
    Q = 0.0  # Q=0 represents a perfect sphere
    conic_I = find_refraction_conic(C, O, conic_center, r_apical, Q, n_outside, n_inside)

    # Should match exactly since both create identical spheres
    assert sphere_I is not None, "sphere_I should not be None for this test case"
    assert conic_I is not None, "conic_I should not be None for this test case"
    np.testing.assert_allclose(sphere_I, conic_I, rtol=1e-12, atol=1e-14)


def test_conic_refraction_basic():
    """Test basic conic refraction scenario."""
    # Realistic conic parameters
    S0 = np.array([0.0, 0.0, 7.98e-3])  # Conic center at realistic position
    r_apical = 7.98e-3  # Realistic apical radius
    Q = -0.18  # Typical prolate cornea
    n_outside = 1.0  # Air
    n_conic = 1.376  # Cornea

    # Realistic eye tracking geometry
    O = np.array([0.0, 0.0, -6e-3])  # Object behind cornea (pupil)
    C = np.array([0.0, 0.0, 50e-3])  # Camera in front

    I = find_refraction_conic(C, O, S0, r_apical, Q, n_outside, n_conic)
    assert I is not None, "I should not be None for this basic refraction scenario"

    # Check types and shapes
    assert isinstance(I, np.ndarray)
    assert I.shape == (3,)
    assert I.dtype == np.float64

    # Should be finite values
    assert np.all(np.isfinite(I))


def test_different_q_values():
    """Test that different Q values give different refraction results."""
    S0 = np.array([0.0, 0.0, 7.98e-3])  # Realistic conic center
    r_apical = 7.98e-3  # Realistic apical radius
    n_outside = 1.0
    n_conic = 1.376
    O = np.array([0.0, 0.0, -6e-3])  # Realistic object position (pupil)
    C = np.array([0.0, 0.0, 50e-3])  # Realistic camera position

    # Prolate conic (typical cornea)
    Q_prolate = -0.18
    I_prolate = find_refraction_conic(C, O, S0, r_apical, Q_prolate, n_outside, n_conic)

    # Oblate conic
    Q_oblate = 0.18
    I_oblate = find_refraction_conic(C, O, S0, r_apical, Q_oblate, n_outside, n_conic)

    # Sphere (Q=0)
    Q_sphere = 0.0
    I_sphere = find_refraction_conic(C, O, S0, r_apical, Q_sphere, n_outside, n_conic)

    # All should give different results (unless by coincidence)
    assert I_prolate is not None, "I_prolate should not be None"
    assert I_oblate is not None, "I_oblate should not be None"
    assert I_sphere is not None, "I_sphere should not be None"

    # Results should be different for different asphericity
    assert not np.allclose(I_prolate, I_oblate, rtol=1e-10)
    assert not np.allclose(I_prolate, I_sphere, rtol=1e-10)
    assert not np.allclose(I_oblate, I_sphere, rtol=1e-10)


def test_conic_output_properties():
    """Test that conic refraction output has correct properties."""
    S0 = np.array([0.0, 0.0, 7.98e-3])  # Realistic conic center
    r_apical = 7.98e-3  # Realistic apical radius
    Q = -0.18
    n_outside = 1.0
    n_conic = 1.376
    O = np.array([0.0, 0.0, -6e-3])  # Realistic object position
    C = np.array([0.0, 0.0, 50e-3])  # Realistic camera position

    I = find_refraction_conic(C, O, S0, r_apical, Q, n_outside, n_conic)
    assert I is not None, "I should not be None for these inputs"

    # Check types and shapes
    assert isinstance(I, np.ndarray)
    assert I.dtype == np.float64
    assert I.shape == (3,)

    # Should be finite values
    assert np.all(np.isfinite(I))


def test_realistic_corneal_parameters():
    """Test with realistic corneal parameters."""
    # Realistic corneal geometry
    S0 = np.array([0.0, 0.0, -0.00435])  # Cornea center position
    r_apical = 7.98e-3  # 7.98mm apical radius
    Q = -0.18  # Typical anterior cornea asphericity
    n_outside = 1.0  # Air
    n_cornea = 1.376  # Cornea

    # Object behind cornea (pupil), camera in front
    O = np.array([0.0, 0.0, -0.006])  # Object 6mm behind origin
    C = np.array([0.0, 0.0, 0.05])  # Camera 5cm in front

    I = find_refraction_conic(C, O, S0, r_apical, Q, n_outside, n_cornea)
    assert I is not None, "I should not be None for realistic corneal parameters"

    # Should be somewhere on the corneal surface
    assert isinstance(I, np.ndarray)
    assert I.shape == (3,)
    assert np.all(np.isfinite(I))

    # Z-coordinate should be near corneal surface
    # For our conic opening toward -Z, the surface extends from center toward -Z
    assert I[2] <= S0[2]  # Should be at or behind cornea center
    # The intersection can be anywhere on the conic surface, so just check it's reasonable
    assert abs(I[2] - S0[2]) < 0.05  # Within 5cm of center (very loose bounds)
