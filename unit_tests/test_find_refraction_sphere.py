"""Unit tests for find_refraction_sphere function."""

import numpy as np
from et_simul.optics.refractions import find_refraction_sphere


def test_basic_refraction():
    """Test basic refraction scenario with MATLAB reference values."""
    # Define sphere
    S0 = np.array([0.0, 0.0, 0.0])  # Sphere center
    Sr = 10.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    # Object inside sphere, camera outside
    O = np.array([2.0, 1.0, 3.0])
    C = np.array([15.0, 8.0, 12.0])

    I = find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere)

    # MATLAB reference values
    expected_I = np.array([6.6936966464330183, 3.5132501989855691, 6.5461055784992670])

    assert I is not None
    np.testing.assert_allclose(I, expected_I, rtol=1e-10, atol=1e-10)

    # Verify point is on sphere surface
    dist_from_center = np.linalg.norm(I - S0)
    np.testing.assert_allclose(dist_from_center, Sr, rtol=1e-15, atol=1e-15)


def test_different_refractive_indices():
    """Test refraction with different refractive indices and MATLAB reference values."""
    # Define sphere
    S0 = np.array([0.0, 0.0, 0.0])
    Sr = 10.0
    n_outside = 1.33  # Water
    n_sphere = 1.4  # Different glass

    # Different positions
    O = np.array([-1.0, 2.0, -2.0])
    C = np.array([-8.0, 10.0, -15.0])

    I = find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere)

    # MATLAB reference values
    expected_I = np.array([-3.9369221058636397, 5.3748224021012501, -7.4573405767896075])

    assert I is not None
    np.testing.assert_allclose(I, expected_I, rtol=1e-10, atol=1e-10)

    # Verify point is on sphere surface
    dist_from_center = np.linalg.norm(I - S0)
    np.testing.assert_allclose(dist_from_center, Sr, rtol=1e-15, atol=1e-15)


def test_large_sphere():
    """Test refraction with large sphere and MATLAB reference values."""
    # Define large sphere
    S0 = np.array([0.0, 0.0, 0.0])
    Sr = 50.0  # Large radius
    n_outside = 1.0
    n_sphere = 1.5

    # Scaled positions
    O = np.array([5.0, -3.0, 8.0])
    C = np.array([80.0, -20.0, 60.0])

    I = find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere)

    # MATLAB reference values
    expected_I = np.array([37.3698407698170811, -10.7448391945206190, 31.4331581538096536])

    assert I is not None
    np.testing.assert_allclose(I, expected_I, rtol=1e-10, atol=1e-10)

    # Verify point is on sphere surface
    dist_from_center = np.linalg.norm(I - S0)
    np.testing.assert_allclose(dist_from_center, Sr, rtol=1e-15, atol=1e-15)


def test_snells_law_verification():
    """Test that solution satisfies Snell's law with MATLAB reference values."""
    # Use same setup as basic test
    S0 = np.array([0.0, 0.0, 0.0])
    Sr = 10.0
    n_outside = 1.0
    n_sphere = 1.5
    O = np.array([2.0, 1.0, 3.0])
    C = np.array([15.0, 8.0, 12.0])

    I = find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere)
    assert I is not None

    # Compute vectors
    n_surface = (I - S0) / np.linalg.norm(I - S0)
    ray_incident = (I - O) / np.linalg.norm(I - O)
    ray_refracted = (C - I) / np.linalg.norm(C - I)

    # MATLAB reference values
    expected_n_surface = np.array([0.669370, 0.351325, 0.654611])
    expected_ray_incident = np.array([0.733730, 0.392877, 0.554336])
    expected_ray_refracted = np.array([0.761852, 0.411524, 0.500230])

    np.testing.assert_allclose(n_surface, expected_n_surface, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ray_incident, expected_ray_incident, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ray_refracted, expected_ray_refracted, rtol=1e-5, atol=1e-5)

    # Verify angles and Snell's law
    cos_theta_i = np.dot(n_surface, ray_incident)
    cos_theta_r = np.dot(n_surface, ray_refracted)
    sin_theta_i = np.sqrt(1 - cos_theta_i**2)
    sin_theta_r = np.sqrt(1 - cos_theta_r**2)

    # Check Snell's law: n1*sin(theta1) = n2*sin(theta2)
    snell_left = n_sphere * sin_theta_i
    snell_right = n_outside * sin_theta_r
    snell_diff = abs(snell_left - snell_right)

    assert snell_diff < 1e-10


def test_output_properties():
    """Test that output has correct properties."""
    S0 = np.array([0.0, 0.0, 0.0])
    Sr = 10.0
    n_outside = 1.0
    n_sphere = 1.5
    O = np.array([2.0, 1.0, 3.0])
    C = np.array([15.0, 8.0, 12.0])

    I = find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere)
    assert I is not None, "I should not be None for these inputs"
    assert not np.any(np.isnan(I)), "I should not contain NaN values"

    # Check types and shapes
    assert isinstance(I, np.ndarray)
    assert I.shape == (3,)
    assert I.dtype == np.float64
