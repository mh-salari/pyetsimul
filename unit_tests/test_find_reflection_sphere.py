"""Unit tests for find_reflection_sphere function."""

import numpy as np
from et_simul.optics.reflections import find_reflection_sphere


def test_specific_reflection_case():
    """Test reflection calculation with MATLAB reference values."""
    # Test parameters from MATLAB
    L = np.array([0.25, 0, 0, 1])  # Light position
    C = np.array([0.1, -0.1, 0, 1])  # Camera position
    S0 = np.array([0.05, 0, 1.0, 1.0])  # Sphere center
    Sr = 0.8  # Sphere radius

    U0 = find_reflection_sphere(L, C, S0, Sr)

    # MATLAB reference values
    expected_U0 = np.array([0.1472592734935877, -0.0408954315633531, 0.2069878958069135, 1])

    np.testing.assert_allclose(U0, expected_U0, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    L = np.array([0.25, 0, 0, 1])
    C = np.array([0.1, -0.1, 0, 1])
    S0 = np.array([0.05, 0, 1.0, 1.0])
    Sr = 0.8

    U0 = find_reflection_sphere(L, C, S0, Sr)

    assert U0 is not None, "U0 should not be None for these inputs"
    # Check types and shapes
    assert isinstance(U0, np.ndarray)
    assert U0.dtype == np.float64
    assert U0.shape == (4,)

    # Should be finite values
    assert np.all(np.isfinite(U0))
