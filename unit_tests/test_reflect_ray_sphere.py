"""Unit tests for reflect_ray_sphere function."""

import numpy as np
from et_simul.optics.reflections import reflect_ray_sphere


def test_basic_center_reflection():
    """Test basic center front reflection with MATLAB reference values."""
    # Ray hitting sphere dead center from front
    R0 = np.array([0.0, 0.0, -5.0, 1.0])  # Ray origin
    Rd = np.array([0.0, 0.0, 1.0, 0.0])  # Ray direction (unit vector)
    S0 = np.array([0.0, 0.0, 0.0, 1.0])  # Sphere center
    Sr = 2.0  # Sphere radius

    U0, Ud = reflect_ray_sphere(R0, Rd, S0, Sr)

    # MATLAB reference values
    expected_U0 = np.array([0.0, 0.0, -2.0, 1.0])
    expected_Ud = np.array([0.0, 0.0, -1.0, 0.0])

    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-14, atol=1e-15)


def test_angled_reflection():
    """Test angled reflection with MATLAB reference values."""
    # Angled ray with non-normalized input direction
    R0 = np.array([0.0, 0.0, 0.0, 1.0])  # Ray origin
    Rd = np.array([1.0, 1.0, 1.0, 0.0])  # Ray direction (not unit)
    S0 = np.array([0.0, 0.0, 0.0, 1.0])  # Sphere center
    Sr = 1.5  # Sphere radius

    U0, Ud = reflect_ray_sphere(R0, Rd, S0, Sr)

    # MATLAB reference values
    expected_U0 = np.array([-0.8660254037844397, -0.8660254037844397, -0.8660254037844397, 1.0])
    expected_Ud = np.array([-0.5773502691896256, -0.5773502691896256, -0.5773502691896256, 0.0])

    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-14, atol=1e-15)

    # Test that reflected direction magnitude is normalized
    assert abs(np.linalg.norm(Ud) - 1.0) < 1e-14


def test_ray_missing_sphere():
    """Test ray that misses sphere - should return None."""
    # Ray that doesn't intersect sphere
    R0 = np.array([0.0, 0.0, 0.0, 1.0])  # Ray origin
    Rd = np.array([1.0, 0.0, 0.0, 0.0])  # Ray direction
    S0 = np.array([0.0, 5.0, 0.0, 1.0])  # Sphere center (away from ray)
    Sr = 1.0  # Sphere radius

    U0, Ud = reflect_ray_sphere(R0, Rd, S0, Sr)

    # Should return None for both when ray misses sphere
    assert U0 is None
    assert Ud is None


def test_homogeneous_coordinates():
    """Test with homogeneous coordinates and MATLAB reference values."""
    # 4D homogeneous coordinates from original working test
    R0 = np.array([1.0, 0.0, -4.0, 1.0])  # Ray origin (homogeneous)
    Rd = np.array([0.0, 0.0, 1.0, 0.0])  # Ray direction (homogeneous)
    S0 = np.array([1.0, 0.0, 0.0, 1.0])  # Sphere center (homogeneous)
    Sr = 2.0  # Sphere radius

    U0, Ud = reflect_ray_sphere(R0, Rd, S0, Sr)

    # MATLAB reference values (4D)
    expected_U0 = np.array(
        [
            1.0,
            0.0,
            -2.0,
            1.0,
        ]
    )
    expected_Ud = np.array(
        [
            0.0,
            0.0,
            -1.0,
            0.0,
        ]
    )

    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-14, atol=1e-15)

    # Test that output has correct 4D format
    assert U0.shape == (4,)
    assert Ud.shape == (4,)

    # Verify homogeneous components
    assert abs(U0[3] - 1.0) < 1e-15  # Point should have homogeneous component = 1
    assert abs(Ud[3] - 0.0) < 1e-15  # Direction should have homogeneous component = 0


def test_output_properties():
    """Test that output has correct properties."""
    R0 = np.array([0.0, 0.0, -5.0, 1.0])
    Rd = np.array([0.0, 0.0, 1.0, 0.0])
    S0 = np.array([0.0, 0.0, 0.0, 1.0])
    Sr = 2.0

    U0, Ud = reflect_ray_sphere(R0, Rd, S0, Sr)
    assert U0 is not None, "U0 should not be None for these inputs"
    assert Ud is not None, "Ud should not be None for these inputs"

    # Check types and shapes
    assert isinstance(U0, np.ndarray)
    assert isinstance(Ud, np.ndarray)
    assert U0.dtype == np.float64
    assert Ud.dtype == np.float64
    assert U0.shape == (4,)
    assert Ud.shape == (4,)

    # Reflected direction should be normalized

    assert np.isclose(np.linalg.norm(Ud[:3]), 1.0, rtol=1e-12)
