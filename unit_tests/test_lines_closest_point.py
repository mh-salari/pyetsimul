"""Unit tests for lines_closest_point function."""

import numpy as np
from et_simul.geometry.utils import lines_closest_point


def test_parallel_lines():
    """Test parallel lines - should return NaN values."""
    p1 = np.array([0, 0, 0, 1], dtype=float)
    d1 = np.array([1, 0, 0, 0], dtype=float)
    p2 = np.array([0, 2, 0, 1], dtype=float)
    d2 = np.array([1, 0, 0, 0], dtype=float)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # For parallel lines, should return NaN like MATLAB
    assert np.all(np.isnan(x1))
    assert np.all(np.isnan(x2))


def test_intersecting_lines():
    """Test intersecting lines - closest points should be identical."""
    p1 = np.array([0, 0, 0, 1], dtype=float)
    d1 = np.array([1, 1, 0, 0], dtype=float)
    p2 = np.array([2, 0, 0, 1], dtype=float)
    d2 = np.array([-1, 1, 0, 0], dtype=float)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # MATLAB reference values
    expected = np.array([1.0, 1.0, 0.0, 1.0])

    np.testing.assert_allclose(x1, expected, rtol=1e-12)
    np.testing.assert_allclose(x2, expected, rtol=1e-12)

    # Both points should be identical for intersecting lines
    np.testing.assert_allclose(x1, x2, rtol=1e-12)


def test_skew_lines():
    """Test skew lines - should return different closest points."""
    p1 = np.array([0, 0, 0, 1], dtype=float)
    d1 = np.array([1, 0, 0, 0], dtype=float)
    p2 = np.array([0, 1, 1, 1], dtype=float)
    d2 = np.array([0, 1, 0, 0], dtype=float)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # MATLAB reference values
    expected_x1 = np.array([0.0, 0.0, 0.0, 1.0])
    expected_x2 = np.array([0.0, 0.0, 1.0, 1.0])

    np.testing.assert_allclose(x1, expected_x1, rtol=1e-12)
    np.testing.assert_allclose(x2, expected_x2, rtol=1e-12)

    # Points should be different for skew lines
    assert not np.allclose(x1, x2, rtol=1e-12)


def test_non_unit_direction_vectors():
    """Test with non-unit direction vectors."""
    p1 = np.array([0, 0, 0, 1], dtype=float)
    d1 = np.array([3, 0, 0, 0], dtype=float)
    p2 = np.array([1, 1, 0, 1], dtype=float)
    d2 = np.array([0, 2, 0, 0], dtype=float)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # MATLAB reference values - these lines actually intersect
    expected = np.array([1.0, 0.0, 0.0, 1.0])

    np.testing.assert_allclose(x1, expected, rtol=1e-12)
    np.testing.assert_allclose(x2, expected, rtol=1e-12)


def test_output_properties():
    """Test that output has correct properties."""
    p1 = np.array([0, 0, 0, 1], dtype=float)
    d1 = np.array([1, 0, 0, 0], dtype=float)
    p2 = np.array([0, 1, 1, 1], dtype=float)
    d2 = np.array([0, 1, 0, 0], dtype=float)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # Check types and shapes
    assert isinstance(x1, np.ndarray)
    assert isinstance(x2, np.ndarray)
    assert x1.dtype == np.float64
    assert x2.dtype == np.float64
    assert x1.shape == (4,)
    assert x2.shape == (4,)

    # For non-parallel cases, values should be finite
    assert np.all(np.isfinite(x1))
    assert np.all(np.isfinite(x2))
