"""Unit tests for line_intersect_2d function."""

import numpy as np
from et_simul.geometry.utils import line_intersect_2d


def test_horizontal_vertical_lines():
    """Test intersection of horizontal and vertical lines with MATLAB reference values."""
    # Horizontal line: y = 0.5, points (0, 0.5) and (1, 0.5)
    p1 = np.array([0.0, 0.5])
    p2 = np.array([1.0, 0.5])

    # Vertical line: x = 0.3, points (0.3, 0) and (0.3, 1)
    p3 = np.array([0.3, 0.0])
    p4 = np.array([0.3, 1.0])

    intersection = line_intersect_2d(p1, p2, p3, p4)

    # MATLAB reference values
    expected_intersection = np.array([0.3, 0.5])

    np.testing.assert_allclose(intersection, expected_intersection, rtol=1e-14, atol=1e-15)


def test_non_integer_coordinates():
    """Test intersection with non-integer coordinates and MATLAB reference values."""
    # Line 1: from (0.1, 0.2) to (0.9, 0.8)
    p1 = np.array([0.1, 0.2])
    p2 = np.array([0.9, 0.8])

    # Line 2: from (0.2, 0.7) to (0.8, 0.3)
    p3 = np.array([0.2, 0.7])
    p4 = np.array([0.8, 0.3])

    intersection = line_intersect_2d(p1, p2, p3, p4)

    # MATLAB reference values
    expected_intersection = np.array([0.5, 0.5])

    np.testing.assert_allclose(intersection, expected_intersection, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    # Simple intersecting lines
    p1 = np.array([0.0, 0.0])
    p2 = np.array([1.0, 1.0])
    p3 = np.array([0.0, 1.0])
    p4 = np.array([1.0, 0.0])

    intersection = line_intersect_2d(p1, p2, p3, p4)

    # Check types and shapes
    assert isinstance(intersection, np.ndarray)
    assert intersection.dtype == np.float64
    assert intersection.shape == (2,)

    # Should be finite values for intersecting lines
    assert np.all(np.isfinite(intersection))

    # Test parallel lines (should return NaN like MATLAB)
    p1_parallel = np.array([0.0, 0.0])
    p2_parallel = np.array([1.0, 1.0])
    p3_parallel = np.array([0.0, 1.0])
    p4_parallel = np.array([1.0, 2.0])  # Parallel to first line

    intersection_parallel = line_intersect_2d(p1_parallel, p2_parallel, p3_parallel, p4_parallel)

    # Should return NaN for parallel lines (like MATLAB)
    assert np.all(np.isnan(intersection_parallel))
