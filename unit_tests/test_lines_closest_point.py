"""Unit tests for lines_closest_point function."""

import numpy as np

from pyetsimul.geometry.utils import lines_closest_point
from pyetsimul.types import Point3D, Vector3D


def test_parallel_lines() -> None:
    """Test parallel lines - should return NaN values."""
    p1 = Point3D(0, 0, 0)
    d1 = Vector3D(1, 0, 0)
    p2 = Point3D(0, 2, 0)
    d2 = Vector3D(1, 0, 0)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # For parallel lines, should return NaN like MATLAB
    assert np.isnan(x1.x)
    assert np.isnan(x1.y)
    assert np.isnan(x1.z)
    assert np.isnan(x2.x)
    assert np.isnan(x2.y)
    assert np.isnan(x2.z)


def test_intersecting_lines() -> None:
    """Test intersecting lines - closest points should be identical."""
    p1 = Point3D(0, 0, 0)
    d1 = Vector3D(1, 1, 0)
    p2 = Point3D(2, 0, 0)
    d2 = Vector3D(-1, 1, 0)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # Expected intersection point
    expected_point = Point3D(1.0, 1.0, 0.0)

    np.testing.assert_allclose([x1.x, x1.y, x1.z], [expected_point.x, expected_point.y, expected_point.z], rtol=1e-12)
    np.testing.assert_allclose([x2.x, x2.y, x2.z], [expected_point.x, expected_point.y, expected_point.z], rtol=1e-12)

    # Both points should be identical for intersecting lines
    assert abs(x1.x - x2.x) < 1e-12
    assert abs(x1.y - x2.y) < 1e-12
    assert abs(x1.z - x2.z) < 1e-12


def test_skew_lines() -> None:
    """Test skew lines - should return different closest points."""
    p1 = Point3D(0, 0, 0)
    d1 = Vector3D(1, 0, 0)
    p2 = Point3D(0, 1, 1)
    d2 = Vector3D(0, 1, 0)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # Expected closest points
    expected_x1 = Point3D(0.0, 0.0, 0.0)
    expected_x2 = Point3D(0.0, 0.0, 1.0)

    np.testing.assert_allclose([x1.x, x1.y, x1.z], [expected_x1.x, expected_x1.y, expected_x1.z], rtol=1e-12)
    np.testing.assert_allclose([x2.x, x2.y, x2.z], [expected_x2.x, expected_x2.y, expected_x2.z], rtol=1e-12)

    # Points should be different for skew lines
    assert not (abs(x1.x - x2.x) < 1e-12 and abs(x1.y - x2.y) < 1e-12 and abs(x1.z - x2.z) < 1e-12)


def test_non_unit_direction_vectors() -> None:
    """Test with non-unit direction vectors."""
    p1 = Point3D(0, 0, 0)
    d1 = Vector3D(3, 0, 0)
    p2 = Point3D(1, 1, 0)
    d2 = Vector3D(0, 2, 0)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # Expected intersection point - these lines actually intersect
    expected = Point3D(1.0, 0.0, 0.0)

    np.testing.assert_allclose([x1.x, x1.y, x1.z], [expected.x, expected.y, expected.z], rtol=1e-12)
    np.testing.assert_allclose([x2.x, x2.y, x2.z], [expected.x, expected.y, expected.z], rtol=1e-12)


def test_output_properties() -> None:
    """Test that output has correct properties."""
    p1 = Point3D(0, 0, 0)
    d1 = Vector3D(1, 0, 0)
    p2 = Point3D(0, 1, 1)
    d2 = Vector3D(0, 1, 0)

    x1, x2 = lines_closest_point(p1, d1, p2, d2)

    # Check types
    assert isinstance(x1, Point3D)
    assert isinstance(x2, Point3D)

    # For non-parallel cases, values should be finite
    assert np.isfinite(x1.x)
    assert np.isfinite(x1.y)
    assert np.isfinite(x1.z)
    assert np.isfinite(x2.x)
    assert np.isfinite(x2.y)
    assert np.isfinite(x2.z)
