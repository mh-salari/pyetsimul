"""Unit tests for reflect_ray_circle function."""

import numpy as np

from pyetsimul.optics.reflections import reflect_ray_circle
from pyetsimul.types import Direction3D, Point3D, Ray


def test_basic_center_reflection() -> None:
    """Test basic center front reflection with MATLAB reference values."""
    # Ray hitting circle dead center from front
    ray_origin = Point3D(0.0, -3.0, 0.0)  # Ray origin
    ray_direction = Direction3D(0.0, 1.0, 0.0)  # Ray direction (unit vector)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 2.0  # Circle radius

    source_point, source_direction = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_source_point = Point3D(0.0, -2.0, 0.0)
    # expected_source_direction = Vector3D(0.0, -1.0, 0.0)

    assert source_point is not None
    assert source_direction is not None
    source_point.point.assert_close(expected_source_point, rtol=1e-14, atol=1e-15)


def test_angled_reflection() -> None:
    """Test angled reflection with MATLAB reference values."""
    # Angled ray with non-normalized input direction
    ray_origin = Point3D(-2.0, -2.0, 0.0)  # Ray origin
    ray_direction = Direction3D(1.0, 1.0, 0.0)  # Ray direction (not normalized)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.5  # Circle radius

    source_point, source_direction = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_source_point = Point3D(-1.0606601717798214, -1.0606601717798214, 0.0)
    expected_source_direction = Direction3D(-0.7071067811865472, -0.7071067811865472, 0.0)

    assert source_point is not None
    assert source_direction is not None
    source_point.point.assert_close(expected_source_point, rtol=1e-14, atol=1e-15)
    source_direction.direction.assert_close(expected_source_direction, rtol=1e-14, atol=1e-15)

    # Test that reflected direction is normalized
    assert abs(source_direction.direction.magnitude() - 1.0) < 1e-14


def test_offset_circle() -> None:
    """Test reflection with offset circle and MATLAB reference values."""
    # Circle not centered at origin
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Direction3D(0.6, 0.8, 0.0)  # Ray direction (not unit)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(2.0, 2.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    source_point, source_direction = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_source_point = Point3D(1.1300909166052995, 1.5067878888070660, 0.0)
    expected_source_direction = Direction3D(-0.9945696889543475, -0.1040727332842388, 0.0)

    assert source_point is not None
    assert source_direction is not None
    source_point.point.assert_close(expected_source_point, rtol=1e-14, atol=1e-15)
    source_direction.direction.assert_close(expected_source_direction, rtol=1e-14, atol=1e-15)


def test_grazing_incidence() -> None:
    """Test grazing incidence with MATLAB reference values."""
    # Ray parallel to tangent
    ray_origin = Point3D(-3.0, 0.0, 0.0)  # Ray origin
    ray_direction = Direction3D(1.0, 0.0, 0.0)  # Ray direction (horizontal)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    source_point, source_direction = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_source_point = Point3D(-1.0, 0.0, 0.0)
    expected_source_direction = Direction3D(-1.0, 0.0, 0.0)

    assert source_point is not None
    assert source_direction is not None
    source_point.point.assert_close(expected_source_point, rtol=1e-14, atol=1e-15)
    source_direction.direction.assert_close(expected_source_direction, rtol=1e-14, atol=1e-15)


def test_ray_missing_circle() -> None:
    """Test ray that misses circle - should return None."""
    # Ray that doesn't intersect circle
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Direction3D(1.0, 0.0, 0.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 3.0, 0.0)  # Circle center (away from ray)
    circle_radius = 1.0  # Circle radius

    source_point, source_direction = reflect_ray_circle(ray, circle_center, circle_radius)

    # Should return None for both when ray misses circle
    assert source_point is None
    assert source_direction is None


def test_output_properties() -> None:
    """Test that output has correct properties."""
    ray_origin = Point3D(0.0, -3.0, 0.0)
    ray_direction = Direction3D(0.0, 1.0, 0.0)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)
    circle_radius = 2.0

    source_point, source_direction = reflect_ray_circle(ray, circle_center, circle_radius)
    assert source_point is not None, "source_point should not be None for these inputs"
    assert source_direction is not None, "source_direction should not be None for these inputs"

    # Check types and shapes
    assert isinstance(source_point.point, Point3D)
    assert isinstance(source_direction, Ray)
    assert isinstance(source_direction.direction, Direction3D)

    # Reflected direction should be normalized
    assert np.isclose(source_direction.direction.magnitude(), 1.0, rtol=1e-12)
