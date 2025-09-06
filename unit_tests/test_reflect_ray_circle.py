"""Unit tests for reflect_ray_circle function."""

import numpy as np
from pyetsimul.optics.reflections import reflect_ray_circle
from pyetsimul.types import Ray, Point3D, Direction3D


def test_basic_center_reflection():
    """Test basic center front reflection with MATLAB reference values."""
    # Ray hitting circle dead center from front
    ray_origin = Point3D(0.0, -3.0, 0.0)  # Ray origin
    ray_direction = Direction3D(0.0, 1.0, 0.0)  # Ray direction (unit vector)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 2.0  # Circle radius

    S0, Sd = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_S0 = Point3D(0.0, -2.0, 0.0)
    # expected_Sd = Vector3D(0.0, -1.0, 0.0)

    assert S0 is not None
    assert Sd is not None
    S0.point.assert_close(expected_S0, rtol=1e-14, atol=1e-15)


def test_angled_reflection():
    """Test angled reflection with MATLAB reference values."""
    # Angled ray with non-normalized input direction
    ray_origin = Point3D(-2.0, -2.0, 0.0)  # Ray origin
    ray_direction = Direction3D(1.0, 1.0, 0.0)  # Ray direction (not normalized)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.5  # Circle radius

    S0, Sd = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_S0 = Point3D(-1.0606601717798214, -1.0606601717798214, 0.0)
    expected_Sd = Direction3D(-0.7071067811865472, -0.7071067811865472, 0.0)

    assert S0 is not None
    assert Sd is not None
    S0.point.assert_close(expected_S0, rtol=1e-14, atol=1e-15)
    Sd.direction.assert_close(expected_Sd, rtol=1e-14, atol=1e-15)

    # Test that reflected direction is normalized
    assert abs(Sd.direction.magnitude() - 1.0) < 1e-14


def test_offset_circle():
    """Test reflection with offset circle and MATLAB reference values."""
    # Circle not centered at origin
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Direction3D(0.6, 0.8, 0.0)  # Ray direction (not unit)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(2.0, 2.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    S0, Sd = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_S0 = Point3D(1.1300909166052995, 1.5067878888070660, 0.0)
    expected_Sd = Direction3D(-0.9945696889543475, -0.1040727332842388, 0.0)

    assert S0 is not None
    assert Sd is not None
    S0.point.assert_close(expected_S0, rtol=1e-14, atol=1e-15)
    Sd.direction.assert_close(expected_Sd, rtol=1e-14, atol=1e-15)


def test_grazing_incidence():
    """Test grazing incidence with MATLAB reference values."""
    # Ray parallel to tangent
    ray_origin = Point3D(-3.0, 0.0, 0.0)  # Ray origin
    ray_direction = Direction3D(1.0, 0.0, 0.0)  # Ray direction (horizontal)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    S0, Sd = reflect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_S0 = Point3D(-1.0, 0.0, 0.0)
    expected_Sd = Direction3D(-1.0, 0.0, 0.0)

    assert S0 is not None
    assert Sd is not None
    S0.point.assert_close(expected_S0, rtol=1e-14, atol=1e-15)
    Sd.direction.assert_close(expected_Sd, rtol=1e-14, atol=1e-15)


def test_ray_missing_circle():
    """Test ray that misses circle - should return None."""
    # Ray that doesn't intersect circle
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Direction3D(1.0, 0.0, 0.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 3.0, 0.0)  # Circle center (away from ray)
    circle_radius = 1.0  # Circle radius

    S0, Sd = reflect_ray_circle(ray, circle_center, circle_radius)

    # Should return None for both when ray misses circle
    assert S0 is None
    assert Sd is None


def test_output_properties():
    """Test that output has correct properties."""
    ray_origin = Point3D(0.0, -3.0, 0.0)
    ray_direction = Direction3D(0.0, 1.0, 0.0)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)
    circle_radius = 2.0

    S0, Sd = reflect_ray_circle(ray, circle_center, circle_radius)
    assert S0 is not None, "S0 should not be None for these inputs"
    assert Sd is not None, "Sd should not be None for these inputs"

    # Check types and shapes
    assert isinstance(S0.point, Point3D)
    assert isinstance(Sd, Ray)
    assert isinstance(Sd.direction, Direction3D)

    # Reflected direction should be normalized
    assert np.isclose(Sd.direction.magnitude(), 1.0, rtol=1e-12)
