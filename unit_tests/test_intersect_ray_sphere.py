"""Unit tests for intersect_ray_sphere function."""

import numpy as np
from pyetsimul.geometry.intersections import intersect_ray_sphere
from pyetsimul.types import Ray, Position3D, IntersectionResult, Point3D, Direction3D


def test_two_intersections():
    """Test ray intersecting sphere with two points and MATLAB reference values."""
    # Ray passes through sphere center
    ray_origin = Point3D(0.0, 0.0, -2.0)  # Ray origin
    ray_direction = Direction3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 1.0  # Sphere radius

    result1, result2 = intersect_ray_sphere(ray, sphere_center, sphere_radius)

    # MATLAB reference values
    expected_pos1 = Point3D(0.0, 0.0, -1.0)
    expected_pos2 = Point3D(0.0, 0.0, 1.0)

    assert result1.intersects
    assert result2.intersects
    result1.point.assert_close(expected_pos1, rtol=1e-14, atol=1e-15)
    result2.point.assert_close(expected_pos2, rtol=1e-14, atol=1e-15)


def test_tangent_intersection():
    """Test ray tangent to sphere and MATLAB reference values."""
    # Ray just touches sphere surface
    ray_origin = Point3D(0.0, 1.0, -2.0)  # Ray origin
    ray_direction = Direction3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 1.0  # Sphere radius

    result1, result2 = intersect_ray_sphere(ray, sphere_center, sphere_radius)

    # MATLAB reference values (both identical for tangent)
    expected_pos = Point3D(0.0, 1.0, 0.0)

    assert result1.intersects
    assert result2.intersects
    result1.point.assert_close(expected_pos, rtol=1e-14, atol=1e-15)
    result2.point.assert_close(expected_pos, rtol=1e-14, atol=1e-15)


def test_ray_missing_sphere():
    """Test ray that misses sphere - should return None."""
    # Ray misses sphere completely
    ray_origin = Point3D(0.0, 2.0, -2.0)  # Ray origin
    ray_direction = Direction3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 1.0  # Sphere radius

    result1, result2 = intersect_ray_sphere(ray, sphere_center, sphere_radius)

    # Should return None for both when ray misses sphere
    assert not result1.intersects
    assert not result2.intersects


def test_ray_inside_sphere():
    """Test ray starting inside sphere and MATLAB reference values."""
    # Ray origin at sphere center
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin (at center)
    ray_direction = Direction3D(1.0, 0.0, 0.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 2.0  # Sphere radius

    result1, result2 = intersect_ray_sphere(ray, sphere_center, sphere_radius)

    # MATLAB reference values
    expected_pos1 = Point3D(-2.0, 0.0, 0.0)
    expected_pos2 = Point3D(2.0, 0.0, 0.0)

    assert result1.intersects
    assert result2.intersects
    result1.point.assert_close(expected_pos1, rtol=1e-14, atol=1e-15)
    result2.point.assert_close(expected_pos2, rtol=1e-14, atol=1e-15)


def test_non_unit_direction():
    """Test with non-unit ray direction and MATLAB reference values."""
    # Ray direction with length 2.0 (should be normalized)
    ray_origin = Point3D(0.0, 0.0, -3.0)  # Ray origin
    ray_direction = Direction3D(0.0, 0.0, 2.0)  # Ray direction (length 2.0)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 1.0  # Sphere radius

    result1, result2 = intersect_ray_sphere(ray, sphere_center, sphere_radius)

    # MATLAB reference values (same as unit direction case)
    expected_pos1 = Point3D(0.0, 0.0, -1.0)
    expected_pos2 = Point3D(0.0, 0.0, 1.0)

    assert result1.intersects
    assert result2.intersects
    result1.point.assert_close(expected_pos1, rtol=1e-14, atol=1e-15)
    result2.point.assert_close(expected_pos2, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    ray_origin = Point3D(0.0, 0.0, -2.0)
    ray_direction = Direction3D(0.0, 0.0, 1.0)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)
    sphere_radius = 1.0

    result1, result2 = intersect_ray_sphere(ray, sphere_center, sphere_radius)
    assert result1.intersects, "result1 should indicate intersection for these inputs"
    assert result2.intersects, "result2 should indicate intersection for these inputs"

    # Check types
    assert isinstance(result1, IntersectionResult)
    assert isinstance(result2, IntersectionResult)
    assert isinstance(result1.point, Point3D)
    assert isinstance(result2.point, Point3D)

    # Both points should be on sphere surface
    dist1 = result1.point.distance_to(sphere_center.to_point3d())
    dist2 = result2.point.distance_to(sphere_center.to_point3d())
    assert np.isclose(dist1, sphere_radius, rtol=1e-12)
    assert np.isclose(dist2, sphere_radius, rtol=1e-12)

    # result1 should be closer to ray origin than result2
    dist_to_origin1 = result1.point.distance_to(ray_origin)
    dist_to_origin2 = result2.point.distance_to(ray_origin)
    assert dist_to_origin1 <= dist_to_origin2
