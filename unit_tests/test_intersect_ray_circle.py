"""Unit tests for intersect_ray_circle function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_circle
from et_simul.types import Ray, Point3D, Vector3D, IntersectionResult


def test_normal_intersection():
    """Test ray intersecting circle with MATLAB reference values."""
    # Ray hitting circle from below (returns closest intersection)
    ray_origin = Point3D(0.0, -2.0, 0.0)  # Ray origin (z=0 for 2D context)
    ray_direction = Vector3D(0.0, 1.0, 0.0)  # Ray direction (z=0 for 2D context)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center (z=0 for 2D context)
    circle_radius = 1.0  # Circle radius

    result = intersect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_pos = Point3D(0.0, -1.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_pos, rtol=1e-14, atol=1e-15)


def test_tangent_intersection():
    """Test ray tangent to circle with MATLAB reference values."""
    # Ray just touching circle surface
    ray_origin = Point3D(1.0, -2.0, 0.0)  # Ray origin
    ray_direction = Vector3D(0.0, 1.0, 0.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    result = intersect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_pos = Point3D(1.0, 0.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_pos, rtol=1e-14, atol=1e-15)


def test_ray_missing_circle():
    """Test ray that misses circle - should return None."""
    # Ray misses circle completely
    ray_origin = Point3D(2.0, -2.0, 0.0)  # Ray origin
    ray_direction = Vector3D(0.0, 1.0, 0.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    result = intersect_ray_circle(ray, circle_center, circle_radius)

    # Should return None when ray misses circle
    assert not result.intersects


def test_ray_inside_circle():
    """Test ray starting inside circle with MATLAB reference values."""
    # Ray origin at circle center
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin (at center)
    ray_direction = Vector3D(1.0, 0.0, 0.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 2.0  # Circle radius

    result = intersect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values (returns the intersection behind the ray origin)
    expected_pos = Point3D(-2.0, 0.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_pos, rtol=1e-14, atol=1e-15)


def test_non_unit_direction():
    """Test with non-unit ray direction and MATLAB reference values."""
    # Ray direction with length 2.0 (should be normalized)
    ray_origin = Point3D(0.0, -3.0, 0.0)  # Ray origin
    ray_direction = Vector3D(0.0, 2.0, 0.0)  # Ray direction (length 2.0)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    result = intersect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values (same as unit direction case)
    expected_pos = Point3D(0.0, -1.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_pos, rtol=1e-14, atol=1e-15)


def test_diagonal_intersection():
    """Test diagonal ray intersection with MATLAB reference values."""
    # Ray moving diagonally
    ray_origin = Point3D(-2.0, -2.0, 0.0)  # Ray origin
    ray_direction = Vector3D(1.0, 1.0, 0.0)  # Ray direction (diagonal)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)  # Circle center
    circle_radius = 1.0  # Circle radius

    result = intersect_ray_circle(ray, circle_center, circle_radius)

    # MATLAB reference values
    expected_pos = Point3D(-0.7071067811865472, -0.7071067811865472, 0.0)

    assert result.intersects
    result.point.assert_close(expected_pos, rtol=1e-14, atol=1e-15)

    # Verify distance from center matches MATLAB precision
    dist_from_center = result.point.distance_to(circle_center)
    expected_dist = 0.9999999999999996  # MATLAB's exact value
    assert abs(dist_from_center - expected_dist) < 1e-15


def test_output_properties():
    """Test that output has correct properties."""
    ray_origin = Point3D(0.0, -2.0, 0.0)
    ray_direction = Vector3D(0.0, 1.0, 0.0)
    ray = Ray(ray_origin, ray_direction)
    circle_center = Point3D(0.0, 0.0, 0.0)
    circle_radius = 1.0

    result = intersect_ray_circle(ray, circle_center, circle_radius)
    assert result.intersects, "result should indicate intersection for these inputs"

    # Check types and shapes
    assert isinstance(result, IntersectionResult)
    assert isinstance(result.point, Point3D)

    # Point should be on circle surface
    dist_to_center = result.point.distance_to(circle_center)
    assert np.isclose(dist_to_center, circle_radius, rtol=1e-12)

    # Point should be on ray: pos = R0 + t*Rd for some t
    # Reconstruct the ray from the origin and normalized direction
    if not np.allclose(np.array(ray_direction), 0):
        # Find t value for the intersection point
        # Assuming ray_direction is not zero vector
        t_val = (
            (result.point.x - ray_origin.x) / ray_direction.x
            if ray_direction.x != 0
            else (result.point.y - ray_origin.y) / ray_direction.y
            if ray_direction.y != 0
            else (result.point.z - ray_origin.z) / ray_direction.z
        )

        pos_on_ray = (
            ray_origin.x + t_val * ray_direction.x,
            ray_origin.y + t_val * ray_direction.y,
            ray_origin.z + t_val * ray_direction.z,
        )

        result.point.assert_close(Point3D(*pos_on_ray), rtol=1e-12)
