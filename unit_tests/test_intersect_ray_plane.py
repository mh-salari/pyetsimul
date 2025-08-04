"""Unit tests for intersect_ray_plane function."""

import numpy as np
from et_simul.geometry.intersections import intersect_ray_plane
from et_simul.types import Ray, Position3D, Direction3D, IntersectionResult, Point3D, Vector3D


def test_normal_intersection():
    """Test normal ray-plane intersection with MATLAB reference values."""
    # Ray perpendicular to plane
    ray_origin = Point3D(0.0, 0.0, -2.0)  # Ray origin
    ray_direction = Vector3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    plane_point = Position3D(0.0, 0.0, 0.0)  # Point on plane
    plane_normal = Direction3D(0.0, 0.0, 1.0)  # Plane normal

    result = intersect_ray_plane(ray, plane_point, plane_normal)

    # MATLAB reference values
    expected_x = Point3D(0.0, 0.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_x, rtol=1e-14, atol=1e-15)


def test_ray_parallel_to_plane():
    """Test ray parallel to plane - should return None."""
    # Ray parallel to plane (no intersection)
    ray_origin = Point3D(0.0, 0.0, 1.0)  # Ray origin
    ray_direction = Vector3D(1.0, 0.0, 0.0)  # Ray direction (parallel to plane)
    ray = Ray(ray_origin, ray_direction)
    plane_point = Position3D(0.0, 0.0, 0.0)  # Point on plane
    plane_normal = Direction3D(0.0, 0.0, 1.0)  # Plane normal

    result = intersect_ray_plane(ray, plane_point, plane_normal)

    # Should return None for parallel ray
    assert not result.intersects


def test_ray_starting_on_plane():
    """Test ray starting on plane with MATLAB reference values."""
    # Ray origin on plane surface
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin (on plane)
    ray_direction = Vector3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    plane_point = Position3D(0.0, 0.0, 0.0)  # Point on plane
    plane_normal = Direction3D(0.0, 0.0, 1.0)  # Plane normal

    result = intersect_ray_plane(ray, plane_point, plane_normal)

    # MATLAB reference values
    expected_x = Point3D(0.0, 0.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_x, rtol=1e-14, atol=1e-15)


def test_oblique_intersection():
    """Test oblique intersection with tilted plane and MATLAB reference values."""
    # Angled ray with tilted plane
    ray_origin = Point3D(0.0, 0.0, -2.0)  # Ray origin
    ray_direction = Vector3D(1.0, 1.0, 1.0)  # Ray direction (diagonal)
    ray = Ray(ray_origin, ray_direction)
    plane_point = Position3D(1.0, 1.0, 0.0)  # Point on plane
    plane_normal = Direction3D(1.0, 1.0, 1.0)  # Plane normal (diagonal)

    result = intersect_ray_plane(ray, plane_point, plane_normal)

    # MATLAB reference values
    expected_x = Point3D(1.3333333333333333, 1.3333333333333333, -0.6666666666666667)

    assert result.intersects
    result.point.assert_close(expected_x, rtol=1e-14, atol=1e-15)

    # Verify point is on plane: (x-P0)·Pn = 0
    plane_check = (np.array(result.point)[:3] - np.array(plane_point)[:3]) @ np.array(plane_normal)[:3]
    assert abs(plane_check) < 1e-14


def test_backward_intersection():
    """Test ray pointing away from plane (backward intersection) with MATLAB reference values."""
    # Ray pointing away from plane
    ray_origin = Point3D(0.0, 0.0, 1.0)  # Ray origin
    ray_direction = Vector3D(0.0, 0.0, 1.0)  # Ray direction (away from plane)
    ray = Ray(ray_origin, ray_direction)
    plane_point = Position3D(0.0, 0.0, 0.0)  # Point on plane
    plane_normal = Direction3D(0.0, 0.0, 1.0)  # Plane normal

    result = intersect_ray_plane(ray, plane_point, plane_normal)

    # MATLAB reference values (backward intersection)
    expected_x = Point3D(0.0, 0.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_x, rtol=1e-14, atol=1e-15)


def test_xy_plane_intersection():
    """Test intersection with XY plane and MATLAB reference values."""
    # Ray hitting XY plane from above
    ray_origin = Point3D(1.0, 2.0, 3.0)  # Ray origin
    ray_direction = Vector3D(0.0, 0.0, -1.0)  # Ray direction (downward)
    ray = Ray(ray_origin, ray_direction)
    plane_point = Position3D(0.0, 0.0, 0.0)  # Point on plane
    plane_normal = Direction3D(0.0, 0.0, 1.0)  # Plane normal

    result = intersect_ray_plane(ray, plane_point, plane_normal)

    # MATLAB reference values
    expected_x = Point3D(1.0, 2.0, 0.0)

    assert result.intersects
    result.point.assert_close(expected_x, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    ray_origin = Point3D(0.0, 0.0, -2.0)
    ray_direction = Vector3D(0.0, 0.0, 1.0)
    ray = Ray(ray_origin, ray_direction)
    plane_point = Position3D(0.0, 0.0, 0.0)
    plane_normal = Direction3D(0.0, 0.0, 1.0)

    result = intersect_ray_plane(ray, plane_point, plane_normal)
    assert result.intersects, "result should indicate intersection for these inputs"

    # Check types and shapes
    assert isinstance(result, IntersectionResult)
    assert isinstance(result.point, Point3D)

    # Point should be on plane: (x-P0)·Pn = 0
    plane_equation = (np.array(result.point) - np.array(plane_point)[:3]) @ np.array(plane_normal)[:3]
    assert abs(plane_equation) < 1e-14

    # Point should be on ray: x = R0 + t*Rd for some t
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
