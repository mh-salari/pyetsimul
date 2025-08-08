"""Unit tests for reflect_ray_sphere function."""

import numpy as np
from pyetsimul.optics.reflections import reflect_ray_sphere
from pyetsimul.types import Ray, Point3D, Vector3D, Position3D, Direction3D, IntersectionResult


def test_basic_center_reflection():
    """Test basic center front reflection with MATLAB reference values."""
    # Ray hitting sphere dead center from front
    ray_origin = Point3D(0.0, 0.0, -5.0)  # Ray origin
    ray_direction = Vector3D(0.0, 0.0, 1.0)  # Ray direction (unit vector)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 2.0  # Sphere radius

    intersection_result, reflected_ray = reflect_ray_sphere(ray, sphere_center, sphere_radius)

    # MATLAB reference values
    expected_intersection = Point3D(0.0, 0.0, -2.0)
    expected_direction = Vector3D(0.0, 0.0, -1.0)

    assert intersection_result is not None
    assert reflected_ray is not None
    assert intersection_result.intersects
    intersection_result.point.assert_close(expected_intersection, rtol=1e-14, atol=1e-15)
    reflected_ray.direction.assert_close(expected_direction, rtol=1e-14, atol=1e-15)


def test_angled_reflection():
    """Test angled reflection with MATLAB reference values."""
    # Angled ray with non-normalized input direction
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Vector3D(1.0, 1.0, 1.0)  # Ray direction (not unit)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 1.5  # Sphere radius

    intersection_result, reflected_ray = reflect_ray_sphere(ray, sphere_center, sphere_radius)

    # MATLAB reference values
    expected_intersection = Point3D(-0.8660254037844397, -0.8660254037844397, -0.8660254037844397)
    expected_direction = Vector3D(-0.5773502691896256, -0.5773502691896256, -0.5773502691896256)

    assert intersection_result is not None
    assert reflected_ray is not None
    assert intersection_result.intersects
    intersection_result.point.assert_close(expected_intersection, rtol=1e-14, atol=1e-15)
    reflected_ray.direction.assert_close(expected_direction, rtol=1e-14, atol=1e-15)

    # Test that reflected direction magnitude is normalized
    assert abs(reflected_ray.direction.magnitude() - 1.0) < 1e-14


def test_ray_missing_sphere():
    """Test ray that misses sphere - should return None."""
    # Ray that doesn't intersect sphere
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Vector3D(1.0, 0.0, 0.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 5.0, 0.0)  # Sphere center (away from ray)
    sphere_radius = 1.0  # Sphere radius

    intersection_result, reflected_ray = reflect_ray_sphere(ray, sphere_center, sphere_radius)

    # Should return None for both when ray misses sphere
    assert intersection_result is None
    assert reflected_ray is None


def test_homogeneous_coordinates():
    """Test with homogeneous coordinates and MATLAB reference values."""
    # Test data that was originally in homogeneous coordinates
    ray_origin = Point3D(1.0, 0.0, -4.0)  # Ray origin
    ray_direction = Vector3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(1.0, 0.0, 0.0)  # Sphere center
    sphere_radius = 2.0  # Sphere radius

    intersection_result, reflected_ray = reflect_ray_sphere(ray, sphere_center, sphere_radius)

    # MATLAB reference values
    expected_intersection = Point3D(1.0, 0.0, -2.0)
    expected_direction = Vector3D(0.0, 0.0, -1.0)

    assert intersection_result is not None
    assert reflected_ray is not None
    assert intersection_result.intersects
    intersection_result.point.assert_close(expected_intersection, rtol=1e-14, atol=1e-15)
    reflected_ray.direction.assert_close(expected_direction, rtol=1e-14, atol=1e-15)

    # Test that output can be converted to correct 4D homogeneous format
    intersection_homogeneous = np.array(intersection_result.point.to_position3d())
    direction_homogeneous = np.array(Direction3D.from_vector3d(reflected_ray.direction))
    assert intersection_homogeneous.shape == (4,)
    assert direction_homogeneous.shape == (4,)

    # Verify homogeneous components
    assert abs(intersection_homogeneous[3] - 1.0) < 1e-15  # Point should have homogeneous component = 1
    assert abs(direction_homogeneous[3] - 0.0) < 1e-15  # Direction should have homogeneous component = 0


def test_output_properties():
    """Test that output has correct properties."""
    ray_origin = Point3D(0.0, 0.0, -5.0)
    ray_direction = Vector3D(0.0, 0.0, 1.0)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 0.0)
    sphere_radius = 2.0

    intersection_result, reflected_ray = reflect_ray_sphere(ray, sphere_center, sphere_radius)
    assert intersection_result is not None, "intersection_result should not be None for these inputs"
    assert reflected_ray is not None, "reflected_ray should not be None for these inputs"

    # Check types and properties
    assert isinstance(intersection_result, IntersectionResult)
    assert isinstance(reflected_ray, Ray)
    assert isinstance(intersection_result.point, Point3D)
    assert isinstance(reflected_ray.direction, Vector3D)
    assert intersection_result.intersects

    # Check that arrays can be converted with correct shapes
    intersection_array = np.array(intersection_result.point)
    direction_array = np.array(reflected_ray.direction)
    assert intersection_array.shape == (3,)
    assert direction_array.shape == (3,)

    # Reflected direction should be normalized
    assert np.isclose(reflected_ray.direction.magnitude(), 1.0, rtol=1e-12)
