"""Unit tests for refract_ray_sphere function."""

import numpy as np
from pyetsimul.optics.refractions import refract_ray_sphere
from pyetsimul.types import Ray, Point3D, Vector3D, Position3D, IntersectionResult


def test_basic_refraction():
    """Test basic refraction scenario with MATLAB reference values."""
    # Ray entering sphere (relevant to corneal refraction)
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Vector3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 5.0)  # Sphere center
    sphere_radius = 2.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass/cornea

    intersection_result, refracted_ray = refract_ray_sphere(ray, sphere_center, sphere_radius, n_outside, n_sphere)

    # MATLAB reference values
    expected_intersection = Point3D(0.0, 0.0, 3.0)
    expected_direction = Vector3D(0.0, 0.0, 1.0)

    assert intersection_result is not None
    assert refracted_ray is not None
    assert intersection_result.intersects
    intersection_result.point.assert_close(expected_intersection, rtol=1e-12, atol=1e-14)
    refracted_ray.direction.assert_close(expected_direction, rtol=1e-12, atol=1e-14)


def test_oblique_incidence():
    """Test oblique incidence refraction with MATLAB reference values."""
    # Diagonal ray - relevant to off-axis eye tracking scenarios
    ray_origin = Point3D(-3.0, 0.0, 0.0)  # Ray origin
    ray_direction = Vector3D(1.0 / np.sqrt(2), 0.0, 1.0 / np.sqrt(2))  # Ray direction (diagonal, normalized)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 5.0)  # Sphere center
    sphere_radius = 2.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    intersection_result, refracted_ray = refract_ray_sphere(ray, sphere_center, sphere_radius, n_outside, n_sphere)

    # MATLAB reference values with many decimals (converted to 4D homogeneous)
    # Values recalculated for normalized diagonal ray direction
    expected_intersection = Point3D(-1.332268e-15, 0.0, 3.0)
    expected_direction = Vector3D(0.4714045207910319, 0.0, 0.8819171036881968)

    assert intersection_result is not None
    assert refracted_ray is not None
    assert intersection_result.intersects
    intersection_result.point.assert_close(expected_intersection, rtol=1e-12, atol=1e-14)
    refracted_ray.direction.assert_close(expected_direction, rtol=1e-12, atol=1e-14)


def test_ray_missing_sphere():
    """Test case where ray misses sphere - should return None."""
    ray_origin = Point3D(0.0, 0.0, 0.0)  # Ray origin
    ray_direction = Vector3D(1.0, 0.0, 0.0)  # Ray direction (misses sphere)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 5.0)  # Sphere center
    sphere_radius = 1.0  # Sphere radius
    n_outside = 1.0  # Air
    n_sphere = 1.5  # Glass

    intersection_result, refracted_ray = refract_ray_sphere(ray, sphere_center, sphere_radius, n_outside, n_sphere)

    # Should return None for both when ray misses sphere
    assert intersection_result is None
    assert refracted_ray is None


def test_output_properties():
    """Test that output has correct properties."""
    ray_origin = Point3D(0.0, 0.0, 0.0)
    ray_direction = Vector3D(0.0, 0.0, 1.0)
    ray = Ray(ray_origin, ray_direction)
    sphere_center = Position3D(0.0, 0.0, 5.0)
    sphere_radius = 2.0
    n_outside = 1.0
    n_sphere = 1.5

    intersection_result, refracted_ray = refract_ray_sphere(ray, sphere_center, sphere_radius, n_outside, n_sphere)
    assert intersection_result is not None, "intersection_result should not be None for these inputs"
    assert refracted_ray is not None, "refracted_ray should not be None for these inputs"

    # Check types and properties
    assert isinstance(intersection_result, IntersectionResult)
    assert isinstance(refracted_ray, Ray)
    assert isinstance(intersection_result.point, Point3D)
    assert isinstance(refracted_ray.direction, Vector3D)
    assert intersection_result.intersects

    # Check that arrays can be converted with correct shapes
    intersection_array = np.array(intersection_result.point)
    direction_array = np.array(refracted_ray.direction)
    assert intersection_array.shape == (3,)
    assert direction_array.shape == (3,)

    # Refracted direction should be normalized
    assert np.isclose(refracted_ray.direction.magnitude(), 1.0, rtol=1e-12)
