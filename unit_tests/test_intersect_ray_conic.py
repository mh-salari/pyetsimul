"""Unit tests for intersect_ray_conic function."""

from pyetsimul.geometry.intersections import intersect_ray_sphere, intersect_ray_conic
from pyetsimul.types import Ray, Position3D, Point3D, Direction3D, IntersectionResult


def test_conic_equals_sphere_on_axis():
    """Test conic with k=0 gives sphere behavior.

    Note: For k=0, the conic equation x² + y² + z² - 2*R*z = 0 produces a sphere
    centered at (0,0,R). To match a sphere at origin, conic center must be at (0,0,-R).
    """
    # On-axis ray
    ray_origin = Point3D(0.0, 0.0, -2.0)  # Ray origin
    ray_direction = Direction3D(0.0, 0.0, 1.0)  # Ray direction
    ray = Ray(ray_origin, ray_direction)
    radius = 1.0

    # For k=0 conic to match sphere at origin, conic center must be at (0,0,-R)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere at origin
    conic_center = Position3D(0.0, 0.0, 0)  # Conic center adjusted for k=0 equivalence

    # Sphere function with proper center
    sphere_result1, sphere_result2 = intersect_ray_sphere(ray, sphere_center, radius)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_result1, conic_result2 = intersect_ray_conic(ray, conic_center, r_param, k)

    # Should match the sphere centered at origin
    assert sphere_result1.intersects and conic_result1.intersects
    assert sphere_result2.intersects and conic_result2.intersects
    sphere_result1.point.assert_close(conic_result1.point, rtol=1e-12, atol=1e-14)
    sphere_result2.point.assert_close(conic_result2.point, rtol=1e-12, atol=1e-14)


def test_conic_equals_sphere_off_axis():
    """Test conic with k=0 gives sphere behavior for off-axis ray.

    Note: For k=0, the conic equation x² + y² + z² - 2*R*z = 0 produces a sphere
    centered at (0,0,R). To match a sphere at origin, conic center must be at (0,0,-R).
    """
    # Off-axis ray
    ray_origin = Point3D(-1.5, 0.5, -2.0)
    ray_direction = Direction3D(0.3, -0.1, 1.0).normalize()
    ray = Ray(ray_origin, ray_direction)
    radius = 1.0

    # For k=0 conic to match sphere at origin, conic center must be at (0,0,-R)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere at origin
    conic_center = Position3D(0.0, 0.0, 0)  # Conic center adjusted for k=0 equivalence

    # Sphere function with proper center
    sphere_result1, sphere_result2 = intersect_ray_sphere(ray, sphere_center, radius)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_result1, conic_result2 = intersect_ray_conic(ray, conic_center, r_param, k)

    # Should match the sphere centered at origin
    assert sphere_result1.intersects and conic_result1.intersects
    assert sphere_result2.intersects and conic_result2.intersects
    sphere_result1.point.assert_close(conic_result1.point, rtol=1e-12, atol=1e-14)
    sphere_result2.point.assert_close(conic_result2.point, rtol=1e-12, atol=1e-14)
