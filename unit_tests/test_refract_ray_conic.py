"""Unit tests for refract_ray_conic function."""

from pyetsimul.optics.refractions import refract_ray_conic, refract_ray_sphere
from pyetsimul.types import Direction3D, Point3D, Position3D, Ray


def test_conic_equals_sphere_refraction() -> None:
    """Test conic refraction with k=0 gives sphere behavior."""
    # Off-axis ray for realistic refraction
    ray_origin = Point3D(-1.0, 0.5, -2.0)
    ray_direction = Direction3D(0.2, -0.1, 1.0).normalize()
    ray = Ray(origin=ray_origin, direction=ray_direction)

    radius = 1.0
    n_outside = 1.0
    n_inside = 1.5

    # For k=0 conic to match sphere at origin, conic center must be at (0,0,-R)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere at origin
    conic_center = Position3D(0.0, 0.0, 0)  # Conic center adjusted for k=0 equivalence

    # Sphere function with proper center
    sphere_intersection, sphere_ray = refract_ray_sphere(ray, sphere_center, radius, n_outside, n_inside)

    # Conic function with k=0
    k = 0.0  # k=0 represents a perfect sphere
    conic_intersection, conic_ray = refract_ray_conic(ray, conic_center, radius, k, n_outside, n_inside)

    # Should match the sphere centered at origin
    assert sphere_intersection is not None, "sphere_intersection should not be None"
    assert conic_intersection is not None, "conic_intersection should not be None"
    assert sphere_intersection.intersects, "sphere should intersect"
    assert conic_intersection.intersects, "conic should intersect"

    # Compare intersection points
    sphere_intersection.point.assert_close(conic_intersection.point, rtol=1e-12, atol=1e-14)

    assert sphere_ray is not None, "sphere_ray should not be None"
    assert conic_ray is not None, "conic_ray should not be None"

    # Compare refracted ray origins
    sphere_ray.origin.assert_close(conic_ray.origin, rtol=1e-12, atol=1e-14)

    # Compare refracted ray directions
    sphere_ray.direction.assert_close(conic_ray.direction, rtol=1e-12, atol=1e-14)
