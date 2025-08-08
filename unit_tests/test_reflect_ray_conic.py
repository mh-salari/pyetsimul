"""Unit tests for reflect_ray_conic function."""

from pyetsimul.optics.reflections import reflect_ray_sphere, reflect_ray_conic
from pyetsimul.types import Ray, Point3D, Vector3D, Position3D


def test_conic_equals_sphere_reflection():
    """Test conic reflection with k=0 gives sphere behavior."""
    # Off-axis ray for realistic reflection
    ray_origin = Point3D(-1.0, 0.5, -2.0)
    ray_direction = Vector3D(0.2, -0.1, 1.0).normalize()
    ray = Ray(origin=ray_origin, direction=ray_direction)
    radius = 1.0

    # For k=0 conic to match sphere at origin, conic center must be at (0,0,-R)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere at origin
    conic_center = Position3D(0.0, 0.0, 0)  # Conic center adjusted for k=0 equivalence

    # Sphere function with proper center
    sphere_intersection, sphere_ray = reflect_ray_sphere(ray, sphere_center, radius)

    # Conic function with k=0
    k = 0.0  # k=0 represents a perfect sphere
    conic_intersection, conic_ray = reflect_ray_conic(ray, conic_center, radius, k)

    # Should match the sphere centered at origin
    assert sphere_intersection is not None, "sphere_intersection should not be None"
    assert conic_intersection is not None, "conic_intersection should not be None"
    sphere_intersection.point.assert_close(conic_intersection.point, rtol=1e-12, atol=1e-14)

    assert sphere_ray is not None, "sphere_ray should not be None"
    assert conic_ray is not None, "conic_ray should not be None"
    sphere_ray.direction.assert_close(conic_ray.direction, rtol=1e-12, atol=1e-14)
