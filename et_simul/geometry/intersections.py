"""Ray-surface intersection and geometric calculation utilities for eye tracking simulation.

Provides functions for intersecting rays with spheres, circles, planes, and conic surfaces, as well as related geometric operations.
"""

import numpy as np
import warnings
from typing import Optional, Tuple
from ..types import Point3D, Vector3D, Ray, IntersectionResult, Position3D, Direction3D


def intersect_ray_sphere(
    ray: Ray, sphere_center: Position3D, sphere_radius: float
) -> Tuple[Optional[IntersectionResult], Optional[IntersectionResult]]:
    """Find intersection points between ray and sphere.

    Uses quadratic equation to find intersection points. Returns both intersection
    points ordered by distance from ray origin (closer first).

    Args:
        ray: Ray with origin and direction
        sphere_center: Sphere center position
        sphere_radius: Sphere radius

    Returns:
        Tuple of (closer_result, farther_result) where closer_result is closer intersection,
        farther_result is farther. Returns (None, None) if no intersection.
    """
    # Normalize ray direction
    direction_normalized = ray.direction.normalize()

    # Vector from sphere center to ray origin
    origin_to_center = ray.origin - sphere_center.to_point3d()

    # Quadratic equation coefficients
    b = 2 * direction_normalized.dot(origin_to_center)
    c = origin_to_center.dot(origin_to_center) - sphere_radius**2

    # Discriminant
    disc = b**2 - 4 * c

    if disc < 0:
        return IntersectionResult.no_intersection(), IntersectionResult.no_intersection()

    # Two solutions
    t1 = (-b + np.sqrt(disc)) / 2
    t2 = (-b - np.sqrt(disc)) / 2

    # Choose closer intersection first
    if abs(t1) < abs(t2):
        t_close, t_far = t1, t2
    else:
        t_close, t_far = t2, t1

    # Compute intersection points using structured type arithmetic
    point_close = ray.origin + direction_normalized * t_close
    point_far = ray.origin + direction_normalized * t_far

    return IntersectionResult.intersection_at(point_close, abs(t_close)), IntersectionResult.intersection_at(
        point_far, abs(t_far)
    )


def intersect_ray_circle(ray: Ray, circle_center: Point3D, circle_radius: float) -> Optional[IntersectionResult]:
    """Find intersection between ray and circle in 2D plane.

    Uses quadratic equation to find intersection points in x-y plane.
    Returns the closest intersection point to ray origin.

    Args:
        ray: Ray with origin and direction
        circle_center: Circle center (2D using x,y components)
        circle_radius: Circle radius

    Returns:
        Intersection result with closest point to ray origin, or None if no intersection
    """
    # Normalize direction of ray (use only x,y components for 2D)
    direction_2d = Vector3D(ray.direction.x, ray.direction.y, 0).normalize()

    # Vector from ray origin to circle center (2D)
    origin_to_center = Vector3D(ray.origin.x - circle_center.x, ray.origin.y - circle_center.y, 0)

    # Quadratic equation coefficients
    b = 2 * direction_2d.dot(origin_to_center)
    c = origin_to_center.dot(origin_to_center) - circle_radius**2

    # Discriminant
    disc = b**2 - 4 * c

    if disc < 0:
        return IntersectionResult.no_intersection()

    # Two solutions
    t1 = (-b + np.sqrt(disc)) / 2
    t2 = (-b - np.sqrt(disc)) / 2

    # Choose closest intersection
    t = t1 if abs(t1) < abs(t2) else t2

    # Calculate intersection point (2D result in 3D space with z from ray)
    intersection_point = Point3D(
        ray.origin.x + t * direction_2d.x,
        ray.origin.y + t * direction_2d.y,
        ray.origin.z,  # Keep original z coordinate
    )

    return IntersectionResult.intersection_at(intersection_point, abs(t))


def intersect_ray_plane(ray: Ray, plane_point: Position3D, plane_normal: Direction3D) -> Optional[IntersectionResult]:
    """Find intersection between ray and plane.

    Solves the parametric ray equation with the plane equation using dot product.
    Returns intersection point with surface normal.

    Args:
        ray: Ray with origin and direction
        plane_point: Point on plane
        plane_normal: Plane normal vector

    Returns:
        Intersection result, or None if ray is parallel to plane
    """
    # Normalize plane normal
    normal_normalized = plane_normal.normalize()

    # Check if ray is parallel to plane
    denom = ray.direction.dot(normal_normalized)

    if abs(denom) < 1e-15:  # Ray is parallel to plane
        return IntersectionResult.no_intersection()

    # Vector from ray origin to plane point
    origin_to_plane = plane_point.to_point3d() - ray.origin

    # Solve for parameter t: (ray.origin + t*ray.direction - plane_point) · plane_normal = 0
    t = origin_to_plane.dot(normal_normalized) / denom

    # Calculate intersection point
    intersection_point = ray.point_at(t)

    return IntersectionResult.intersection_at(intersection_point, abs(t), normal_normalized)


def intersect_ray_conic(
    ray: Ray, conic_center: Position3D, radius: float, conic_constant: float
) -> Tuple[Optional[IntersectionResult], Optional[IntersectionResult]]:
    """Find intersection between ray and conic section.

    Uses quadratic equation derived from conic surface equation.
    Returns both intersection points ordered by distance from ray origin.

    Args:
        ray: Ray with origin and direction
        conic_center: Conic center position
        radius: Radius parameter (R in the formula, meters)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Tuple (closer_result, farther_result): Intersection results closer and farther from ray origin.
        Returns (None, None) if no intersection.
    """
    # Normalize ray direction
    direction_normalized = ray.direction.normalize()

    # Ray parameters
    x0, y0, z0 = ray.origin.x, ray.origin.y, ray.origin.z
    dx, dy, dz = direction_normalized.x, direction_normalized.y, direction_normalized.z

    # Conic translation parameters - standard translation for apex positioning
    cx, cy = conic_center.x, conic_center.y
    cz = conic_center.z - radius / (1 + conic_constant)  # Standard -R/(1+k) translation

    # Translated conic equation: (x-cx)² + (y-cy)² + (1+k)(z-cz)² - 2*R*(z-cz) = 0
    # Substitute ray equation: (x0+t*dx-cx)² + (y0+t*dy-cy)² + (1+k)(z0+t*dz-cz)² - 2*R*(z0+t*dz-cz) = 0
    A = dx**2 + dy**2 + (1 + conic_constant) * dz**2
    B = 2 * ((x0 - cx) * dx + (y0 - cy) * dy + (1 + conic_constant) * (z0 - cz) * dz - radius * dz)
    C = (x0 - cx) ** 2 + (y0 - cy) ** 2 + (1 + conic_constant) * (z0 - cz) ** 2 - 2 * radius * (z0 - cz)

    # Solve quadratic equation
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None, None  # No real roots, no intersection

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)

    # Filter out intersections behind the ray origin (t < 0)
    ts = [t for t in [t1, t2] if t >= 0]

    if len(ts) == 0:
        return None, None
    elif len(ts) == 1:
        point = ray.point_at(ts[0])
        return IntersectionResult.intersection_at(point, ts[0]), None
    else:
        ts.sort()
        point1 = ray.point_at(ts[0])
        point2 = ray.point_at(ts[1])

        result1 = IntersectionResult.intersection_at(point1, ts[0])
        result2 = IntersectionResult.intersection_at(point2, ts[1])

        return result1, result2


def conic_surface_normal(point: Point3D, conic_center: Position3D, radius: float, conic_constant: float) -> Vector3D:
    """Calculate surface normal at a point on conic section surface.

    Uses gradient of conic equation to compute normal vector.
    Handles degenerate case at conic apex.

    Args:
        point: Point on conic surface
        conic_center: Conic center position
        radius: Radius parameter (R in the formula, meters)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Unit normal vector pointing outward from conic surface
    """
    # Conic translation parameters - standard translation for apex positioning
    cx, cy = conic_center.x, conic_center.y
    cz = conic_center.z - radius / (1 + conic_constant)  # Standard -R/(1+k) translation

    # Translate to conic coordinate system
    x = point.x - cx
    y = point.y - cy
    z = point.z - cz

    # Calculate gradient: ∇F = (2(x-cx), 2(y-cy), 2(1+k)(z-cz) - 2R)
    normal_x = 2 * x
    normal_y = 2 * y
    normal_z = 2 * (1 + conic_constant) * z - 2 * radius

    normal = Vector3D(normal_x, normal_y, normal_z)

    # Normalize to unit vector
    try:
        return normal.normalize()
    except ValueError:
        warnings.warn("Degenerate normal vector at conic apex", RuntimeWarning)
        return Vector3D(0, 0, 1)  # Default to z-axis


def point_on_conic_surface(
    conic_center: Position3D, direction: Vector3D, radius: float, conic_constant: float
) -> Optional[Point3D]:
    """Calculate point on conic surface given direction from start point.

    Uses quadratic equation to find intersection of ray with conic surface.
    Chooses best intersection point based on direction alignment.

    Args:
        conic_center: Starting point of the ray
        direction: Direction vector (will be normalized)
        radius: Radius parameter of the conic (R)
        conic_constant: Shape parameter of the conic (k)

    Returns:
        Point on conic surface, or None if no intersection
    """
    # Normalize direction vector
    direction_normalized = direction.normalize()

    # Ray parameters
    x0, y0, z0 = conic_center.x, conic_center.y, conic_center.z
    dx, dy, dz = direction_normalized.x, direction_normalized.y, direction_normalized.z

    # Conic translation parameters - standard translation for apex positioning
    cx, cy = conic_center.x, conic_center.y
    cz = conic_center.z - radius / (1 + conic_constant)  # Standard -R/(1+k) translation

    # Ray equation: P(t) = (x0, y0, z0) + t * (dx, dy, dz)
    # Substitute into conic: (x-cx)² + (y-cy)² + (1+k)(z-cz)² - 2R(z-cz) = 0
    # (x0+t*dx-cx)² + (y0+t*dy-cy)² + (1+k)(z0+t*dz-cz)² - 2R(z0+t*dz-cz) = 0

    # Expand and collect terms: At² + Bt + C = 0
    A = dx**2 + dy**2 + (1 + conic_constant) * dz**2
    B = 2 * ((x0 - cx) * dx + (y0 - cy) * dy + (1 + conic_constant) * (z0 - cz) * dz - radius * dz)
    C = (x0 - cx) ** 2 + (y0 - cy) ** 2 + (1 + conic_constant) * (z0 - cz) ** 2 - 2 * radius * (z0 - cz)

    # Solve quadratic equation
    discriminant = B**2 - 4 * A * C

    if discriminant < 0:
        return None  # No real intersections

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-B + sqrt_disc) / (2 * A)
    t2 = (-B - sqrt_disc) / (2 * A)

    # Choose the appropriate solution - typically the one that goes in the intended direction
    candidates = []
    for t in [t1, t2]:
        if abs(t) > 1e-12:  # Non-trivial solution
            point = Point3D(x0 + t * dx, y0 + t * dy, z0 + t * dz)
            candidates.append((t, point))

    if not candidates:
        return None

    if len(candidates) == 1:
        return candidates[0][1]

    # Multiple candidates: choose the one that aligns best with direction
    best_point = None
    best_dot = -np.inf

    for t, point in candidates:
        # Vector from start point to intersection point
        start_to_point = point - conic_center.to_point3d()
        # How well does this align with intended direction?
        dot_product = start_to_point.dot(direction_normalized)
        if dot_product > best_dot:
            best_dot = dot_product
            best_point = point

    return best_point
