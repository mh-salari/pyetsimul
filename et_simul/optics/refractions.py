"""Light refraction calculation utilities for eye tracking simulation.

Implements Snell's law, ray-surface intersection, and optimization for refraction on spherical and conic surfaces.
"""

import numpy as np
from typing import Optional, Tuple
from scipy.optimize import brentq
from ..types import Point3D, Ray, IntersectionResult, Position3D, Vector3D
from ..geometry.intersections import intersect_ray_sphere, intersect_ray_conic, conic_surface_normal
from ..geometry.intersections import point_on_conic_surface


def _refraction_objective_sphere(
    a: float,
    camera_pos: Position3D,
    object_pos: Position3D,
    sphere_center: Position3D,
    sphere_radius: float,
    n_outside: float,
    n_sphere: float,
) -> Tuple[float, Point3D]:
    """Objective function for refraction finding on sphere.

    Uses interpolation between camera and object directions to find refraction point.
    Returns Snell's law difference for optimization.

    Args:
        a: Interpolation parameter between camera and object directions
        camera_pos: Camera position
        object_pos: Object position
        sphere_center: Sphere center position
        sphere_radius: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Tuple of (diff, surface_point) where diff is Snell's law difference and surface_point is on sphere surface
    """
    # Compute vectors from sphere center to camera and object
    to_camera = (camera_pos - sphere_center).normalize()
    to_object = (object_pos - sphere_center).normalize()

    # Interpolate and normalize to get surface normal
    normal_vec = (to_camera * a + to_object * (1 - a)).normalize()

    # Compute point on surface of sphere
    surface_point = sphere_center.to_point3d() + (normal_vec * sphere_radius)

    # Compute angles with surface normal
    camera_to_surface = (camera_pos - surface_point).normalize()
    surface_to_object = (surface_point - object_pos.to_point3d()).normalize()

    cos_angle_c = normal_vec.dot(camera_to_surface)
    cos_angle_o = normal_vec.dot(surface_to_object)

    # Safe sqrt to handle numerical errors
    sin_angle_c = np.sqrt(max(0, 1 - cos_angle_c**2))
    sin_angle_o = np.sqrt(max(0, 1 - cos_angle_o**2))

    # Snell's law difference
    diff = n_outside * sin_angle_c - n_sphere * sin_angle_o

    return diff, surface_point


def find_refraction_sphere(
    camera_pos: Position3D,
    object_pos: Position3D,
    sphere_center: Position3D,
    sphere_radius: float,
    n_outside: float,
    n_sphere: float,
) -> Optional[Point3D]:
    """Find refraction point on sphere surface.

    Uses optimization to find point where object ray refracts to camera position.
    Implements Snell's law using numerical root finding.

    Args:
        camera_pos: Camera/observer position
        object_pos: Object position inside sphere
        sphere_center: Sphere center position
        sphere_radius: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Position on sphere surface where refraction occurs, or None if not found.
    """
    try:
        a = brentq(
            lambda x: _refraction_objective_sphere(
                x, camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
            )[0],
            0,
            1,
        )
        _, result = _refraction_objective_sphere(
            a, camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
        )
        return result
    except (ValueError, RuntimeError):
        return None


def _refraction_objective_conic(
    alpha: float,
    camera_pos: Position3D,
    object_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
    n_outside: float,
    n_conic: float,
) -> Tuple[float, Optional[Point3D]]:
    """Objective function for refraction finding on conic surface.

    Uses interpolation between camera and object directions to find refraction point.
    Projects to conic surface using proper conic section geometry.
    Returns Snell's law difference for optimization.

    Args:
        alpha: Interpolation parameter between camera and object directions
        camera_pos: Camera position
        object_pos: Object position
        conic_center: Conic center position (typically corneal apex)
        radius: Radius of curvature at apex (meters)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic
        n_conic: Refractive index of conic

    Returns:
        Tuple of (diff, intersection) where diff is Snell's law difference and intersection is surface point
    """

    # Calculate directions from conic center to camera and object
    to_camera = (camera_pos - conic_center).normalize()
    to_object = (object_pos - conic_center).normalize()

    # Interpolate direction
    interpolated = to_camera * alpha + to_object * (1 - alpha)
    if interpolated.magnitude() == 0:
        # Handle collinear case: camera and object are on opposite sides of conic center
        # Use perpendicular direction to break symmetry
        direction = Vector3D(1.0, 0.0, 0.0) if abs(to_camera.x) < 0.9 else Vector3D(0.0, 1.0, 0.0)
    else:
        direction = interpolated.normalize()

    # Find intersection with conic surface along this direction
    intersection = point_on_conic_surface(conic_center, direction, radius, conic_constant)
    if intersection is None:
        return float("inf"), None

    # Get surface normal at intersection point
    normal = conic_surface_normal(intersection, conic_center, radius, conic_constant)

    # Compute angles with surface normal
    camera_to_intersection = (camera_pos - intersection).normalize()
    intersection_to_object = (intersection - object_pos).normalize()

    cos_angle_c = normal.dot(camera_to_intersection)
    cos_angle_o = normal.dot(intersection_to_object)

    sin_angle_c = np.sqrt(max(0, 1 - cos_angle_c**2))
    sin_angle_o = np.sqrt(max(0, 1 - cos_angle_o**2))

    # Snell's law difference
    diff = n_outside * sin_angle_c - n_conic * sin_angle_o

    return diff, intersection


def find_refraction_conic(
    camera_pos: Position3D,
    object_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
    n_outside: float,
    n_conic: float,
) -> Optional[Point3D]:
    """Find refraction point on conic surface.

    Uses optimization to find point where object ray refracts to camera position.
    Implements Snell's law using numerical root finding on conic geometry.

    Args:
        camera_pos: Camera/observer position
        object_pos: Object position inside conic
        conic_center: Conic center position (typically corneal apex)
        radius: Radius of curvature at apex (meters)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic
        n_conic: Refractive index of conic

    Returns:
        Position on conic surface where refraction occurs, or None if not found.
    """
    # Calculate directions from conic center
    to_camera = (camera_pos - conic_center).normalize()
    to_object = (object_pos - conic_center).normalize()

    if abs(to_camera.z - to_object.z) < 1e-9:
        if to_camera.z >= 0:
            return None
        upper_bound = 1.0
    else:
        alpha_zero = -to_object.z / (to_camera.z - to_object.z)
        upper_bound = min(1.0, max(0.5, alpha_zero - 1e-9))  # Ensure minimum search interval

    f0, _ = _refraction_objective_conic(
        0, camera_pos, object_pos, conic_center, radius, conic_constant, n_outside, n_conic
    )
    f1, _ = _refraction_objective_conic(
        upper_bound, camera_pos, object_pos, conic_center, radius, conic_constant, n_outside, n_conic
    )

    if np.isinf(f0) or np.isinf(f1) or f0 * f1 > 0:
        return None

    try:
        alpha = brentq(
            lambda x: _refraction_objective_conic(
                x, camera_pos, object_pos, conic_center, radius, conic_constant, n_outside, n_conic
            )[0],
            0,
            upper_bound,
        )
        _, intersection = _refraction_objective_conic(
            alpha, camera_pos, object_pos, conic_center, radius, conic_constant, n_outside, n_conic
        )
        return intersection
    except (ValueError, RuntimeError):
        return None


def refract_ray_sphere(
    ray: Ray, sphere_center: Position3D, sphere_radius: float, n_outside: float, n_sphere: float
) -> Tuple[Optional[IntersectionResult], Optional[Ray]]:
    """Refract ray through sphere surface.

    Finds intersection point and computes refracted ray direction using Snell's law.
    Handles total internal reflection when critical angle is exceeded.

    Args:
        ray: Input ray with origin and direction
        sphere_center: Sphere center position
        sphere_radius: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Tuple of (intersection_result, refracted_ray) where intersection_result contains
        the intersection point and refracted_ray is the refracted ray.
        Returns (None, None) if no intersection or total internal reflection.
    """
    # Find point of intersection
    intersection_result, _ = intersect_ray_sphere(ray, sphere_center, sphere_radius)

    if intersection_result is None or not intersection_result.intersects:
        return None, None

    intersection_point = intersection_result.point

    # Find surface normal at point of intersection (pointing inwards)
    normal_vec = (sphere_center.to_point3d() - intersection_point).normalize()

    # Calculate angles
    incident_normalized = ray.direction.normalize()
    costh1 = incident_normalized.dot(normal_vec)
    costh2_squared = 1 - (n_outside / n_sphere) ** 2 * (1 - costh1**2)

    # Check for total internal reflection
    if costh2_squared < 0:
        return intersection_result, None

    costh2 = np.sqrt(costh2_squared)

    # Snell's law refraction formula
    n_ratio = n_outside / n_sphere
    refracted_direction = incident_normalized * n_ratio + normal_vec * (costh2 - n_ratio * costh1)

    refracted_ray = Ray(origin=intersection_point, direction=refracted_direction)
    return intersection_result, refracted_ray


def refract_ray_conic(
    ray: Ray, conic_center: Position3D, radius: float, conic_constant: float, n_outside: float, n_conic: float
) -> Tuple[Optional[IntersectionResult], Optional[Ray]]:
    """Refract ray through conic surface.

    Finds intersection point and computes refracted ray direction using Snell's law.
    Uses proper conic surface normal calculation for accurate refraction.
    Handles total internal reflection when critical angle is exceeded.

    Args:
        ray: Input ray with origin and direction
        conic_center: Conic center position (typically corneal apex)
        radius: Radius parameter (R in the formula, meters)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic (e.g., air = 1.0)
        n_conic: Refractive index of conic (e.g., cornea = 1.376)

    Returns:
        Tuple of (intersection_result, refracted_ray) where:
        - intersection_result: Contains intersection point on conic surface
        - refracted_ray: Refracted ray
        Returns (None, None) if no intersection or total internal reflection.
    """
    # Find intersection point
    intersection_result, _ = intersect_ray_conic(ray, conic_center, radius, conic_constant)

    if intersection_result is None or not intersection_result.intersects:
        return None, None

    intersection_point = intersection_result.point

    # Calculate surface normal at intersection point
    surface_normal = conic_surface_normal(intersection_point, conic_center, radius, conic_constant)

    # For refraction, we need inward-pointing normal (toward conic interior)
    center_to_point = intersection_point - conic_center.to_point3d()
    if surface_normal.dot(center_to_point) > 0:  # Normal points outward
        surface_normal = surface_normal * -1  # Flip to point inward

    # Apply Snell's law
    incident_normalized = ray.direction.normalize()
    costh1 = incident_normalized.dot(surface_normal)
    costh2_squared = 1 - (n_outside / n_conic) ** 2 * (1 - costh1**2)

    # Check for total internal reflection
    if costh2_squared < 0:
        return intersection_result, None

    costh2 = np.sqrt(costh2_squared)

    # Snell's law refraction formula
    n_ratio = n_outside / n_conic
    refracted_direction = incident_normalized * n_ratio + surface_normal * (costh2 - n_ratio * costh1)

    refracted_ray = Ray(origin=intersection_point, direction=refracted_direction)
    return intersection_result, refracted_ray
