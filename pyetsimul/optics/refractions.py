"""Light refraction calculation utilities for eye tracking simulation.

Implements Snell's law, ray-surface intersection, and optimization for refraction on spherical and conic surfaces.
"""

from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.optimize import brentq

from ..geometry.intersections import (
    conic_surface_normal,
    intersect_ray_conic,
    intersect_ray_sphere,
    point_on_conic_surface,
)
from ..types import Direction3D, IntersectionResult, Point3D, Position3D, Ray, TransformationMatrix, Vector3D

if TYPE_CHECKING:
    from ..core.cornea import Cornea
    from ..core.eye import Eye


def _refraction_objective_sphere(
    a: float,
    camera_pos: Position3D,
    object_pos: Position3D,
    sphere_center: Position3D,
    sphere_radius: float,
    n_outside: float,
    n_sphere: float,
) -> tuple[float, Point3D]:
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
) -> Point3D | None:
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
        alpha = brentq(
            lambda x: _refraction_objective_sphere(
                x, camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
            )[0],
            0,
            1,
        )
        _, result = _refraction_objective_sphere(
            cast("float", alpha), camera_pos, object_pos, sphere_center, sphere_radius, n_outside, n_sphere
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
) -> tuple[float, Point3D | None]:
    """Objective function for refraction finding on conic surface.

    Uses interpolation between camera and object directions to find refraction point.
    Projects to conic surface using proper conic section geometry.
    Returns Snell's law difference for optimization.

    Args:
        alpha: Interpolation parameter between camera and object directions
        camera_pos: Camera position
        object_pos: Object position
        conic_center: Conic center position (typically corneal apex)
        radius: Radius of curvature at apex (mm)
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
) -> Point3D | None:
    """Find refraction point on conic surface.

    Uses optimization to find point where object ray refracts to camera position.
    Implements Snell's law using numerical root finding on conic geometry.

    Args:
        camera_pos: Camera/observer position
        object_pos: Object position inside conic
        conic_center: Conic center position (typically corneal apex)
        radius: Radius of curvature at apex (mm)
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
            cast("float", alpha), camera_pos, object_pos, conic_center, radius, conic_constant, n_outside, n_conic
        )
        return intersection
    except (ValueError, RuntimeError):
        return None


def refract_ray_sphere(
    ray: Ray, sphere_center: Position3D, sphere_radius: float, n_outside: float, n_sphere: float
) -> tuple[IntersectionResult | None, Ray | None]:
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

    intersection_point = cast("Point3D", intersection_result.point)

    # Find surface normal at point of intersection (pointing inwards)
    normal_vec = (sphere_center.to_point3d() - intersection_point).to_direction3d().normalize()

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
) -> tuple[IntersectionResult | None, Ray | None]:
    """Refract ray through conic surface.

    Finds intersection point and computes refracted ray direction using Snell's law.
    Uses proper conic surface normal calculation for accurate refraction.
    Handles total internal reflection when critical angle is exceeded.

    Args:
        ray: Input ray with origin and direction
        conic_center: Conic center position (typically corneal apex)
        radius: Radius parameter (R in the formula, mm)
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

    intersection_point = cast("Point3D", intersection_result.point)

    # Calculate surface normal at intersection point
    surface_normal = conic_surface_normal(intersection_point, conic_center, radius, conic_constant)

    # For refraction, we need inward-pointing normal (toward conic interior)
    center_to_point = intersection_point - conic_center.to_point3d()
    if surface_normal.dot(center_to_point) > 0:  # Normal points outward
        surface_normal *= -1  # Flip to point inward

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


def refract_ray_dual_surface(
    eye: "Eye", ray_origin: Point3D, ray_direction: Direction3D
) -> tuple[Point3D | None, Point3D | None, Direction3D | None]:
    """Computes refraction through both anterior and posterior corneal surfaces.

    Models complete corneal optical path by calculating refraction at both:
    1. Anterior surface: air (n=1.0) → cornea (n=1.376)
    2. Posterior surface: cornea (n=1.376) → aqueous humor (n=1.336)

    This provides more accurate modeling of light rays passing through the cornea
    compared to single-surface refraction which only considers the anterior surface.

    Args:
        eye: Eye object containing corneal geometry and refractive indices
        ray_origin: Ray origin (Position3D)
        ray_direction: Ray direction (3D vector)

    Returns:
        Tuple of (anterior_point, posterior_point, final_direction) where:

        - anterior_point: Point where ray strikes anterior corneal surface
        - posterior_point: Point where ray strikes posterior corneal surface
        - final_direction: Direction of ray after exiting posterior surface

        Returns (None, None, None) if ray doesn't intersect with cornea.

    """
    # Get corneal center in world coordinates
    cornea_center_homogeneous = eye.trans @ np.array(eye.cornea.center)
    cornea_center = Position3D.from_array(cornea_center_homogeneous)

    # Refraction at outer surface of cornea
    ray = Ray(origin=ray_origin, direction=ray_direction)
    intersection_result, refracted_ray = refract_ray_sphere(
        ray,
        cornea_center,
        eye.cornea.anterior_radius,
        1.0,  # Air refractive index
        eye.cornea.refractive_index,
    )
    if intersection_result is None or refracted_ray is None:
        return None, None, None
    outer_point = intersection_result.point
    intermediate_direction = refracted_ray.direction

    if outer_point is None or intermediate_direction is None:
        return None, None, None

    # Refraction at inner surface of cornea
    posterior_center_homogeneous = eye.trans @ np.array(eye.cornea.get_posterior_center())
    posterior_center = Position3D.from_array(posterior_center_homogeneous)
    ray2 = Ray(origin=outer_point, direction=intermediate_direction)
    intersection_result2, refracted_ray2 = refract_ray_sphere(
        ray2,
        posterior_center,
        eye.cornea.posterior_radius,
        eye.cornea.refractive_index,
        eye.n_aqueous_humor,
    )
    if intersection_result2 is None or refracted_ray2 is None:
        return outer_point, None, None
    inner_point = intersection_result2.point
    final_direction = refracted_ray2.direction

    return outer_point, inner_point, final_direction


def find_refraction_point(
    cornea: "Cornea", eye_transform: TransformationMatrix, camera_position: Position3D, object_position: Position3D
) -> Position3D | None:
    """Computes observed position of intraocular objects through corneal refraction.

    Pure function that calculates where camera observes intraocular object through corneal refraction.
    Determines corneal surface point where object ray refracts to camera.

    Note: This function does not check corneal boundaries - that should be done by the caller
    if needed (e.g., using Eye.point_within_cornea()).

    Args:
        cornea: Cornea object with find_refraction method
        eye_transform: Eye transformation matrix
        camera_position: Camera position (Position3D)
        object_position: Object position inside eye (Position3D)

    Returns:
        Position3D on corneal surface where refraction occurs, or None if no solution exists

    """
    # Find refraction point on corneal surface using cornea's refraction method
    refraction_point = cornea.find_refraction(
        camera_position,
        object_position,
        1.0,  # Air refractive index
        cornea.refractive_index,
        eye_transform,
    )

    return None if refraction_point is None else refraction_point.to_position3d()
