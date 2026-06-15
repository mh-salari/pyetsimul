"""Light refraction utilities for eye tracking simulation.

Implements Snell's law and ray-surface refraction for spherical and conic surfaces.
"""

from typing import TYPE_CHECKING, cast

import numpy as np

from ..geometry.intersections import (
    conic_surface_normal,
    intersect_ray_conic,
    intersect_ray_sphere,
)
from ..types import Direction3D, IntersectionResult, Point3D, Position3D, Ray, TransformationMatrix

if TYPE_CHECKING:
    from ..core.cornea import Cornea
    from ..core.eye import Eye


def refract_direction(
    incident: Direction3D, surface_normal: Direction3D, n_from: float, n_to: float
) -> Direction3D | None:
    """Refract a unit direction at a surface using Snell's law in vector form.

    Direction-agnostic: the surface normal is oriented to face the incident medium before the law is
    applied, so the same routine is correct whether the ray enters or leaves the surface. The two-surface
    exit path needs the leaving case, which the single-surface refract_ray_* helpers (inward-only normal)
    do not handle.

    Args:
        incident: Incident unit direction.
        surface_normal: Surface unit normal (either orientation accepted).
        n_from: Refractive index of the medium the ray is leaving.
        n_to: Refractive index of the medium the ray is entering.

    Returns:
        Refracted unit direction, or None on total internal reflection.

    """
    i = np.array([incident.x, incident.y, incident.z])
    n = np.array([surface_normal.x, surface_normal.y, surface_normal.z])
    cos_incidence = -float(np.dot(i, n))
    if cos_incidence < 0.0:  # normal points along the ray; flip it to face the incident medium
        n = -n
        cos_incidence = -cos_incidence
    eta = n_from / n_to
    cos_transmission_squared = 1.0 - eta * eta * (1.0 - cos_incidence * cos_incidence)
    if cos_transmission_squared < 0.0:
        return None  # total internal reflection
    refracted = eta * i + (eta * cos_incidence - np.sqrt(cos_transmission_squared)) * n
    return Direction3D(refracted[0], refracted[1], refracted[2]).normalize()


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
    cornea: "Cornea",
    eye_transform: TransformationMatrix,
    camera_position: Position3D,
    object_position: Position3D,
    n_aqueous: float,
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
        n_aqueous: Refractive index of the aqueous humor (used only by the two-surface corneal path)

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
        n_aqueous,
    )

    return None if refraction_point is None else refraction_point.to_position3d()
