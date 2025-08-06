"""
Eye optical calculations extracted from the Eye class.

This module contains optical calculation functions that were previously
part of the Eye class, extracted for better modularity and testability.
"""

import numpy as np
from typing import Optional, Tuple
from ..types import Position3D, Direction3D, Vector3D, Ray
from .refractions import refract_ray_sphere
from ..core.light import Light
from ..core.camera import Camera


def find_corneal_reflection(eye, light: Light, camera: Camera) -> Optional[Position3D]:
    """Finds the position of a corneal reflex.

    Determines the point on corneal surface where light ray reflects to camera.
    Uses exact reflection calculation with corneal surface geometry.

    Args:
        eye: Eye object
        light: Light source object
        camera: Camera object

    Returns:
        Position3D of corneal reflex, or None if not within cornea
    """
    # Find reflection point on corneal surface
    camera_position = Position3D.from_array(camera.trans[:, 3])
    cr_point3d = eye.cornea.find_reflection(light.position, camera_position, eye.trans)

    cr = None
    if cr_point3d is not None:
        cr = Position3D.from_point3d(cr_point3d)
        # Check if point is within corneal boundaries
        if not eye.point_within_cornea(cr):
            cr = None

    return cr


def find_corneal_reflection_simple(eye, light: Light, camera: Camera) -> Optional[Position3D]:
    """Finds the position of a corneal reflex (simplified).

    Uses paraxial approximation for faster corneal reflex calculation.
    Based on Morimoto, Amir and Flicker approximation method.

    Args:
        eye: Eye object
        light: Light source object
        camera: Camera object

    Returns:
        Position3D of corneal reflex, or None if not found
    """
    # Get corneal center in world coordinates
    cc_homogeneous = eye.trans @ np.array(eye.cornea.center)
    cc = Position3D.from_array(cc_homogeneous)

    # Vector from corneal center to camera
    camera_pos = Position3D.from_array(camera.trans[:, 3])
    to_cam = Direction3D.from_vector3d(camera_pos - cc).normalize()

    # Paraxial approximation calculation
    light_to_cornea = Direction3D.from_vector3d(light.position - cc)
    denominator = 2 * light_to_cornea.dot(to_cam)

    if abs(denominator) < 1e-10:  # Avoid division by zero
        return None

    w = eye.cornea.anterior_radius / denominator

    # Calculate corneal reflex position
    cr = cc + (light_to_cornea.to_vector3d() * w)

    # Check if point is within corneal boundaries
    if not eye.point_within_cornea(cr):
        return None

    return cr



def refract_ray_dual_surface(
    eye, ray_origin: Position3D, ray_direction: Vector3D
) -> Tuple[Optional[Position3D], Optional[Position3D], Optional[Vector3D]]:
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


def find_refraction_point(eye, camera_position: Position3D, object_position: Position3D) -> Optional[Position3D]:
    """Computes observed position of intraocular objects.

    Calculates where camera observes intraocular object through corneal refraction.
    Determines corneal surface point where object ray refracts to camera.

    Args:
        eye: Eye object
        camera_position: Camera position (Position3D)
        object_position: Object position inside eye (Position3D)

    Returns:
        Position3D on corneal surface where refraction occurs, or None
    """
    # Find refraction point on corneal surface
    refraction_point = eye.cornea.find_refraction(
        camera_position,
        object_position,
        1.0,  # Air refractive index
        eye.cornea.refractive_index,
        eye.trans,
    )

    if refraction_point is None:
        return None

    # Check if point is within corneal boundaries
    if not eye.point_within_cornea(refraction_point.to_position3d()):
        refraction_point = None

    return refraction_point
