"""Glint size computation using convex mirror optics.

Computes the physical diameter of a corneal reflection (glint) produced by an
extended light source (LED). The cornea acts as a convex mirror, forming a
virtual image of the LED on its surface.

Convex mirror formula:
    glint_diameter = led_diameter * R / (2 * u - R)

where R is the local radius of curvature at the reflection point and u is the
distance from the light source to the reflection point.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

from ..types import Position3D

if TYPE_CHECKING:
    from ..core.cornea import Cornea


def compute_glint_diameter_sphere(
    light_pos: Position3D,
    cr_pos: Position3D,
    anterior_radius: float,
    led_diameter: float,
) -> float:
    """Compute glint diameter for a spherical cornea.

    Args:
        light_pos: 3D position of the light source.
        cr_pos: 3D position of the corneal reflection on the surface.
        anterior_radius: Radius of curvature of the anterior corneal surface (mm).
        led_diameter: Physical diameter of the LED light source (mm).

    Returns:
        Glint diameter on the corneal surface (mm).

    """
    u = light_pos.distance_to(cr_pos)
    radius = anterior_radius
    return led_diameter * radius / (2 * u - radius)


def _local_sagittal_radius(
    cr_pos: Position3D,
    cornea_center: Position3D,
    anterior_radius: float,
    anterior_k: float,
) -> float:
    """Compute local sagittal radius of curvature at a point on a conic surface.

    For a conic of revolution: R_s = sqrt(R_apex^2 - (1+k) * r^2)
    where r is the radial distance from the optical axis to the reflection point.

    Args:
        cr_pos: 3D position of the reflection point (world coordinates).
        cornea_center: 3D position of the cornea center (world coordinates).
        anterior_radius: Apical radius of curvature (mm).
        anterior_k: Conic constant (k=0 sphere, k<0 prolate ellipsoid).

    Returns:
        Local sagittal radius of curvature at the reflection point (mm).

    """
    # Radial distance squared from optical axis (x and y offsets from center)
    r_squared = (cr_pos.x - cornea_center.x) ** 2 + (cr_pos.y - cornea_center.y) ** 2
    rs_squared = anterior_radius**2 - (1 + anterior_k) * r_squared
    return math.sqrt(rs_squared)


def compute_glint_diameter_conic(
    light_pos: Position3D,
    cr_pos: Position3D,
    cornea_center: Position3D,
    anterior_radius: float,
    anterior_k: float,
    led_diameter: float,
) -> float:
    """Compute glint diameter for a conic cornea using local curvature.

    Args:
        light_pos: 3D position of the light source.
        cr_pos: 3D position of the corneal reflection on the surface.
        cornea_center: 3D position of the cornea center (world coordinates).
        anterior_radius: Apical radius of curvature (mm).
        anterior_k: Conic constant.
        led_diameter: Physical diameter of the LED light source (mm).

    Returns:
        Glint diameter on the corneal surface (mm).

    """
    radius = _local_sagittal_radius(cr_pos, cornea_center, anterior_radius, anterior_k)
    u = light_pos.distance_to(cr_pos)
    return led_diameter * radius / (2 * u - radius)


def compute_glint_diameter(
    light_pos: Position3D,
    cr_pos: Position3D,
    cornea: Cornea,
    eye_transform: np.ndarray,
    led_diameter: float,
) -> float:
    """Compute glint diameter for any cornea type.

    Dispatches to sphere or conic computation based on the cornea type.

    Args:
        light_pos: 3D position of the light source.
        cr_pos: 3D position of the corneal reflection on the surface.
        cornea: Cornea object (SphericalCornea or ConicCornea).
        eye_transform: 4x4 eye transformation matrix (for cornea center in world coords).
        led_diameter: Physical diameter of the LED light source (mm).

    Returns:
        Glint diameter on the corneal surface (mm).

    """
    if cornea.cornea_type == "spherical":
        return compute_glint_diameter_sphere(light_pos, cr_pos, cornea.anterior_radius, led_diameter)
    if cornea.cornea_type == "conic":
        world_center = eye_transform @ np.array(cornea.center)
        cornea_center = Position3D.from_array(world_center)
        return compute_glint_diameter_conic(
            light_pos,
            cr_pos,
            cornea_center,
            cornea.anterior_radius,
            cornea.anterior_k,
            led_diameter,
        )
    raise TypeError(f"Unsupported cornea type: {cornea.cornea_type}")
