"""Glint size computation using convex mirror optics.

Computes the diameter of a corneal reflection (glint) produced by an extended light source (LED). The cornea
acts as a convex mirror and forms a small virtual image of the LED just behind its surface.

For a light at distance u from the reflection point and a local radius of curvature R of the cornea there,
the mirror equation of a convex mirror gives the image

    diameter = led_diameter * R / (2 * u + R),     depth behind the surface = u * R / (2 * u + R)

A camera at distance c from the reflection point looks at this image through the surface, where it spans

    glint_diameter = diameter * c / (c + depth)

The formula is paraxial and gives the mean diameter of the glint, which is slightly elliptical. For the light
and camera positions of a remote eye tracker it is within about 0.3 % of the mean of the two glint diameters
that the reflection solver gives for the ends of two perpendicular diameters of the light.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..core.cornea import Cornea
    from ..types import Position3D


def _mean_local_radius(axis_distance_squared: float, apex_radius: float, conic_constant: float) -> float:
    """Mean of the two principal radii of curvature of a conicoid at a point off its axis.

    For the conicoid h^2 = 2 R z - (1 + k) z^2, at distance h from the axis the sagittal radius is
    sqrt(R^2 - k h^2) and the tangential radius is its cube over R^2. A sphere (k = 0) gives R everywhere.

    Args:
        axis_distance_squared: Squared distance h^2 of the point from the axis of the conicoid (mm^2).
        apex_radius: Radius of curvature R at the apex (mm).
        conic_constant: Conic constant k (0 sphere, k < 0 prolate ellipsoid).

    Returns:
        Mean of the sagittal and the tangential radius at the point (mm).

    """
    sagittal = math.sqrt(apex_radius**2 - conic_constant * axis_distance_squared)
    tangential = sagittal**3 / apex_radius**2
    return (sagittal + tangential) / 2.0


def compute_glint_diameter(
    light_pos: Position3D,
    cr_pos: Position3D,
    camera_pos: Position3D,
    cornea: Cornea,
    eye_transform: np.ndarray,
    led_diameter: float,
) -> float:
    """Diameter of the glint of an extended light, as a camera sees it on the corneal surface.

    The local radius of curvature is that of the anterior surface at the reflection point: the radius of a
    spherical cornea, and the mean principal radius of a conic one. A toric cornea is taken as the conic it
    is built on, the mean of its two meridians.

    Args:
        light_pos: 3D position of the light source.
        cr_pos: 3D position of the corneal reflection on the surface.
        camera_pos: 3D position of the camera.
        cornea: Cornea of the eye.
        eye_transform: 4x4 eye transformation matrix (eye-local to world).
        led_diameter: Physical diameter of the LED light source (mm).

    Returns:
        Glint diameter at the corneal surface (mm).

    """
    # Distance of the reflection point from the corneal axis, which is the eye-local z axis through the
    # cornea centre.
    local = np.linalg.solve(np.asarray(eye_transform, dtype=float), np.array([cr_pos.x, cr_pos.y, cr_pos.z, 1.0]))
    axis_distance_squared = (local[0] - cornea.center.x) ** 2 + (local[1] - cornea.center.y) ** 2
    conic_constant = 0.0 if cornea.cornea_type == "spherical" else cornea.anterior_k
    radius = _mean_local_radius(axis_distance_squared, cornea.anterior_radius, conic_constant)

    light_distance = light_pos.distance_to(cr_pos)
    camera_distance = camera_pos.distance_to(cr_pos)
    image_diameter = led_diameter * radius / (2.0 * light_distance + radius)
    image_depth = light_distance * radius / (2.0 * light_distance + radius)
    return image_diameter * camera_distance / (camera_distance + image_depth)
