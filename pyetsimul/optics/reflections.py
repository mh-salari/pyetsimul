"""Light reflection calculation utilities for eye tracking simulation.

Implements geometric and optimization-based methods for finding glint positions on spherical and conic surfaces.
"""

import numpy as np
import warnings
from typing import Optional, Tuple, TYPE_CHECKING
from scipy.optimize import brentq
from ..types import Point3D, Vector3D, Ray, IntersectionResult, Position3D, Direction3D
from ..geometry.intersections import (
    intersect_ray_circle,
    intersect_ray_sphere,
    intersect_ray_conic,
    conic_surface_normal,
)

from ..geometry.intersections import point_on_conic_surface

if TYPE_CHECKING:
    from ..core.light import Light
    from ..core.camera import Camera


def _reflection_objective_sphere(
    a: float, light_pos: Position3D, camera_pos: Position3D, sphere_center: Position3D, sphere_radius: float
) -> Tuple[float, Point3D]:
    """Objective function for reflection finding on sphere.

    Uses interpolation between light and camera directions to find reflection point.
    Returns angle difference between incident and reflected rays for optimization.

    Args:
        a: Interpolation parameter between light and camera directions
        light_pos: Light source position
        camera_pos: Camera position
        sphere_center: Sphere center position
        sphere_radius: Sphere radius

    Returns:
        Tuple of (angle_diff, glint_pos) where angle_diff is the reflection angle error
        and glint_pos is the potential glint position
    """
    # Suppress numpy warnings to provide cleaner reflection error messages
    with np.errstate(invalid="ignore", divide="ignore"):
        # Calculate directions from sphere center to light and camera
        to_camera = (camera_pos - sphere_center).normalize()
        to_light = (light_pos - sphere_center).normalize()

        # Interpolate between directions and normalize
        n_vec = (to_camera * a + to_light * (1 - a)).normalize()

        # Calculate glint position on sphere surface
        glint_pos = sphere_center.to_point3d() + (n_vec * sphere_radius)

        # Calculate angle differences for reflection law validation
        camera_to_glint = (camera_pos.to_point3d() - glint_pos).normalize()
        light_to_glint = (light_pos.to_point3d() - glint_pos).normalize()

        angle_c = np.arccos(np.clip(n_vec.dot(camera_to_glint), -1, 1))
        angle_l = np.arccos(np.clip(n_vec.dot(light_to_glint), -1, 1))
        angle_diff = angle_c - angle_l
        return angle_diff, glint_pos


def find_reflection_sphere(
    light_pos: Position3D, camera_pos: Position3D, sphere_center: Position3D, sphere_radius: float
) -> Optional[Point3D]:
    """Find reflection point on sphere surface.

    Uses optimization to find point where light ray reflects to camera position.
    Implements reflection law using numerical root finding.

    Args:
        light_pos: Light source position
        camera_pos: Camera position
        sphere_center: Sphere center position
        sphere_radius: Sphere radius

    Returns:
        Position of glint on sphere surface, or None if no reflection found
    """
    try:
        a = brentq(
            lambda a: _reflection_objective_sphere(a, light_pos, camera_pos, sphere_center, sphere_radius)[0], 0, 1
        )
        _, glint_pos = _reflection_objective_sphere(a, light_pos, camera_pos, sphere_center, sphere_radius)
        return glint_pos
    except ValueError:
        warnings.warn(
            f"No glint found due to degenerate geometry: Light={light_pos}, Camera={camera_pos}, Sphere center={sphere_center}",
            RuntimeWarning,
        )
        return None


def _reflection_objective_conic(
    alpha: float,
    light_pos: Position3D,
    camera_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
) -> Tuple[float, Optional[Point3D]]:
    """Objective function for reflection finding on conic surface.

    Uses interpolation between light and camera directions to find reflection point.
    Projects to conic surface using proper conic section geometry.
    Returns angle difference between incident and reflected rays for optimization.

    Args:
        alpha: Interpolation parameter between light and camera directions
        light_pos: Light source position
        camera_pos: Camera position
        conic_center: Conic center position
        radius: Radius parameter of the conic (R)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Tuple of (angle_diff, glint_pos) where angle_diff is the reflection angle error
        and glint_pos is the potential glint position. Returns (inf, None) if no valid point.
    """
    # Suppress numpy warnings to provide cleaner reflection error messages
    with np.errstate(invalid="ignore", divide="ignore"):
        # Calculate directions from conic center to light and camera
        to_camera = (camera_pos - conic_center).normalize()
        to_light = (light_pos - conic_center).normalize()

        # Interpolate between directions and normalize
        n_vec = (to_camera * alpha + to_light * (1 - alpha)).normalize()

        # Find point on conic surface using proper conic geometry
        glint_pos = point_on_conic_surface(conic_center, n_vec, radius, conic_constant)
        if glint_pos is None:
            return float("inf"), None

        # Calculate surface normal at reflection point
        surface_normal = conic_surface_normal(glint_pos, conic_center, radius, conic_constant)

        # Calculate angle differences for reflection law validation
        camera_to_glint = (camera_pos.to_point3d() - glint_pos).normalize()
        light_to_glint = (light_pos.to_point3d() - glint_pos).normalize()

        angle_c = np.arccos(np.clip(surface_normal.dot(camera_to_glint), -1, 1))
        angle_l = np.arccos(np.clip(surface_normal.dot(light_to_glint), -1, 1))
        angle_diff = angle_c - angle_l

        return angle_diff, glint_pos


def find_reflection_conic(
    light_pos: Position3D, camera_pos: Position3D, conic_center: Position3D, radius: float, conic_constant: float
) -> Optional[Point3D]:
    """Find reflection point on conic surface.

    Uses optimization to find point where light ray reflects to camera position.
    Implements reflection law using numerical root finding on conic geometry.

    Args:
        light_pos: Light source position
        camera_pos: Camera position
        conic_center: Conic center position (typically corneal apex)
        radius: Radius of curvature at apex (meters)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Position of glint on conic surface, or None if no reflection found
    """
    try:
        alpha = brentq(
            lambda alpha: _reflection_objective_conic(
                alpha, light_pos, camera_pos, conic_center, radius, conic_constant
            )[0],
            0,
            1,
        )
        _, glint_pos = _reflection_objective_conic(alpha, light_pos, camera_pos, conic_center, radius, conic_constant)
        return glint_pos
    except (ValueError, TypeError):
        warnings.warn(
            f"No glint found on conic surface: Light={light_pos}, Camera={camera_pos}, Conic center={conic_center}",
            RuntimeWarning,
        )
        return None


def reflect_ray_circle(
    ray: Ray, circle_center: Point3D, circle_radius: float
) -> Tuple[Optional[IntersectionResult], Optional[Ray]]:
    """Reflect ray off circle surface.

    Finds intersection point and computes reflected ray direction using reflection law.
    Uses 2D circle geometry in x,y plane for surface normal calculation.

    Args:
        ray: Input ray with origin and direction
        circle_center: Circle center (2D using x,y components)
        circle_radius: Circle radius

    Returns:
        Tuple of (intersection_result, reflected_ray) where intersection_result contains
        the intersection point and reflected_ray is the reflected ray.
        Returns (None, None) if no intersection.
    """
    # Find intersection point
    intersection_result = intersect_ray_circle(ray, circle_center, circle_radius)

    if intersection_result is None or not intersection_result.intersects:
        return None, None

    intersection_point = intersection_result.point

    # Calculate surface normal at intersection (2D normal in x,y plane)
    normal_vec = Vector3D(
        intersection_point.x - circle_center.x,
        intersection_point.y - circle_center.y,
        0,  # Circle is in x,y plane
    ).normalize()

    # Apply reflection formula: reflected = incident - 2*normal*(incident·normal)
    incident_normalized = ray.direction.normalize()
    dot_product = incident_normalized.dot(normal_vec)
    reflected_direction = incident_normalized - normal_vec * (2 * dot_product)

    reflected_ray = Ray(origin=intersection_point, direction=reflected_direction)

    return intersection_result, reflected_ray


def reflect_ray_sphere(
    ray: Ray, sphere_center: Position3D, sphere_radius: float
) -> Tuple[Optional[IntersectionResult], Optional[Ray]]:
    """Reflect ray off sphere surface.

    Finds intersection point and computes reflected ray direction using reflection law.
    Uses outward-pointing surface normal for proper reflection calculation.

    Args:
        ray: Input ray with origin and direction
        sphere_center: Sphere center position
        sphere_radius: Sphere radius

    Returns:
        Tuple of (intersection_result, reflected_ray) where intersection_result contains
        the intersection point and reflected_ray is the reflected ray.
        Returns (None, None) if no intersection.
    """
    # Find intersection with sphere
    intersection_result, _ = intersect_ray_sphere(ray, sphere_center, sphere_radius)

    if intersection_result is None or not intersection_result.intersects:
        return None, None

    intersection_point = intersection_result.point

    # Calculate surface normal at intersection (outward pointing)
    normal_vec = (intersection_point - sphere_center.to_point3d()).normalize()

    # Apply reflection formula: reflected = incident - 2*normal*(incident·normal)
    incident_normalized = ray.direction.normalize()
    dot_product = incident_normalized.dot(normal_vec)
    reflected_direction = incident_normalized - normal_vec * (2 * dot_product)

    reflected_ray = Ray(origin=intersection_point, direction=reflected_direction)

    return intersection_result, reflected_ray


def reflect_ray_conic(
    ray: Ray, conic_center: Position3D, radius: float, conic_constant: float
) -> Tuple[Optional[IntersectionResult], Optional[Ray]]:
    """Reflect ray off conic surface.

    Finds intersection point and computes reflected ray direction using reflection law.
    Uses proper conic surface normal calculation for accurate reflection.

    Args:
        ray: Input ray with origin and direction
        conic_center: Conic center position (typically corneal apex)
        radius: Radius parameter (R in the formula, meters)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Tuple of (intersection_result, reflected_ray) where:
        - intersection_result: Contains intersection point on conic surface
        - reflected_ray: Reflected ray
        Returns (None, None) if no intersection.
    """
    # Find intersection point
    intersection_result, _ = intersect_ray_conic(ray, conic_center, radius, conic_constant)

    if intersection_result is None or not intersection_result.intersects:
        return None, None

    intersection_point = intersection_result.point

    # Calculate surface normal at intersection point
    surface_normal = conic_surface_normal(intersection_point, conic_center, radius, conic_constant)

    # For reflection, we typically want outward-pointing normal
    center_to_point = intersection_point - conic_center.to_point3d()
    if surface_normal.dot(center_to_point) < 0:  # Normal points inward
        surface_normal = surface_normal * -1  # Flip to point outward

    # Apply reflection formula: reflected = incident - 2*normal*(incident·normal)
    incident_normalized = ray.direction.normalize()
    dot_product = incident_normalized.dot(surface_normal)
    reflected_direction = incident_normalized - surface_normal * (2 * dot_product)

    reflected_ray = Ray(origin=intersection_point, direction=reflected_direction)

    return intersection_result, reflected_ray


def find_corneal_reflection(eye, light: "Light", camera: "Camera") -> Optional[Position3D]:
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
        # Check if point is on visible cornea (within boundaries and not occluded by eyelid)
        if not eye.point_on_visible_cornea(cr):
            cr = None

    return cr


def find_corneal_reflection_simple(eye, light: "Light", camera: "Camera") -> Optional[Position3D]:
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

    # Check if point is on visible cornea (within boundaries and not occluded by eyelid)
    if not eye.point_on_visible_cornea(cr):
        return None

    return cr
