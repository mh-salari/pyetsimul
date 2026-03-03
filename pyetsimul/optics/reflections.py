"""Light reflection calculation utilities for eye tracking simulation.

Implements geometric and optimization-based methods for finding glint positions on spherical and conic surfaces.
"""

import warnings
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.optimize import brentq, fsolve

from ..geometry.intersections import (
    conic_surface_normal,
    intersect_ray_circle,
    intersect_ray_conic,
    intersect_ray_sphere,
    point_on_conic_surface,
)
from ..types import Direction3D, IntersectionResult, Point3D, Position3D, Ray

if TYPE_CHECKING:
    from ..core.camera import Camera
    from ..core.eye import Eye
    from ..core.light import Light


def _reflection_objective_sphere(
    a: float, light_pos: Position3D, camera_pos: Position3D, sphere_center: Position3D, sphere_radius: float
) -> tuple[float, Point3D]:
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
) -> Point3D | None:
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
        alpha = brentq(
            lambda a: _reflection_objective_sphere(a, light_pos, camera_pos, sphere_center, sphere_radius)[0], 0, 1
        )
        _, glint_pos = _reflection_objective_sphere(
            cast("float", alpha), light_pos, camera_pos, sphere_center, sphere_radius
        )
        return glint_pos
    except ValueError:
        warnings.warn(
            f"No glint found due to degenerate geometry: Light={light_pos}, Camera={camera_pos}, Sphere center={sphere_center}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def _reflection_residuals_conic(
    params: np.ndarray,
    to_camera: np.ndarray,
    to_light: np.ndarray,
    perp: np.ndarray,
    light_pos: Position3D,
    camera_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
) -> np.ndarray:
    """Residual function for 2D reflection finding on conic surface.

    Uses (alpha, beta) parameterization where alpha interpolates between light
    and camera directions, and beta adds an out-of-plane component. This 2D
    search is necessary because aspherical conic surfaces (k != 0) can have
    reflection points outside the light-center-camera plane.

    Returns a 2-element residual vector:
        [0]: Equal angles condition (N dot d_incident + N dot d_camera)
        [1]: Coplanarity condition (N dot (d_incident x d_camera))

    """
    alpha, beta = params

    # Search direction: in-plane interpolation + out-of-plane offset
    direction = to_camera * alpha + to_light * (1 - alpha) + perp * beta
    norm = np.linalg.norm(direction)
    if norm < 1e-15:
        return np.array([1e10, 1e10])
    direction /= norm

    n_vec = Direction3D(direction[0], direction[1], direction[2])
    glint_pos = point_on_conic_surface(conic_center, n_vec, radius, conic_constant)
    if glint_pos is None:
        return np.array([1e10, 1e10])

    surface_normal = conic_surface_normal(glint_pos, conic_center, radius, conic_constant)
    n = np.array([surface_normal.x, surface_normal.y, surface_normal.z])

    # Incident direction: light -> glint (pointing inward)
    d_in = np.array([glint_pos.x - light_pos.x, glint_pos.y - light_pos.y, glint_pos.z - light_pos.z])
    d_in /= np.linalg.norm(d_in)

    # Camera direction: glint -> camera (pointing outward)
    d_cam = np.array([camera_pos.x - glint_pos.x, camera_pos.y - glint_pos.y, camera_pos.z - glint_pos.z])
    d_cam /= np.linalg.norm(d_cam)

    # Condition 1: equal angles — N·d_in = -(N·d_cam)
    residual_angle = np.dot(n, d_in) + np.dot(n, d_cam)

    # Condition 2: coplanarity — N·(d_in x d_cam) = 0
    residual_coplanar = np.dot(n, np.cross(d_in, d_cam))

    return np.array([residual_angle, residual_coplanar])


def find_reflection_conic(
    light_pos: Position3D, camera_pos: Position3D, conic_center: Position3D, radius: float, conic_constant: float
) -> Point3D | None:
    """Find reflection point on conic surface.

    Uses 2D root-finding to find the point where light ray reflects to camera.
    The search is parameterized by (alpha, beta) where alpha interpolates between
    light and camera directions and beta adds an out-of-plane component, necessary
    for aspherical surfaces where the reflection point may not lie in the
    light-center-camera plane.

    Args:
        light_pos: Light source position
        camera_pos: Camera position
        conic_center: Conic center position (typically corneal apex)
        radius: Radius of curvature at apex (mm)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Position of glint on conic surface, or None if no reflection found

    """
    try:
        to_camera_dir = (camera_pos - conic_center).normalize()
        to_light_dir = (light_pos - conic_center).normalize()
        to_camera = np.array([to_camera_dir.x, to_camera_dir.y, to_camera_dir.z])
        to_light = np.array([to_light_dir.x, to_light_dir.y, to_light_dir.z])

        # Perpendicular to the light-center-camera plane
        perp = np.cross(to_camera, to_light)
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-15:
            # Light and camera are collinear from center — degenerate
            return None
        perp /= perp_norm

        solution, _info, ier, _msg = fsolve(
            _reflection_residuals_conic,
            x0=np.array([0.5, 0.0]),
            args=(to_camera, to_light, perp, light_pos, camera_pos, conic_center, radius, conic_constant),
            full_output=True,
        )

        if ier != 1:
            return None

        # Reconstruct the glint position from the solution
        alpha, beta = solution
        direction = to_camera * alpha + to_light * (1 - alpha) + perp * beta
        direction /= np.linalg.norm(direction)
        n_vec = Direction3D(direction[0], direction[1], direction[2])
        return point_on_conic_surface(conic_center, n_vec, radius, conic_constant)

    except (ValueError, TypeError):
        warnings.warn(
            f"No glint found on conic surface: Light={light_pos}, Camera={camera_pos}, Conic center={conic_center}",
            RuntimeWarning,
            stacklevel=2,
        )
        return None


def reflect_ray_circle(
    ray: Ray, circle_center: Point3D, circle_radius: float
) -> tuple[IntersectionResult | None, Ray | None]:
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

    intersection_point = cast("Point3D", intersection_result.point)

    # Calculate surface normal at intersection (2D normal in x,y plane)
    normal_vec = Direction3D(
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
) -> tuple[IntersectionResult | None, Ray | None]:
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

    intersection_point = cast("Point3D", intersection_result.point)

    # Calculate surface normal at intersection (outward pointing)
    normal_vec = (intersection_point - sphere_center.to_point3d()).to_direction3d().normalize()

    # Apply reflection formula: reflected = incident - 2*normal*(incident·normal)
    incident_normalized = ray.direction.normalize()
    dot_product = incident_normalized.dot(normal_vec)
    reflected_direction = incident_normalized - normal_vec * (2 * dot_product)

    reflected_ray = Ray(origin=intersection_point, direction=reflected_direction)

    return intersection_result, reflected_ray


def reflect_ray_conic(
    ray: Ray, conic_center: Position3D, radius: float, conic_constant: float
) -> tuple[IntersectionResult | None, Ray | None]:
    """Reflect ray off conic surface.

    Finds intersection point and computes reflected ray direction using reflection law.
    Uses proper conic surface normal calculation for accurate reflection.

    Args:
        ray: Input ray with origin and direction
        conic_center: Conic center position (typically corneal apex)
        radius: Radius parameter (R in the formula, mm)
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

    intersection_point = cast("Point3D", intersection_result.point)

    # Calculate surface normal at intersection point
    surface_normal = conic_surface_normal(intersection_point, conic_center, radius, conic_constant)

    # For reflection, we typically want outward-pointing normal
    center_to_point = intersection_point - conic_center.to_point3d()
    if surface_normal.dot(center_to_point) < 0:  # Normal points inward
        surface_normal *= -1  # Flip to point outward

    # Apply reflection formula: reflected = incident - 2*normal*(incident·normal)
    incident_normalized = ray.direction.normalize()
    dot_product = incident_normalized.dot(surface_normal)
    reflected_direction = incident_normalized - surface_normal * (2 * dot_product)

    reflected_ray = Ray(origin=intersection_point, direction=reflected_direction)

    return intersection_result, reflected_ray


def find_corneal_reflection(eye: "Eye", light: "Light", camera: "Camera") -> Position3D | None:
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


def find_corneal_reflection_simple(eye: "Eye", light: "Light", camera: "Camera") -> Position3D | None:
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
    to_cam = (camera_pos - cc).to_direction3d().normalize()

    # Paraxial approximation calculation
    light_to_cornea = (light.position - cc).to_direction3d()
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
