"""Light reflection calculation utilities for eye tracking simulation.

Implements geometric and optimization-based methods for finding glint positions on spherical and conic surfaces.
"""

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
from ..log import debug
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
        debug(
            f"No glint found due to degenerate geometry: Light={light_pos}, Camera={camera_pos}, "
            f"Sphere center={sphere_center}"
        )
        return None


# Conic reflection solve.
_REFLECTION_SOLVER_XTOL = 1e-13  # relative step at which the root finder stops
# Largest accepted length of the difference between the unit direction of the reflected ray and the unit
# direction from the glint to the camera.
_REFLECTION_TOL = 1e-10
_NO_SURFACE_RESIDUAL = 1e10  # residual of a search direction that does not meet the surface
_OPPOSITE_DIRECTIONS_TOL = 1e-12  # length of the bisector below which the light and the camera are opposite


def _reflection_mismatch_conic(
    params: np.ndarray,
    bisector: np.ndarray,
    across: tuple[np.ndarray, np.ndarray],
    light_pos: Position3D,
    camera_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
    lateral: tuple[float, float, float],
) -> tuple[np.ndarray, Point3D] | None:
    """Reflection-law mismatch at the surface point that the search parameters name, and that point.

    The point lies in the direction ``bisector + a * across[0] + b * across[1]`` from the conic centre. The
    mismatch is the unit direction of the ray reflected there minus the unit direction from the point to the
    camera; it is the zero vector at the glint. None if the direction does not meet the surface.
    """
    direction = bisector + params[0] * across[0] + params[1] * across[1]
    n_vec = Direction3D(direction[0], direction[1], direction[2])
    glint_pos = point_on_conic_surface(conic_center, n_vec, radius, conic_constant, *lateral)
    if glint_pos is None:
        return None

    surface_normal = conic_surface_normal(glint_pos, conic_center, radius, conic_constant, *lateral)
    n = np.array([surface_normal.x, surface_normal.y, surface_normal.z])

    # Incident direction: light -> glint (pointing inward)
    d_in = np.array([glint_pos.x - light_pos.x, glint_pos.y - light_pos.y, glint_pos.z - light_pos.z])
    d_in /= np.linalg.norm(d_in)

    # Camera direction: glint -> camera (pointing outward)
    d_cam = np.array([camera_pos.x - glint_pos.x, camera_pos.y - glint_pos.y, camera_pos.z - glint_pos.z])
    d_cam /= np.linalg.norm(d_cam)

    reflected = d_in - 2.0 * (n @ d_in) * n
    return reflected - d_cam, glint_pos


def _reflection_residuals_conic(
    params: np.ndarray, bisector: np.ndarray, across: tuple[np.ndarray, np.ndarray], *surface: object
) -> np.ndarray:
    """Residual function for 2D reflection finding on conic surface.

    The two components of the reflection-law mismatch along the two search directions. Both are zero where
    the ray from the light reflects to the camera, also for a light at the camera position, where the ray
    returns along the surface normal. ``surface`` holds the remaining arguments of
    ``_reflection_mismatch_conic``.
    """
    found = _reflection_mismatch_conic(params, bisector, across, *surface)
    if found is None:
        return np.array([_NO_SURFACE_RESIDUAL, _NO_SURFACE_RESIDUAL])
    mismatch, _ = found
    return np.array([mismatch @ across[0], mismatch @ across[1]])


def find_reflection_conic(
    light_pos: Position3D,
    camera_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
    axx: float = 1.0,
    ayy: float = 1.0,
    axy: float = 0.0,
) -> Point3D | None:
    """Find reflection point on conic surface.

    Uses 2D root-finding to find the point where light ray reflects to camera. The search starts in the
    direction that bisects the directions from the conic centre to the light and to the camera, and moves
    along two directions perpendicular to it. Two are needed because on an aspherical surface the reflection
    point does not have to lie in the light-center-camera plane.

    Args:
        light_pos: Light source position
        camera_pos: Camera position
        conic_center: Conic center position (typically corneal apex)
        radius: Radius of curvature at apex (mm)
        conic_constant: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        axx: Lateral x^2 coefficient. Default 1.0; with ayy and axy other than (1, 1, 0) the surface
            becomes toric, its two meridian apex radii R/axx and R/ayy with axy setting the steep axis.
        ayy: Lateral y^2 coefficient. Default 1.0.
        axy: Lateral cross-term coefficient. Default 0.0.

    Returns:
        Position of glint on conic surface, or None if no reflection found

    """
    try:
        to_camera_dir = (camera_pos - conic_center).normalize()
        to_light_dir = (light_pos - conic_center).normalize()
        to_camera = np.array([to_camera_dir.x, to_camera_dir.y, to_camera_dir.z])
        to_light = np.array([to_light_dir.x, to_light_dir.y, to_light_dir.z])

        bisector = to_camera + to_light
        bisector_norm = np.linalg.norm(bisector)
        if bisector_norm < _OPPOSITE_DIRECTIONS_TOL:
            debug(
                f"No glint found: the light at {light_pos} and the camera at {camera_pos} are on opposite sides "
                "of the conic"
            )
            return None
        bisector /= bisector_norm

        # Two unit directions perpendicular to the bisector and to each other, built from the coordinate
        # axis that is least aligned with it.
        axis = np.eye(3)[np.argmin(np.abs(bisector))]
        first = axis - (axis @ bisector) * bisector
        first /= np.linalg.norm(first)
        across = (first, np.cross(bisector, first))

        search = (bisector, across, light_pos, camera_pos, conic_center, radius, conic_constant, (axx, ayy, axy))
        # With full_output the root finder reports how it stopped as a value and not as a warning. The solution
        # is judged by its mismatch: at this step tolerance the finder also stops where rounding leaves it no
        # further progress, which is a solution, and on directions that miss the surface, which is not.
        solution, *_ = fsolve(
            _reflection_residuals_conic, x0=np.zeros(2), args=search, xtol=_REFLECTION_SOLVER_XTOL, full_output=True
        )
        found = _reflection_mismatch_conic(solution, *search)
        if found is None or np.linalg.norm(found[0]) > _REFLECTION_TOL:
            debug(f"No glint found: the reflection solver did not converge. Light={light_pos}, Camera={camera_pos}")
            return None
        return found[1]

    except (ValueError, TypeError):
        debug(f"No glint found on conic surface: Light={light_pos}, Camera={camera_pos}, Conic center={conic_center}")
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
            debug(f"No glint found: the reflection of the light at {light.position} is off the visible cornea")
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
