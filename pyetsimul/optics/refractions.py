"""Light refraction calculation utilities for eye tracking simulation.

Implements Snell's law, ray-surface intersection, and optimization for refraction on spherical and conic surfaces.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, cast

import numpy as np
from scipy.optimize import brentq, fsolve, least_squares

from ..geometry.intersections import (
    conic_surface_normal,
    intersect_ray_conic,
    intersect_ray_sphere,
    point_on_conic_surface,
)
from ..types import Direction3D, IntersectionResult, Point3D, Position3D, Ray, TransformationMatrix

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


def _refraction_snell_conic_1d(
    alpha: float,
    to_camera: np.ndarray,
    to_object: np.ndarray,
    camera_pos: Position3D,
    object_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
    n_outside: float,
    n_conic: float,
) -> float:
    """1D Snell's law residual on conic surface for in-plane search (beta=0).

    Evaluates n_outside * sin(theta_cam) - n_conic * sin(theta_obj) at the conic
    surface point determined by alpha. Used with brentq on [0, 1] to find the
    in-plane refraction point directly on the conic (not spherical approximation).

    """
    direction = to_camera * alpha + to_object * (1 - alpha)
    norm = np.linalg.norm(direction)
    if norm < 1e-15:
        return 1e10
    direction /= norm

    n_vec = Direction3D(direction[0], direction[1], direction[2])
    intersection = point_on_conic_surface(conic_center, n_vec, radius, conic_constant)
    if intersection is None:
        return 1e10

    surface_normal = conic_surface_normal(intersection, conic_center, radius, conic_constant)
    n = np.array([surface_normal.x, surface_normal.y, surface_normal.z])

    d_obj = np.array([intersection.x - object_pos.x, intersection.y - object_pos.y, intersection.z - object_pos.z])
    d_obj /= np.linalg.norm(d_obj)

    d_cam = np.array([camera_pos.x - intersection.x, camera_pos.y - intersection.y, camera_pos.z - intersection.z])
    d_cam /= np.linalg.norm(d_cam)

    cos_cam = np.dot(n, d_cam)
    cos_obj = -np.dot(n, d_obj)
    sin_cam = np.sqrt(max(0, 1 - cos_cam**2))
    sin_obj = np.sqrt(max(0, 1 - cos_obj**2))

    return n_outside * sin_cam - n_conic * sin_obj


def _refraction_residuals_conic(
    params: np.ndarray,
    to_camera: np.ndarray,
    to_object: np.ndarray,
    perp: np.ndarray,
    camera_pos: Position3D,
    object_pos: Position3D,
    conic_center: Position3D,
    radius: float,
    conic_constant: float,
    n_outside: float,
    n_conic: float,
) -> np.ndarray:
    """Residual function for 2D refraction finding on conic surface.

    Uses (alpha, beta) parameterization where alpha interpolates between camera
    and object directions, and beta adds an out-of-plane component. This 2D
    search is necessary because aspherical conic surfaces (k != 0) can have
    refraction points outside the camera-center-object plane.

    Returns a 2-element residual vector:
        [0]: Snell's law condition (n_outside * sin_camera - n_conic * sin_object)
        [1]: Coplanarity condition (N dot (d_object x d_camera))

    """
    alpha, beta = params

    # Search direction: in-plane interpolation + out-of-plane offset
    direction = to_camera * alpha + to_object * (1 - alpha) + perp * beta
    norm = np.linalg.norm(direction)
    if norm < 1e-15:
        return np.array([1e10, 1e10])
    direction /= norm

    n_vec = Direction3D(direction[0], direction[1], direction[2])
    intersection = point_on_conic_surface(conic_center, n_vec, radius, conic_constant)
    if intersection is None:
        return np.array([1e10, 1e10])

    surface_normal = conic_surface_normal(intersection, conic_center, radius, conic_constant)
    n = np.array([surface_normal.x, surface_normal.y, surface_normal.z])

    # Direction from object to refraction point (incident ray inside conic)
    d_obj = np.array([intersection.x - object_pos.x, intersection.y - object_pos.y, intersection.z - object_pos.z])
    d_obj /= np.linalg.norm(d_obj)

    # Direction from refraction point to camera (outgoing ray)
    d_cam = np.array([camera_pos.x - intersection.x, camera_pos.y - intersection.y, camera_pos.z - intersection.z])
    d_cam /= np.linalg.norm(d_cam)

    # Condition 1: Snell's law — n_outside * sin(θ_cam) = n_conic * sin(θ_obj)
    cos_cam = np.dot(n, d_cam)
    cos_obj = -np.dot(n, d_obj)  # negate because d_obj points away from interior
    sin_cam = np.sqrt(max(0, 1 - cos_cam**2))
    sin_obj = np.sqrt(max(0, 1 - cos_obj**2))
    residual_snell = n_outside * sin_cam - n_conic * sin_obj

    # Condition 2: coplanarity — N · (d_obj x d_cam) = 0
    residual_coplanar = np.dot(n, np.cross(d_obj, d_cam))

    return np.array([residual_snell, residual_coplanar])


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

    Uses a two-stage approach to find the point where an object ray refracts
    toward the camera through the conic surface:

    Stage 1 (brentq): Solve the 1D in-plane Snell's law residual with beta=0.
    The Snell residual is monotonic and sign-changing in alpha ∈ [0, 1], so
    brentq is guaranteed to find the unique correct root. This works directly
    on the conic surface — no spherical approximation needed.

    Stage 2 (fsolve): Starting from (alpha_brentq, 0), refine with the full
    2D system (Snell + coplanarity) to find the small out-of-plane beta
    correction needed for aspherical surfaces (k != 0).

    This approach works for any conic constant k because Stage 1 is bounded
    (no wrong roots) and Stage 2 starts very close to the solution (won't drift).

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
    try:
        to_camera_dir = (camera_pos - conic_center).normalize()
        to_object_dir = (object_pos - conic_center).normalize()
        to_camera = np.array([to_camera_dir.x, to_camera_dir.y, to_camera_dir.z])
        to_object = np.array([to_object_dir.x, to_object_dir.y, to_object_dir.z])

        # Stage 1: Find in-plane solution using brentq on the conic Snell residual.
        # alpha ∈ [0, 1] interpolates between object and camera directions —
        # the residual changes sign across this interval, guaranteeing a unique root.
        args_1d = (
            to_camera,
            to_object,
            camera_pos,
            object_pos,
            conic_center,
            radius,
            conic_constant,
            n_outside,
            n_conic,
        )
        alpha_0 = brentq(_refraction_snell_conic_1d, 0, 1, args=args_1d)

        # Stage 2: Refine with 2D fsolve for the small out-of-plane (beta) correction.
        # For aspherical surfaces the refraction point may deviate slightly from
        # the camera-center-object plane, captured by the beta parameter.
        perp = np.cross(to_camera, to_object)
        perp_norm = np.linalg.norm(perp)
        if perp_norm < 1e-15:
            arb = np.array([1.0, 0.0, 0.0]) if abs(to_camera[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            perp = np.cross(to_camera, arb)
            perp /= np.linalg.norm(perp)
        else:
            perp /= perp_norm

        args_2d = (
            to_camera,
            to_object,
            perp,
            camera_pos,
            object_pos,
            conic_center,
            radius,
            conic_constant,
            n_outside,
            n_conic,
        )
        solution, info, ier, _msg = fsolve(_refraction_residuals_conic, [alpha_0, 0.0], args=args_2d, full_output=True)

        if ier != 1 and np.max(np.abs(info["fvec"])) > 1e-8:
            return None

        alpha, beta = solution
        direction = to_camera * alpha + to_object * (1 - alpha) + perp * beta
        direction /= np.linalg.norm(direction)
        n_vec = Direction3D(direction[0], direction[1], direction[2])
        return point_on_conic_surface(conic_center, n_vec, radius, conic_constant)

    except (ValueError, TypeError):
        return None


def _refract_direction(
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


def _solve_two_surface_launch(
    camera_pos: Position3D,
    object_pos: Position3D,
    trace_outward: Callable[[np.ndarray], tuple[Point3D, Direction3D] | None],
) -> Point3D | None:
    """Find the object launch direction whose doubly-refracted ray passes through the camera.

    Generic to the corneal shape: trace_outward(launch_direction) performs the surface-specific double
    refraction and returns (exit_point, exit_direction) of the ray leaving the cornea, or None if it does
    not get through. The launch direction is searched (least-squares) in the plane around the straight
    object->camera line until the exit ray's perpendicular miss to the camera vanishes.

    Args:
        camera_pos: Camera position.
        object_pos: Intraocular object position.
        trace_outward: Callable mapping a launch direction (length-3 array) to (exit Point3D, exit
            Direction3D) or None.

    Returns:
        Anterior exit point of the converged ray, or None if no direction reaches the camera.

    """
    camera = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
    base = camera - np.array([object_pos.x, object_pos.y, object_pos.z])
    base /= np.linalg.norm(base)
    perp_a = np.cross(base, [0.0, 0.0, 1.0])
    if np.linalg.norm(perp_a) < 1e-9:
        perp_a = np.cross(base, [0.0, 1.0, 0.0])
    perp_a /= np.linalg.norm(perp_a)
    perp_b = np.cross(base, perp_a)

    def residual(offset: np.ndarray) -> list[float]:
        traced = trace_outward(base + offset[0] * perp_a + offset[1] * perp_b)
        if traced is None:
            return [1e3, 1e3]
        exit_point, exit_direction = traced
        origin = np.array([exit_point.x, exit_point.y, exit_point.z])
        direction = np.array([exit_direction.x, exit_direction.y, exit_direction.z])
        miss = (camera - origin) - np.dot(camera - origin, direction) * direction
        across = np.cross(direction, perp_a)
        across /= np.linalg.norm(across)
        return [float(np.dot(miss, across)), float(np.dot(miss, np.cross(direction, across)))]

    solution = least_squares(residual, [0.0, 0.0], xtol=1e-12, ftol=1e-12)
    if np.max(np.abs(residual(solution.x))) > 1e-6:
        return None
    traced = trace_outward(base + solution.x[0] * perp_a + solution.x[1] * perp_b)
    return traced[0] if traced is not None else None


def find_refraction_dual_sphere(
    camera_pos: Position3D,
    object_pos: Position3D,
    anterior_center: Position3D,
    anterior_radius: float,
    posterior_center: Position3D,
    posterior_radius: float,
    n_outside: float,
    n_cornea: float,
    n_aqueous: float,
) -> Point3D | None:
    """Anterior exit point of the two-surface refraction through a spherical cornea.

    A ray leaves the object in the aqueous, refracts at the posterior spherical surface (aqueous -> cornea)
    then the anterior spherical surface (cornea -> air), and reaches the camera. The returned point is
    where the exiting ray leaves the anterior surface -- the apparent corneal point the camera images. All
    positions must share one coordinate frame.

    Args:
        camera_pos: Camera position.
        object_pos: Intraocular object position (in the aqueous).
        anterior_center: Center of the anterior spherical surface.
        anterior_radius: Radius of the anterior spherical surface.
        posterior_center: Center of the posterior spherical surface.
        posterior_radius: Radius of the posterior spherical surface.
        n_outside: Refractive index outside the cornea (air).
        n_cornea: Refractive index of the cornea.
        n_aqueous: Refractive index of the aqueous humor.

    Returns:
        Anterior exit point, or None if no valid path reaches the camera.

    """

    def trace_outward(launch: np.ndarray) -> tuple[Point3D, Direction3D] | None:
        unit = launch / np.linalg.norm(launch)
        ray = Ray(origin=object_pos.to_point3d(), direction=Direction3D(unit[0], unit[1], unit[2]))
        posterior_hit, _ = intersect_ray_sphere(ray, posterior_center, posterior_radius)
        if posterior_hit is None or not posterior_hit.intersects:
            return None
        posterior_point = posterior_hit.point
        posterior_normal = (posterior_point - posterior_center.to_point3d()).to_direction3d().normalize()
        cornea_direction = _refract_direction(ray.direction, posterior_normal, n_aqueous, n_cornea)
        if cornea_direction is None:
            return None
        anterior_hit, _ = intersect_ray_sphere(
            Ray(origin=posterior_point, direction=cornea_direction), anterior_center, anterior_radius
        )
        if anterior_hit is None or not anterior_hit.intersects:
            return None
        anterior_point = anterior_hit.point
        anterior_normal = (anterior_point - anterior_center.to_point3d()).to_direction3d().normalize()
        exit_direction = _refract_direction(cornea_direction, anterior_normal, n_cornea, n_outside)
        if exit_direction is None:
            return None
        return anterior_point, exit_direction

    return _solve_two_surface_launch(camera_pos, object_pos, trace_outward)


def find_refraction_dual_conic(
    camera_pos: Position3D,
    object_pos: Position3D,
    anterior_center: Position3D,
    anterior_radius: float,
    anterior_k: float,
    posterior_center: Position3D,
    posterior_radius: float,
    posterior_k: float,
    n_outside: float,
    n_cornea: float,
    n_aqueous: float,
) -> Point3D | None:
    """Anterior exit point of the two-surface refraction through a conic cornea.

    As find_refraction_dual_sphere, but each corneal surface is an aspheric conic, so intersection and
    normal use the conic primitives. A ray leaves the object in the aqueous, refracts at the posterior
    conic surface (aqueous -> cornea) then the anterior conic surface (cornea -> air), and reaches the
    camera; the returned point is where the exiting ray leaves the anterior surface.

    Args:
        camera_pos: Camera position.
        object_pos: Intraocular object position (in the aqueous).
        anterior_center: Center of the anterior conic surface.
        anterior_radius: Apex radius of curvature of the anterior surface.
        anterior_k: Conic constant of the anterior surface.
        posterior_center: Center of the posterior conic surface.
        posterior_radius: Apex radius of curvature of the posterior surface.
        posterior_k: Conic constant of the posterior surface.
        n_outside: Refractive index outside the cornea (air).
        n_cornea: Refractive index of the cornea.
        n_aqueous: Refractive index of the aqueous humor.

    Returns:
        Anterior exit point, or None if no valid path reaches the camera.

    """

    def trace_outward(launch: np.ndarray) -> tuple[Point3D, Direction3D] | None:
        unit = launch / np.linalg.norm(launch)
        ray = Ray(origin=object_pos.to_point3d(), direction=Direction3D(unit[0], unit[1], unit[2]))
        posterior_hit, _ = intersect_ray_conic(ray, posterior_center, posterior_radius, posterior_k)
        if posterior_hit is None or not posterior_hit.intersects:
            return None
        posterior_point = posterior_hit.point
        posterior_normal = conic_surface_normal(posterior_point, posterior_center, posterior_radius, posterior_k)
        cornea_direction = _refract_direction(ray.direction, posterior_normal, n_aqueous, n_cornea)
        if cornea_direction is None:
            return None
        anterior_hit, _ = intersect_ray_conic(
            Ray(origin=posterior_point, direction=cornea_direction), anterior_center, anterior_radius, anterior_k
        )
        if anterior_hit is None or not anterior_hit.intersects:
            return None
        anterior_point = anterior_hit.point
        anterior_normal = conic_surface_normal(anterior_point, anterior_center, anterior_radius, anterior_k)
        exit_direction = _refract_direction(cornea_direction, anterior_normal, n_cornea, n_outside)
        if exit_direction is None:
            return None
        return anterior_point, exit_direction

    return _solve_two_surface_launch(camera_pos, object_pos, trace_outward)


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
