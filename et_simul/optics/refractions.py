import numpy as np
from scipy.optimize import brentq
from ..geometry.intersections import intersect_ray_sphere, intersect_ray_conic, conic_surface_normal


def _refraction_objective_sphere(a, C, O, S0, Sr, n_outside, n_sphere):
    """Objective function for finding refraction point on sphere surface.

    Args:
        a: Interpolation parameter between camera and object directions
        C: Camera position (4D homogeneous)
        O: Object position (4D homogeneous)
        S0: Sphere center (4D homogeneous)
        Sr: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Tuple of (diff, U0) where diff is Snell's law difference and U0 is surface point (4D homogeneous)
    """
    # Compute vectors from sphere center to camera and object (work in 4D homogeneous space)
    C_vec = C - S0
    O_vec = O - S0
    to_c = C_vec / np.linalg.norm(C_vec[:3])
    to_o = O_vec / np.linalg.norm(O_vec[:3])

    # Interpolate and normalize to get surface normal (3D spatial component)
    n_spatial = a * to_c[:3] + (1 - a) * to_o[:3]
    n_spatial = n_spatial / np.linalg.norm(n_spatial)

    # Compute point on surface of sphere (4D homogeneous)
    U0 = S0.copy()
    U0[:3] = S0[:3] + Sr * n_spatial

    # Compute angles with surface normal
    C_to_U0 = C - U0
    U0_to_O = U0 - O
    cos_angle_c = np.dot(n_spatial, C_to_U0[:3] / np.linalg.norm(C_to_U0[:3]))
    cos_angle_o = np.dot(n_spatial, U0_to_O[:3] / np.linalg.norm(U0_to_O[:3]))

    # Safe sqrt to handle numerical errors
    sin_angle_c = np.sqrt(max(0, 1 - cos_angle_c**2))
    sin_angle_o = np.sqrt(max(0, 1 - cos_angle_o**2))

    # Snell's law difference
    diff = n_outside * sin_angle_c - n_sphere * sin_angle_o

    return diff, U0


def find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere):
    """Computes image produced by refracting sphere.



    I = find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere) finds the position
    on a sphere with center S0 and radius Sr where a ray emanating from an
    object at a position 'O' inside the sphere is refracted to pass directly
    through point 'C' (this could be a camera, for example). The refractive
    index of the sphere is 'n_sphere', that of the outside medium is
    'n_outside'.

    Args:
        C: Camera/observer position (3D or 4D homogeneous)
        O: Object position inside sphere (3D or 4D homogeneous)
        S0: Sphere center (3D or 4D homogeneous)
        Sr: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Position on sphere surface where refraction occurs (same coordinate type as input), or None if not found.
    """
    # Work directly with 4D homogeneous coordinates
    try:
        a = brentq(lambda x: _refraction_objective_sphere(x, C, O, S0, Sr, n_outside, n_sphere)[0], 0, 1)
        _, result = _refraction_objective_sphere(a, C, O, S0, Sr, n_outside, n_sphere)
        return result
    except (ValueError, RuntimeError):
        return None


def _refraction_objective_conic(alpha, C, O, S0, r, k, n_outside, n_conic):
    """Objective function for finding refraction point on conic surface.

    Args:
        alpha: Interpolation parameter between camera and object directions
        C: Camera position (4D homogeneous)
        O: Object position (4D homogeneous)
        S0: Conic center (4D homogeneous, typically corneal apex)
        r: Radius of curvature at apex (meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic
        n_conic: Refractive index of conic

    Returns:
        Tuple of (diff, intersection) where diff is Snell's law difference and intersection is surface point (4D homogeneous)
    """
    # Import here to avoid circular import
    from ..geometry.intersections import point_on_conic_surface, conic_surface_normal

    # Work with 4D homogeneous coordinates, use spatial components for vector math
    C_vec = C - S0
    O_vec = O - S0
    to_c = C_vec[:3] / np.linalg.norm(C_vec[:3])
    to_o = O_vec[:3] / np.linalg.norm(O_vec[:3])

    direction = alpha * to_c + (1 - alpha) * to_o
    direction = direction / np.linalg.norm(direction)

    # Find intersection with conic surface along this direction
    intersection = point_on_conic_surface(S0, direction, r, k)
    if intersection is None:
        return float("inf"), None

    # Get surface normal at intersection point
    normal = conic_surface_normal(intersection, S0, r, k)
    if normal is None:
        return float("inf"), None

    # Compute angles with surface normal using 4D homogeneous coordinates
    C_to_intersection = C - intersection
    intersection_to_O = intersection - O
    to_camera = C_to_intersection[:3] / np.linalg.norm(C_to_intersection[:3])
    to_object = intersection_to_O[:3] / np.linalg.norm(intersection_to_O[:3])

    cos_angle_c = np.dot(normal, to_camera)
    cos_angle_o = np.dot(normal, to_object)

    sin_angle_c = np.sqrt(max(0, 1 - cos_angle_c**2))
    sin_angle_o = np.sqrt(max(0, 1 - cos_angle_o**2))

    # Snell's law difference
    diff = n_outside * sin_angle_c - n_conic * sin_angle_o

    return diff, intersection


def find_refraction_conic(C, O, S0, r, k, n_outside, n_conic):
    """Computes image produced by refracting conic section.

    I = find_refraction_conic(C, O, S0, r, k, n_outside, n_conic) finds the position
    on a conic surface with center S0 and conic constant k where a ray emanating from an
    object at a position 'O' inside the conic is refracted to pass directly
    through point 'C' (this could be a camera, for example). The refractive
    index of the conic is 'n_conic', that of the outside medium is 'n_outside'.

    Args:
        C: Camera/observer position (4D homogeneous)
        O: Object position inside conic (4D homogeneous)
        S0: Conic center (4D homogeneous, typically corneal apex)
        r: Radius of curvature at apex (meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic
        n_conic: Refractive index of conic

    Returns:
        Position on conic surface where refraction occurs (4D homogeneous), or None if not found.
    """
    # Work with 4D homogeneous coordinates
    C_vec = C - S0
    O_vec = O - S0
    to_c = C_vec[:3] / np.linalg.norm(C_vec[:3])
    to_o = O_vec[:3] / np.linalg.norm(O_vec[:3])

    if abs(to_c[2] - to_o[2]) < 1e-9:
        if to_c[2] >= 0:
            return None
        upper_bound = 1.0
    else:
        alpha_zero = -to_o[2] / (to_c[2] - to_o[2])
        upper_bound = min(1.0, max(0, alpha_zero - 1e-9))

    f0, _ = _refraction_objective_conic(0, C, O, S0, r, k, n_outside, n_conic)
    f1, _ = _refraction_objective_conic(upper_bound, C, O, S0, r, k, n_outside, n_conic)

    if np.isinf(f0) or np.isinf(f1) or f0 * f1 > 0:
        return None

    try:
        alpha = brentq(
            lambda x: _refraction_objective_conic(x, C, O, S0, r, k, n_outside, n_conic)[0],
            0,
            upper_bound,
        )
        _, I = _refraction_objective_conic(alpha, C, O, S0, r, k, n_outside, n_conic)
        return I
    except (ValueError, RuntimeError):
        return None


def refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere):
    """Refracts ray at surface of sphere.

    [U0, Ud] = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere) finds
    the point 'U0' at which a ray (specified by its origin 'R0' and direction
    'Rd') strikes a sphere (with center 'S0' and radius 'Sr') and computes
    the direction 'Ud' of the refracted ray. The refractive index outside
    the sphere is 'n_outside', the refractive index of the sphere is
    'n_sphere'.

    Args:
        R0: Ray origin (4D homogeneous)
        Rd: Ray direction (4D homogeneous)
        S0: Sphere center (4D homogeneous)
        Sr: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Tuple of (U0, Ud) where U0 is intersection point, Ud is refracted direction (4D homogeneous).
        Returns (None, None) if no intersection or total internal reflection.
    """
    # Normalize ray direction using spatial component
    Rd_normalized = Rd[:3] / np.linalg.norm(Rd[:3])

    # Find point of intersection
    U0, _ = intersect_ray_sphere(R0, Rd, S0, Sr)

    if U0 is None:
        return None, None

    # Find surface normal at point of intersection (pointing inwards) using 4D coordinates
    S0_to_U0 = S0 - U0
    N = S0_to_U0[:3] / np.linalg.norm(S0_to_U0[:3])

    # Find cosines
    costh1 = np.dot(Rd_normalized, N)
    costh2_squared = 1 - (n_outside / n_sphere) ** 2 * (1 - costh1**2)

    # Check for total internal reflection
    if costh2_squared < 0:
        return U0, None

    costh2 = np.sqrt(costh2_squared)

    # Snell's law refraction formula
    Ud_3d = (n_outside / n_sphere) * Rd_normalized + (costh2 - (n_outside / n_sphere) * costh1) * N

    # Format Ud as homogeneous direction (w=0)
    Ud = np.zeros(4)
    Ud[:3] = Ud_3d
    Ud[3] = 0.0
    return U0, Ud


def refract_ray_conic(R0, Rd, S0, r_apical, k, n_outside, n_conic):
    """
    Refract ray at surface of conic section.

    This is the conic equivalent of refract_ray_sphere() used in the eye simulator,
    implementing proper corneal asphericity with k-value.

    Args:
        R0: Ray origin (4D homogeneous)
        Rd: Ray direction (4D homogeneous)
        S0: Conic center (4D homogeneous, typically corneal apex)
        r_apical: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic (e.g., air = 1.0)
        n_conic: Refractive index of conic (e.g., cornea = 1.376)

    Returns:
        Tuple of (U0, Ud) where:
        - U0: Intersection point on conic surface (4D homogeneous)
        - Ud: Refracted ray direction (4D homogeneous)
        Returns (None, None) if no intersection or total internal reflection.
    """
    # Normalize ray direction using spatial component
    Rd_normalized = Rd[:3] / np.linalg.norm(Rd[:3])

    # Find intersection point
    U0, _ = intersect_ray_conic(R0, Rd, S0, r_apical, k)

    if U0 is None:
        return None, None

    # Calculate surface normal at intersection point
    N = conic_surface_normal(U0, S0, r_apical, k)

    # For refraction, we need inward-pointing normal (toward conic interior)
    center_to_point_vec = U0 - S0
    if np.dot(N, center_to_point_vec[:3]) > 0:  # Normal points outward
        N = -N  # Flip to point inward

    # Apply Snell's law
    costh1 = np.dot(Rd_normalized, N)
    costh2_squared = 1 - (n_outside / n_conic) ** 2 * (1 - costh1**2)

    # Check for total internal reflection
    if costh2_squared < 0:
        return U0, None

    costh2 = np.sqrt(costh2_squared)

    # Snell's law refraction formula
    Ud_3d = (n_outside / n_conic) * Rd_normalized + (costh2 - (n_outside / n_conic) * costh1) * N

    # Format Ud as homogeneous direction (w=0)
    Ud = np.zeros(4)
    Ud[:3] = Ud_3d
    Ud[3] = 0.0
    return U0, Ud
