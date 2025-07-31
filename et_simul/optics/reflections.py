import numpy as np
import warnings
from scipy.optimize import brentq
from ..geometry.intersections import (
    intersect_ray_circle,
    intersect_ray_sphere,
    intersect_ray_conic,
    conic_surface_normal,
)


def _reflect_objective_sphere(a, L, C, S0, Sr):
    """Objective function for reflection finding.

    Helper function used by find_reflection to locate glint positions on spherical
    surfaces. Returns angle difference between incident and reflected rays for
    a given interpolation parameter 'a'.

    Args:
        a: Interpolation parameter between light and camera directions
        L: Light source position (4D homogeneous)
        C: Camera position (4D homogeneous)
        S0: Sphere center (4D homogeneous)
        Sr: Sphere radius

    Returns:
        Tuple of (angle_diff, U0) where angle_diff is the reflection angle error
        and U0 is the potential glint position (4D homogeneous)
    """
    # Suppress numpy warnings to provide cleaner reflection error messages
    with np.errstate(invalid="ignore", divide="ignore"):
        # Work with 4D homogeneous coordinates
        C_vec = C - S0
        L_vec = L - S0
        to_c = C_vec[:3] / np.linalg.norm(C_vec[:3])
        to_l = L_vec[:3] / np.linalg.norm(L_vec[:3])
        n = a * to_c + (1 - a) * to_l
        n = n / np.linalg.norm(n)
        
        # Create 4D homogeneous point
        U0 = S0.copy()
        U0[:3] = S0[:3] + Sr * n
        
        # Calculate angles using 4D coordinates
        C_to_U0 = C - U0
        L_to_U0 = L - U0
        angle_c = np.arccos(np.clip(np.dot(n, C_to_U0[:3] / np.linalg.norm(C_to_U0[:3])), -1, 1))
        angle_l = np.arccos(np.clip(np.dot(n, L_to_U0[:3] / np.linalg.norm(L_to_U0[:3])), -1, 1))
        angle_diff = angle_c - angle_l
        return angle_diff, U0


def find_reflection_sphere(L, C, S0, Sr):
    """Finds position of a glint on the surface of a sphere.

    U0 = find_reflection(L, C, S0, Sr) finds the position on a sphere with
    center S0 and radius Sr where a ray emanating from a light source at 'L'
    is reflected to pass directly through point 'C' (this could be a camera,
    for example).

    Args:
        L: Light source position (4D homogeneous)
        C: Camera position (4D homogeneous)
        S0: Sphere center (4D homogeneous)
        Sr: Sphere radius

    Returns:
        U0: Position of glint on sphere surface (4D homogeneous), or None if no reflection found
    """
    # Work directly with 4D homogeneous coordinates
    try:
        a = brentq(lambda a: _reflect_objective_sphere(a, L, C, S0, Sr)[0], 0, 1)
        _, U0 = _reflect_objective_sphere(a, L, C, S0, Sr)
        return U0
    except ValueError:
        warnings.warn(
            f"No glint found due to degenerate geometry: Light={L}, Camera={C}, Sphere center={S0}",
            RuntimeWarning,
        )
        return None


def _reflect_objective_conic(alpha, L, C, S0, r, k):
    """Objective function for conic section reflection finding.

    Uses the same interpolation approach as the sphere version but projects
    to the conic surface using proper conic section geometry.
    """
    # Import here to avoid circular import
    from ..geometry.intersections import point_on_conic_surface, conic_surface_normal

    # Suppress numpy warnings to provide cleaner reflection error messages
    with np.errstate(invalid="ignore", divide="ignore"):
        # Work with 4D homogeneous coordinates for vector calculations
        C_vec = C - S0
        L_vec = L - S0
        to_c = C_vec[:3] / np.linalg.norm(C_vec[:3])
        to_l = L_vec[:3] / np.linalg.norm(L_vec[:3])
        n = alpha * to_c + (1 - alpha) * to_l
        n = n / np.linalg.norm(n)

        U0 = point_on_conic_surface(S0, n, r, k)
        if U0 is None:
            return float("inf"), None

        surface_normal = conic_surface_normal(U0, S0, r, k)
        
        # Calculate angles using 4D homogeneous coordinates
        C_to_U0 = C - U0
        L_to_U0 = L - U0
        angle_c = np.arccos(np.clip(np.dot(surface_normal, C_to_U0[:3] / np.linalg.norm(C_to_U0[:3])), -1, 1))
        angle_l = np.arccos(np.clip(np.dot(surface_normal, L_to_U0[:3] / np.linalg.norm(L_to_U0[:3])), -1, 1))
        angle_diff = angle_c - angle_l

        return angle_diff, U0


def find_reflection_conic(L, C, S0, r, k):
    """Finds position of a glint on the surface of a conic section.

    Args:
        L: Light source position (4D homogeneous)
        C: Camera position (4D homogeneous)
        S0: Conic center (4D homogeneous, typically corneal apex)
        r: Radius of curvature at apex (meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        U0: Position of glint on conic surface (4D homogeneous), or None if no reflection found
    """
    try:
        alpha = brentq(lambda alpha: _reflect_objective_conic(alpha, L, C, S0, r, k)[0], 0, 1)
        _, U0 = _reflect_objective_conic(alpha, L, C, S0, r, k)
        return U0
    except (ValueError, TypeError):
        warnings.warn(
            f"No glint found on conic surface: Light={L}, Camera={C}, Conic center={S0}",
            RuntimeWarning,
        )
        return None


def reflect_ray_circle(R0, Rd, C0, Cr):
    """Reflects ray on circle.



    [S0, Sd] = reflect_ray_circle(R0, Rd, C0, Cr) finds the point 'S0' at
    which a ray (specified by its origin 'R0' and direction 'Rd') strikes a
    circle (with center 'C0' and radius 'Cr') and computes the direction 'Sd'
    of the reflected ray. The result is undefined if the original ray does
    not intersect the circle.

    Args:
        R0: Ray origin (2D)
        Rd: Ray direction (2D)
        C0: Circle center (2D)
        Cr: Circle radius

    Returns:
        Tuple of (S0, Sd) where S0 is intersection point, Sd is reflected direction.
        Returns (None, None) if no intersection.
    """
    # Copy inputs to avoid modifying original arrays
    R0_work = R0.copy()
    Rd_work = Rd.copy()
    C0_work = C0.copy()

    # Normalize ray direction
    Rd_work = Rd_work / np.linalg.norm(Rd_work)

    # Find intersection point
    S0 = intersect_ray_circle(R0_work, Rd_work, C0_work, Cr)

    if S0 is None:
        return None, None

    # Calculate surface normal at intersection
    N = (S0 - C0_work) / np.linalg.norm(S0 - C0_work)

    # Apply reflection formula: Sd = Rd - 2*N*(Rd'*N)
    Sd = Rd_work - 2 * N * np.dot(Rd_work, N)

    return S0, Sd


def reflect_ray_sphere(R0, Rd, S0, Sr):
    """Reflects ray on sphere.

    [U0, Ud] = reflect_ray_sphere(R0, Rd, S0, Sr) find the point 'U0' at
    which a ray (specified by its origin 'R0' and direction 'Rd') strikes a
    sphere (with center 'S0' and radius 'Sr') and computes the direction 'Ud'
    of the reflected ray. The result is undefined if the original ray does
    not intersect the sphere.

    Args:
        R0: Ray origin (4D homogeneous)
        Rd: Ray direction (4D homogeneous)
        S0: Sphere center (4D homogeneous)
        Sr: Sphere radius

    Returns:
        Tuple of (U0, Ud) where U0 is intersection point, Ud is reflected direction (4D homogeneous).
        Returns (None, None) if no intersection.
    """
    # Normalize ray direction using spatial component
    Rd_normalized = Rd[:3] / np.linalg.norm(Rd[:3])

    # Find intersection with sphere
    U0, _ = intersect_ray_sphere(R0, Rd, S0, Sr)

    if U0 is None:
        return None, None

    # Calculate surface normal at intersection using 4D coordinates
    U0_to_S0 = U0 - S0
    N = U0_to_S0[:3] / np.linalg.norm(U0_to_S0[:3])

    # Apply reflection formula: Ud = Rd - 2*N*(Rd'*N)
    Ud_3d = Rd_normalized - 2 * N * np.dot(Rd_normalized, N)

    # Format Ud as homogeneous direction (w=0)
    Ud = np.zeros(4)
    Ud[:3] = Ud_3d
    Ud[3] = 0.0
    return U0, Ud


def reflect_ray_conic(R0, Rd, S0, r_apical, k):
    """
    Reflect ray at surface of conic section.

    This is the conic equivalent of reflect_ray_sphere() used in the eye simulator.

    Args:
        R0: Ray origin (4D homogeneous)
        Rd: Ray direction (4D homogeneous)
        S0: Conic center (4D homogeneous, typically corneal apex)
        r_apical: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Tuple of (U0, Ud) where:
        - U0: Intersection point on conic surface (4D homogeneous)
        - Ud: Reflected ray direction (4D homogeneous)
        Returns (None, None) if no intersection.
    """
    # Normalize ray direction using spatial component
    Rd_normalized = Rd[:3] / np.linalg.norm(Rd[:3])

    # Find intersection point
    U0, _ = intersect_ray_conic(R0, Rd, S0, r_apical, k)

    if U0 is None:
        return None, None

    # Calculate surface normal at intersection point
    N = conic_surface_normal(U0, S0, r_apical, k)

    # For reflection, we typically want outward-pointing normal
    center_to_point_vec = U0 - S0
    if np.dot(N, center_to_point_vec[:3]) < 0:  # Normal points inward
        N = -N  # Flip to point outward

    # Apply reflection formula
    Ud_3d = Rd_normalized - 2 * N * np.dot(Rd_normalized, N)

    # Format Ud as homogeneous direction (w=0)
    Ud = np.zeros(4)
    Ud[:3] = Ud_3d
    Ud[3] = 0.0
    return U0, Ud
