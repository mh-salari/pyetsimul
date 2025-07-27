import numpy as np
import warnings
from scipy.optimize import brentq
from ..geometry.intersections import intersect_ray_circle, intersect_ray_sphere


def reflect_objective(a, L, C, S0, Sr):
    """Objective function for reflection finding.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    Helper function used by find_reflection to locate glint positions on spherical
    surfaces. Returns angle difference between incident and reflected rays for
    a given interpolation parameter 'a'.

    Args:
        a: Interpolation parameter between light and camera directions
        L: Light source position
        C: Camera position
        S0: Sphere center
        Sr: Sphere radius

    Returns:
        Tuple of (angle_diff, U0) where angle_diff is the reflection angle error
        and U0 is the potential glint position
    """
    # Suppress numpy warnings to provide cleaner reflection error messages
    with np.errstate(invalid='ignore', divide='ignore'):
        # to_c=(C-S0)/norm(C-S0)
        to_c = (C - S0) / np.linalg.norm(C - S0)

        # to_l=(L-S0)/norm(L-S0)
        to_l = (L - S0) / np.linalg.norm(L - S0)

        # n=a*to_c+(1-a)*to_l
        n = a * to_c + (1 - a) * to_l

        # n=n/norm(n)
        n = n / np.linalg.norm(n)

        # U0=S0+Sr*n
        U0 = S0 + Sr * n

        # angle_c=arccos(n'*(C-U0)/norm(C-U0)) with safe clipping
        angle_c = np.arccos(np.clip(np.dot(n, (C - U0) / np.linalg.norm(C - U0)), -1, 1))

        # angle_l=arccos(n'*(L-U0)/norm(L-U0)) with safe clipping
        angle_l = np.arccos(np.clip(np.dot(n, (L - U0) / np.linalg.norm(L - U0)), -1, 1))

        # angle_diff=angle_c-angle_l
        angle_diff = angle_c - angle_l

        return angle_diff, U0


def find_reflection(L, C, S0, Sr):
    """Finds position of a glint on the surface of a sphere.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    U0 = find_reflection(L, C, S0, Sr) finds the position on a sphere with
    center S0 and radius Sr where a ray emanating from a light source at 'L'
    is reflected to pass directly through point 'C' (this could be a camera,
    for example).

    Args:
        L: Light source position
        C: Camera position
        S0: Sphere center
        Sr: Sphere radius

    Returns:
        U0: Position of glint on sphere surface, or None if no reflection found
    """
    try:
        # a=brentq(@(a) reflect_objective(a, L, C, S0, Sr), 0, 1)
        a = brentq(lambda a: reflect_objective(a, L, C, S0, Sr)[0], 0, 1)

        # [dummy, U0]=reflect_objective(a, L, C, S0, Sr)
        _, U0 = reflect_objective(a, L, C, S0, Sr)

        return U0
    except ValueError as e:
        warnings.warn(
            f"No glint found due to degenerate geometry: Light={L}, Camera={C}, Sphere center={S0}",
            RuntimeWarning,
        )
        return None


def reflect_ray_circle(R0, Rd, C0, Cr):
    """Reflects ray on circle.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

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

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    [U0, Ud] = reflect_ray_sphere(R0, Rd, S0, Sr) find the point 'U0' at
    which a ray (specified by its origin 'R0' and direction 'Rd') strikes a
    sphere (with center 'S0' and radius 'Sr') and computes the direction 'Ud'
    of the reflected ray. The result is undefined if the original ray does
    not intersect the sphere.

    Args:
        R0: Ray origin (3D or homogeneous)
        Rd: Ray direction (3D or homogeneous)
        S0: Sphere center (3D or homogeneous)
        Sr: Sphere radius

    Returns:
        Tuple of (U0, Ud) where U0 is intersection point, Ud is reflected direction.
        Returns (None, None) if no intersection.
    """
    # Store original inputs for format preservation
    R0_orig = R0.copy()
    Rd_orig = Rd.copy()
    S0_orig = S0.copy()

    # Extract 3D components for calculations
    R0_3d = R0[:3] if len(R0) == 4 else R0
    Rd_3d = Rd[:3] if len(Rd) == 4 else Rd
    S0_3d = S0[:3] if len(S0) == 4 else S0

    # Normalize ray direction
    Rd_3d = Rd_3d / np.linalg.norm(Rd_3d)

    # Find intersection with sphere
    U0_3d, _ = intersect_ray_sphere(
        R0_orig,
        np.concatenate([Rd_3d, [0]]) if len(Rd_orig) == 4 else Rd_3d,
        S0_orig,
        Sr,
    )

    if U0_3d is None:
        return None, None

    # Calculate surface normal at intersection
    N = (U0_3d - S0_3d) / np.linalg.norm(U0_3d - S0_3d)

    # Apply reflection formula: Ud = Rd - 2*N*(Rd'*N)
    Ud_3d = Rd_3d - 2 * N * np.dot(Rd_3d, N)

    # Preserve input format for output
    if len(R0_orig) == 4:  # Input was homogeneous
        # Format U0 as homogeneous position (w=1)
        U0 = np.zeros(4)
        U0[:3] = U0_3d
        U0[3] = 1  # Position vector has 1 in homogeneous component

        # Format Ud as homogeneous direction (w=0)
        Ud = np.zeros(4)
        Ud[:3] = Ud_3d
        Ud[3] = 0  # Direction vector has 0 in homogeneous component
    else:
        # Keep 3D format
        U0 = U0_3d
        Ud = Ud_3d

    return U0, Ud
