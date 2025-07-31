import numpy as np
from scipy.optimize import brentq
from ..geometry.intersections import intersect_ray_sphere, intersect_ray_conic, conic_surface_normal


def _refraction_objective_sphere(a, C_3d, O_3d, S0_3d, Sr, n_outside, n_sphere):
    """Objective function for finding refraction point on sphere surface.

    Args:
        a: Interpolation parameter between camera and object directions
        C_3d: Camera position (3D)
        O_3d: Object position (3D)
        S0_3d: Sphere center (3D)
        Sr: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Tuple of (diff, U0) where diff is Snell's law difference and U0 is surface point
    """
    # Compute vectors from sphere center to camera and object
    to_c = (C_3d - S0_3d) / np.linalg.norm(C_3d - S0_3d)
    to_o = (O_3d - S0_3d) / np.linalg.norm(O_3d - S0_3d)

    # Interpolate and normalize to get surface normal
    n = a * to_c + (1 - a) * to_o
    n = n / np.linalg.norm(n)

    # Compute point on surface of sphere
    U0 = S0_3d + Sr * n

    # Compute angles with surface normal
    cos_angle_c = np.dot(n, (C_3d - U0) / np.linalg.norm(C_3d - U0))
    cos_angle_o = np.dot(n, (U0 - O_3d) / np.linalg.norm(U0 - O_3d))

    # Safe sqrt to handle numerical errors
    sin_angle_c = np.sqrt(max(0, 1 - cos_angle_c**2))
    sin_angle_o = np.sqrt(max(0, 1 - cos_angle_o**2))

    # Snell's law difference
    diff = n_outside * sin_angle_c - n_sphere * sin_angle_o

    return diff, U0


def find_refraction_sphere(C, O, S0, Sr, n_outside, n_sphere):
    """Computes image produced by refracting sphere.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

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
    # Extract 3D spatial components for calculations
    C_3d = C[:3] if len(C) > 3 else C
    O_3d = O[:3] if len(O) > 3 else O
    S0_3d = S0[:3] if len(S0) > 3 else S0

    # Determine output coordinate type from input (preserve input format)
    is_homogeneous = len(C) > 3 or len(O) > 3 or len(S0) > 3

    # Find zero of objective function
    try:
        a = brentq(lambda x: _refraction_objective_sphere(x, C_3d, O_3d, S0_3d, Sr, n_outside, n_sphere)[0], 0, 1)
        _, I_3d = _refraction_objective_sphere(a, C_3d, O_3d, S0_3d, Sr, n_outside, n_sphere)

        # Return result in same coordinate type as input
        if is_homogeneous:
            result = np.array([I_3d[0], I_3d[1], I_3d[2], 1.0])
        else:
            result = I_3d
        return result
    except (ValueError, RuntimeError):
        return None


def _refraction_objective_conic(alpha, C_3d, O_3d, S0_3d, r, k, n_outside, n_conic):
    """Objective function for finding refraction point on conic surface.

    Args:
        alpha: Interpolation parameter between camera and object directions
        C_3d: Camera position (3D)
        O_3d: Object position (3D)
        S0_3d: Conic center (3D, typically corneal apex)
        r: Radius of curvature at apex (meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic
        n_conic: Refractive index of conic

    Returns:
        Tuple of (diff, intersection) where diff is Snell's law difference and intersection is surface point
    """
    # Import here to avoid circular import
    from ..geometry.intersections import point_on_conic_surface, conic_surface_normal

    # Interpolate between object and camera directions from conic center
    to_c = (C_3d - S0_3d) / np.linalg.norm(C_3d - S0_3d)
    to_o = (O_3d - S0_3d) / np.linalg.norm(O_3d - S0_3d)

    # Interpolated direction (not necessarily normalized)
    direction = alpha * to_c + (1 - alpha) * to_o
    direction = direction / np.linalg.norm(direction)

    # Find intersection with conic surface along this direction
    intersection = point_on_conic_surface(S0_3d, direction, r, k)
    if intersection is None:
        return float("inf"), None

    # Get surface normal at intersection point
    normal = conic_surface_normal(intersection, S0_3d, r, k)
    if normal is None:
        return float("inf"), None

    # Compute angles with surface normal
    to_camera = (C_3d - intersection) / np.linalg.norm(C_3d - intersection)
    to_object = (intersection - O_3d) / np.linalg.norm(intersection - O_3d)

    cos_angle_c = np.dot(normal, to_camera)
    cos_angle_o = np.dot(normal, to_object)

    # Safe sqrt to handle numerical errors
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
        C: Camera/observer position (3D or 4D homogeneous)
        O: Object position inside conic (3D or 4D homogeneous)
        S0: Conic center (3D or 4D homogeneous, typically corneal apex)
        r: Radius of curvature at apex (meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic
        n_conic: Refractive index of conic

    Returns:
        Position on conic surface where refraction occurs (same coordinate type as input), or None if not found.
    """
    # Extract 3D spatial components for calculations
    C_3d = C[:3] if len(C) > 3 else C
    O_3d = O[:3] if len(O) > 3 else O
    S0_3d = S0[:3] if len(S0) > 3 else S0

    # Determine output coordinate type from input (preserve input format)
    is_homogeneous = len(C) > 3 or len(O) > 3 or len(S0) > 3

    # --- Start of fix ---
    # The conic surface opens toward -z. The objective function interpolates a direction
    # between the object and camera. If this direction's z-component is positive,
    # it points away from the surface, and there's no intersection, causing the
    # objective to return 'inf' and the solver to fail.
    # We must limit the search interval for the interpolation parameter 'alpha'
    # to a range that produces valid directions.

    to_c = (C_3d - S0_3d) / np.linalg.norm(C_3d - S0_3d)
    to_o = (O_3d - S0_3d) / np.linalg.norm(O_3d - S0_3d)

    to_c_z = to_c[2]
    to_o_z = to_o[2]

    # Calculate the alpha at which the direction's z-component is zero
    # alpha * to_c_z + (1-alpha) * to_o_z = 0  => alpha = -to_o_z / (to_c_z - to_o_z)
    if abs(to_c_z - to_o_z) < 1e-9:
        # This can happen if both points are on a plane perpendicular to the z-axis
        # relative to the conic center. If the direction doesn't point towards -z, no solution.
        if to_c_z >= 0:
            return None
        upper_bound = 1.0
    else:
        alpha_zero = -to_o_z / (to_c_z - to_o_z)
        # The valid search range is [0, alpha_zero]. We use a slightly smaller
        # upper bound to avoid floating point issues at the boundary.
        upper_bound = min(1.0, max(0, alpha_zero - 1e-9))

    # Check if there's a sign change in the objective function over the valid range
    f0, _ = _refraction_objective_conic(0, C_3d, O_3d, S0_3d, r, k, n_outside, n_conic)
    f1, _ = _refraction_objective_conic(upper_bound, C_3d, O_3d, S0_3d, r, k, n_outside, n_conic)
    # --- End of fix ---

    if np.isinf(f0) or np.isinf(f1) or f0 * f1 > 0:
        return None  # No sign change, no root found

    try:
        # Find zero of objective function within the valid range
        alpha = brentq(
            lambda x: _refraction_objective_conic(x, C_3d, O_3d, S0_3d, r, k, n_outside, n_conic)[0],
            0,
            upper_bound,
        )
        _, I_3d = _refraction_objective_conic(alpha, C_3d, O_3d, S0_3d, r, k, n_outside, n_conic)

        if I_3d is None:
            return None

        # Return result in same coordinate type as input
        if is_homogeneous:
            result = np.array([I_3d[0], I_3d[1], I_3d[2], 1.0])
        else:
            result = I_3d
        return result
    except (ValueError, RuntimeError):
        return None


def refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere):
    """Refracts ray at surface of sphere.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    [U0, Ud] = refract_ray_sphere(R0, Rd, S0, Sr, n_outside, n_sphere) finds
    the point 'U0' at which a ray (specified by its origin 'R0' and direction
    'Rd') strikes a sphere (with center 'S0' and radius 'Sr') and computes
    the direction 'Ud' of the refracted ray. The refractive index outside
    the sphere is 'n_outside', the refractive index of the sphere is
    'n_sphere'.

    [] is returned for 'U0' and 'Ud' if the original ray does not intersect
    the sphere.

    3D or homogeneous coordinates may be passed in; the same type of
    coordinates is passed out.

    Args:
        R0: Ray origin (3D or 4D homogeneous)
        Rd: Ray direction (3D or 4D homogeneous)
        S0: Sphere center (3D or 4D homogeneous)
        Sr: Sphere radius
        n_outside: Refractive index outside sphere
        n_sphere: Refractive index of sphere

    Returns:
        Tuple of (U0, Ud) where U0 is intersection point, Ud is refracted direction.
        Returns (None, None) if no intersection.
    """
    # Extract 3D spatial components for calculations
    # R0_3d = R0[:3] if len(R0) > 3 else R0
    Rd_3d = Rd[:3] if len(Rd) > 3 else Rd
    S0_3d = S0[:3] if len(S0) > 3 else S0

    # Determine output coordinate type from ray inputs
    is_homogeneous = len(R0) > 3 or len(Rd) > 3

    # Normalize ray direction (using 3D spatial components)
    Rd_normalized = Rd_3d / np.linalg.norm(Rd_3d)

    # Find point of intersection
    U0, _ = intersect_ray_sphere(R0, Rd, S0, Sr)

    if U0 is None:
        return None, None

    # Extract 3D components from intersection result for calculations
    U0_3d = U0[:3] if len(U0) > 3 else U0

    # Find surface normal at point of intersection (pointing inwards)
    N = (S0_3d - U0_3d) / np.linalg.norm(S0_3d - U0_3d)

    # Find cosines
    costh1 = np.dot(Rd_normalized, N)
    costh2_squared = 1 - (n_outside / n_sphere) ** 2 * (1 - costh1**2)

    # Check for total internal reflection
    if costh2_squared < 0:
        return U0, None

    costh2 = np.sqrt(costh2_squared)

    # Snell's law refraction formula
    Ud_3d = (n_outside / n_sphere) * Rd_normalized + (costh2 - (n_outside / n_sphere) * costh1) * N

    # Return results in same coordinate type as input
    if is_homogeneous:
        # Format U0 as homogeneous position (w=1) if not already
        if len(U0) == 3:
            U0_out = np.array([U0[0], U0[1], U0[2], 1.0])
        else:
            U0_out = U0
        # Format Ud as homogeneous direction (w=0)
        Ud = np.array([Ud_3d[0], Ud_3d[1], Ud_3d[2], 0.0])
        return U0_out, Ud
    else:
        return U0, Ud_3d


def refract_ray_conic(R0, Rd, S0, r_apical, k, n_outside, n_conic):
    """
    Refract ray at surface of conic section.

    This is the conic equivalent of refract_ray_sphere() used in the eye simulator,
    implementing proper corneal asphericity with k-value.

    Args:
        R0: Ray origin (3D or 4D homogeneous)
        Rd: Ray direction (3D or 4D homogeneous)
        S0: Conic center (3D or 4D homogeneous, typically corneal apex)
        r_apical: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)
        n_outside: Refractive index outside conic (e.g., air = 1.0)
        n_conic: Refractive index of conic (e.g., cornea = 1.376)

    Returns:
        Tuple of (U0, Ud) where:
        - U0: Intersection point on conic surface
        - Ud: Refracted ray direction
        Returns (None, None) if no intersection or total internal reflection.
    """
    # Extract 3D spatial components for calculations
    Rd_3d = Rd[:3] if len(Rd) > 3 else Rd

    # Determine output coordinate type from ray inputs
    is_homogeneous = len(R0) > 3 or len(Rd) > 3

    # Normalize ray direction (using 3D spatial components)
    Rd_normalized = Rd_3d / np.linalg.norm(Rd_3d)

    # Step 1: Find intersection point
    U0, _ = intersect_ray_conic(R0, Rd, S0, r_apical, k)

    if U0 is None:
        return None, None

    # Extract 3D components from intersection result
    U0_3d = U0[:3] if len(U0) > 3 else U0

    # Step 2: Calculate surface normal at intersection point
    N = conic_surface_normal(U0, S0, r_apical, k)

    # For refraction, we need inward-pointing normal (toward conic interior)
    # Check if normal points outward and flip if needed
    S0_3d = S0[:3] if len(S0) > 3 else S0
    center_to_point = U0_3d - S0_3d
    if np.dot(N, center_to_point) > 0:  # Normal points outward
        N = -N  # Flip to point inward

    # Step 3: Apply Snell's law
    costh1 = np.dot(Rd_normalized, N)
    costh2_squared = 1 - (n_outside / n_conic) ** 2 * (1 - costh1**2)

    # Check for total internal reflection
    if costh2_squared < 0:
        return U0, None

    costh2 = np.sqrt(costh2_squared)

    # Snell's law refraction formula
    Ud_3d = (n_outside / n_conic) * Rd_normalized + (costh2 - (n_outside / n_conic) * costh1) * N

    # Step 4: Return results in same coordinate type as input
    if is_homogeneous:
        # Format U0 as homogeneous position (w=1) if not already
        if len(U0) == 3:
            U0_out = np.array([U0[0], U0[1], U0[2], 1.0])
        else:
            U0_out = U0
        # Format Ud as homogeneous direction (w=0)
        Ud = np.array([Ud_3d[0], Ud_3d[1], Ud_3d[2], 0.0])
        return U0_out, Ud
    else:
        return U0, Ud_3d
