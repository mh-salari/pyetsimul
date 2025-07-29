import numpy as np
from scipy.optimize import brentq
import warnings


def intersect_ray_sphere(R0, Rd, S0, Sr):
    """Finds intersection between ray and sphere.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    [pos, pos2] = intersect_ray_sphere(R0, Rd, S0, Sr) finds the intersection
    between a ray (specified by its origin 'R0' and direction 'Rd') and a
    sphere (center 'S0' and radius 'Sr'). The intersection that is closer to
    'R0' is returned in 'pos', and the other intersection is returned in
    'pos2'. [] is returned for 'pos' and 'pos2' if the ray does not intersect
    the sphere.

    3D or homogeneous coordinates may be passed in; 3D coordinates
    are returned.

    Args:
        R0: Ray origin (3D or homogeneous coordinates)
        Rd: Ray direction (3D or homogeneous coordinates)
        S0: Sphere center (3D or homogeneous coordinates)
        Sr: Sphere radius

    Returns:
        Tuple of (pos, pos2) where pos is closer intersection, pos2 is farther.
        Both are 3D coordinates. Returns (None, None) if no intersection.
    """

    # Extract 3D components for calculations
    R0_3d = R0[:3] if len(R0) == 4 else R0
    Rd_3d = Rd[:3] if len(Rd) == 4 else Rd
    S0_3d = S0[:3] if len(S0) == 4 else S0

    # Normalize direction of ray
    Rd_3d = Rd_3d / np.linalg.norm(Rd_3d)

    # Quadratic equation coefficients
    b = 2 * (Rd_3d[0] * (R0_3d[0] - S0_3d[0]) + Rd_3d[1] * (R0_3d[1] - S0_3d[1]) + Rd_3d[2] * (R0_3d[2] - S0_3d[2]))
    c = (R0_3d[0] - S0_3d[0]) ** 2 + (R0_3d[1] - S0_3d[1]) ** 2 + (R0_3d[2] - S0_3d[2]) ** 2 - Sr**2

    # discriminant
    disc = b**2 - 4 * c

    if disc < 0:
        return None, None
    else:
        # Two solutions
        t1 = (-b + np.sqrt(disc)) / 2
        t2 = (-b - np.sqrt(disc)) / 2

        # Choose closer intersection first
        if abs(t1) < abs(t2):
            t = t1
            t_ = t2
        else:
            t = t2
            t_ = t1

        # Compute intersection points - always return 3D coordinates
        pos = R0_3d + t * Rd_3d
        pos2 = R0_3d + t_ * Rd_3d

        return pos, pos2


def intersect_ray_circle(R0, Rd, C0, Cr):
    """Finds intersection between ray and circle.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    pos = intersect_ray_circle(R0, Rd, C0, Cr) finds the intersection (in
    2-D) between a ray (specified by its origin 'R0' and direction 'Rd') and
    a circle (center C0 and radius Cr). The intersection that is closest to
    R0 is returned. [] is returned if the ray does not intersect the circle.

    Args:
        R0: Ray origin (2D)
        Rd: Ray direction (2D)
        C0: Circle center (2D)
        Cr: Circle radius

    Returns:
        Intersection point closest to ray origin, or None if no intersection
    """
    # Normalize direction of ray
    Rd = Rd / np.linalg.norm(Rd)

    # Quadratic equation coefficients
    b = 2 * (Rd[0] * (R0[0] - C0[0]) + Rd[1] * (R0[1] - C0[1]))
    c = (R0[0] - C0[0]) ** 2 + (R0[1] - C0[1]) ** 2 - Cr**2

    # Discriminant
    disc = b**2 - 4 * c

    if disc < 0:
        return None
    else:
        # Two solutions
        t1 = (-b + np.sqrt(disc)) / 2
        t2 = (-b - np.sqrt(disc)) / 2

        # Choose closest intersection
        if abs(t1) < abs(t2):
            t = t1
        else:
            t = t2

        pos = R0 + t * Rd

        return pos


def intersect_ray_plane(R0, Rd, P0, Pn):
    """Finds intersection between ray and plane.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    pos = intersect_ray_plane(R0, Rd, P0, Pn) finds the intersection
    between a ray (specified by its origin 'R0' and direction 'Rd') and
    a plane (specified by point 'P0' on the plane and normal vector 'Pn').
    Returns None if the ray is parallel to the plane.

    The function solves the parametric ray equation:
        x = R0 + t * Rd
    with the plane equation:
        (x - P0) · Pn = 0

    Args:
        R0: Ray origin (3D coordinates)
        Rd: Ray direction (3D coordinates)
        P0: Point on plane (3D coordinates)
        Pn: Plane normal vector (3D coordinates)

    Returns:
        Intersection point (3D coordinates), or None if ray is parallel to plane
    """
    # Normalize plane normal
    Pn = Pn / np.linalg.norm(Pn)

    # Check if ray is parallel to plane
    denom = np.dot(Rd, Pn)

    if abs(denom) < 1e-15:  # Ray is parallel to plane
        return None

    # Solve for parameter t: (R0 + t*Rd - P0) · Pn = 0
    # t = (P0 - R0) · Pn / (Rd · Pn)
    t = np.dot(P0 - R0, Pn) / denom

    # Calculate intersection point
    intersection = R0 + t * Rd

    return intersection


def intersect_ray_spheroid(R0, Rd, S0, a, b, c):
    """
    Find intersection between ray and prolate spheroid.

    For prolate spheroid: a = b (equal horizontal/vertical radii), c ≠ a (different depth)
    Spheroid equation: (x-S0x)²/a² + (y-S0y)²/b² + (z-S0z)²/c² = 1

    Args:
        R0: Ray origin (3D coordinates)
        Rd: Ray direction (3D coordinates)
        S0: Spheroid center (3D coordinates)
        a: Semi-axis length in X direction (horizontal)
        b: Semi-axis length in Y direction (vertical)
        c: Semi-axis length in Z direction (anterior-posterior)

    Returns:
        Tuple of (pos, pos2) where pos is closer intersection, pos2 is farther.
        Returns (None, None) if no intersection.
    """
    # Extract 3D components
    R0_3d = R0[:3] if len(R0) == 4 else R0
    Rd_3d = Rd[:3] if len(Rd) == 4 else Rd
    S0_3d = S0[:3] if len(S0) == 4 else S0

    # Normalize ray direction
    Rd_3d = Rd_3d / np.linalg.norm(Rd_3d)

    # Translate coordinate system so spheroid center is at origin
    R0_translated = R0_3d - S0_3d

    # Ray equation: P(t) = R0_translated + t * Rd_3d
    # Spheroid equation: x²/a² + y²/b² + z²/c² = 1

    def spheroid_equation(t):
        """Evaluate spheroid equation at parameter t along ray."""
        point = R0_translated + t * Rd_3d
        x, y, z = point[0], point[1], point[2]
        return (x**2 / a**2) + (y**2 / b**2) + (z**2 / c**2) - 1.0

    # Find intersections by searching for zeros
    # Use reasonable search ranges based on spheroid dimensions
    t_candidates = []

    # Estimate reasonable search bound based on spheroid size and ray origin distance
    max_axis = max(a, b, c)
    ray_origin_dist = np.linalg.norm(R0_translated)
    search_bound = ray_origin_dist + 2 * max_axis  # Conservative bound

    # Use fine-grained search to ensure we find roots
    n_segments = 100
    t_values = np.linspace(-search_bound, search_bound, n_segments + 1)

    for i in range(n_segments):
        t_min, t_max = t_values[i], t_values[i + 1]

        try:
            # Check if there's a sign change in this interval
            f_min = spheroid_equation(t_min)
            f_max = spheroid_equation(t_max)

            if f_min * f_max < 0:  # Sign change indicates root
                # Use Brent's method for robust root finding
                t_root = brentq(spheroid_equation, t_min, t_max, xtol=1e-12)

                # Verify this is actually a root
                if abs(spheroid_equation(t_root)) < 1e-10:
                    # Check if we already have this root (within tolerance)
                    is_duplicate = any(abs(t_root - t_existing) < 1e-8 for t_existing in t_candidates)
                    if not is_duplicate:
                        t_candidates.append(t_root)

        except (ValueError, RuntimeError):
            # Root finding failed in this interval
            continue

    # Sort candidates by distance from ray origin
    t_candidates.sort(key=abs)

    if len(t_candidates) == 0:
        return None, None
    elif len(t_candidates) == 1:
        # Only one intersection (tangent case)
        t = t_candidates[0]
        pos = R0_3d + t * Rd_3d
        return pos, None
    else:
        # Two or more intersections - return closest two
        t1, t2 = t_candidates[0], t_candidates[1]
        pos1 = R0_3d + t1 * Rd_3d
        pos2 = R0_3d + t2 * Rd_3d
        return pos1, pos2


def spheroid_surface_normal(point, S0, a, b, c):
    """
    Calculate surface normal at a point on prolate spheroid.

    For spheroid equation: (x-S0x)²/a² + (y-S0y)²/b² + (z-S0z)²/c² = 1
    Normal vector is gradient: ∇f = (2(x-S0x)/a², 2(y-S0y)/b², 2(z-S0z)/c²)

    Args:
        point: Point on spheroid surface (3D coordinates)
        S0: Spheroid center (3D coordinates)
        a, b, c: Semi-axis lengths

    Returns:
        Unit normal vector pointing outward from spheroid surface
    """
    # Extract 3D components
    point_3d = point[:3] if len(point) == 4 else point
    S0_3d = S0[:3] if len(S0) == 4 else S0

    # Translate to spheroid-centered coordinates
    x, y, z = point_3d - S0_3d

    # Calculate gradient (surface normal before normalization)
    normal = np.array([2 * x / (a**2), 2 * y / (b**2), 2 * z / (c**2)])

    # Normalize to unit vector
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude < 1e-15:
        warnings.warn("Degenerate normal vector at spheroid center")
        return np.array([0, 0, 1])  # Default to z-axis

    return normal / normal_magnitude
