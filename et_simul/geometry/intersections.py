import numpy as np
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


def intersect_ray_conic(R0, Rd, S0, r, k):
    """
    Find intersection between ray and conic section using correct formula:
    x² + y² + (1+k)z² = R²/(1+k)
    where R = r is the radius parameter and k is the conic constant.

    Args:
        R0: Ray origin (3D coordinates)
        Rd: Ray direction (3D coordinates)
        S0: Conic center (3D coordinates, typically corneal apex)
        r: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Tuple (pos1, pos2): Intersection points closer and farther from R0.
        Returns (None, None) if no intersection.
    """
    # Extract 3D components and normalize
    R0_3d = R0[:3] if len(R0) == 4 else R0
    Rd_3d = Rd[:3] if len(Rd) == 4 else Rd
    S0_3d = S0[:3] if len(S0) == 4 else S0

    Rd_3d = Rd_3d / np.linalg.norm(Rd_3d)

    # Calculate (1+k) value from k-value
    one_plus_k = 1 + k

    # Translate coordinates so conic center is at origin
    R0_rel = R0_3d - S0_3d

    # Ray equation: P(t) = R0_rel + t * Rd_3d
    # Conic equation: x² + y² + (1+k)z² = R²/(1+k)
    # Substitute ray into conic: (R0x + t*Rdx)² + (R0y + t*Rdy)² + (1+k)(R0z + t*Rdz)² = R²/(1+k)

    # Expand and collect coefficients of At² + Bt + C = 0
    # Coefficient of t²
    A = Rd_3d[0] ** 2 + Rd_3d[1] ** 2 + one_plus_k * Rd_3d[2] ** 2

    # Coefficient of t
    B = 2 * (R0_rel[0] * Rd_3d[0] + R0_rel[1] * Rd_3d[1] + one_plus_k * R0_rel[2] * Rd_3d[2])

    # Constant term
    C = R0_rel[0] ** 2 + R0_rel[1] ** 2 + one_plus_k * R0_rel[2] ** 2 - (r**2 / one_plus_k)

    # Solve quadratic equation
    disc = B**2 - 4 * A * C
    if disc < 0:
        return None, None  # No real roots, no intersection

    sqrt_disc = np.sqrt(disc)
    t1 = (-B - sqrt_disc) / (2 * A)
    t2 = (-B + sqrt_disc) / (2 * A)

    # Filter out intersections behind the ray origin (t < 0)
    ts = [t for t in [t1, t2] if t >= 0]

    if len(ts) == 0:
        return None, None
    elif len(ts) == 1:
        pos = R0_3d + ts[0] * Rd_3d
        return pos, None
    else:
        ts.sort()
        pos1 = R0_3d + ts[0] * Rd_3d
        pos2 = R0_3d + ts[1] * Rd_3d
        return pos1, pos2


def conic_surface_normal(point, S0, r, k):
    """
    Calculate surface normal at a point on conic section surface.

    For conic equation: F(x,y,z) = x² + y² + (1+k)z² - R²/(1+k) = 0
    the normal vector is the gradient: ∇F = (2x, 2y, 2(1+k)z)

    Args:
        point: Point on conic surface (3D coordinates)
        S0: Conic center (3D coordinates, typically corneal apex)
        r: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Unit normal vector pointing outward from conic surface
    """
    # Extract 3D components
    point_3d = point[:3] if len(point) == 4 else point
    S0_3d = S0[:3] if len(S0) == 4 else S0

    # Calculate (1+k) value from k-value
    one_plus_k = 1 + k

    # Translate to conic-centered coordinates
    x, y, z = point_3d - S0_3d

    # Calculate gradient: ∇F = (2x, 2y, 2(1+k)z)
    normal = np.array([2 * x, 2 * y, 2 * one_plus_k * z])

    # Normalize to unit vector
    normal_magnitude = np.linalg.norm(normal)
    if normal_magnitude < 1e-15:
        warnings.warn("Degenerate normal vector at conic apex", RuntimeWarning)
        return np.array([0, 0, 1])  # Default to z-axis

    result = normal / normal_magnitude
    return result


def point_on_conic_surface(center, direction, r, k):
    """
    Find point on conic surface given direction from center.

    Given a direction from the conic center, find the point where this direction
    intersects the conic surface using the correct formula: x² + y² + (1+k)z² = R²/(1+k)

    Args:
        center: Conic center (3D coordinates)
        direction: Direction vector from center (3D coordinates, need not be normalized)
        r: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Point on conic surface, or None if no intersection found
    """
    # Extract 3D components and normalize direction
    center_3d = center[:3] if len(center) == 4 else center
    direction_3d = direction[:3] if len(direction) == 4 else direction
    direction_normalized = direction_3d / np.linalg.norm(direction_3d)

    # Calculate (1+k) value from k-value
    one_plus_k = 1 + k

    # Ray from center: P(t) = t * direction (relative to center)
    # Conic equation (relative to center): x² + y² + (1+k)z² = R²/(1+k)
    # Substitute: (t*dx)² + (t*dy)² + (1+k)(t*dz)² = R²/(1+k)
    # Simplify: t²(dx² + dy² + (1+k)*dz²) = R²/(1+k)
    # Solve for t: t = ±R/√((1+k)(dx² + dy² + (1+k)*dz²))

    dx, dy, dz = direction_normalized

    denominator = dx**2 + dy**2 + one_plus_k * dz**2
    if abs(denominator) < 1e-15:
        return None  # Degenerate case

    # For the corneal surface facing -z direction (toward camera), use:
    # t = +R/√((1+k)(dx² + dy² + (1+k)dz²))
    # This ensures the apex is at [0, 0, -R/(1+k)] as required
    t = r / np.sqrt(one_plus_k * denominator)

    # Calculate point on surface (relative to origin) and then translate
    surface_point = t * direction_normalized

    # Translate back to the original coordinate system
    result = center_3d + surface_point
    return result
