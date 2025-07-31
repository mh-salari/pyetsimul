import numpy as np
import warnings


def intersect_ray_sphere(R0, Rd, S0, Sr):
    """Finds intersection between ray and sphere.

    [pos, pos2] = intersect_ray_sphere(R0, Rd, S0, Sr) finds the intersection
    between a ray (specified by its origin 'R0' and direction 'Rd') and a
    sphere (center 'S0' and radius 'Sr'). The intersection that is closer to
    'R0' is returned in 'pos', and the other intersection is returned in
    'pos2'. [] is returned for 'pos' and 'pos2' if the ray does not intersect
    the sphere.

    Args:
        R0: Ray origin (4D homogeneous)
        Rd: Ray direction (4D homogeneous)
        S0: Sphere center (4D homogeneous)
        Sr: Sphere radius

    Returns:
        Tuple of (pos, pos2) where pos is closer intersection, pos2 is farther.
        Both are 4D homogeneous coordinates. Returns (None, None) if no intersection.
    """
    # Normalize ray direction using spatial component
    Rd_normalized = Rd[:3] / np.linalg.norm(Rd[:3])

    # Work with 4D homogeneous coordinates for vector operations
    R0_to_S0 = R0 - S0
    
    # Quadratic equation coefficients using spatial components
    b = 2 * np.dot(Rd_normalized, R0_to_S0[:3])
    c = np.dot(R0_to_S0[:3], R0_to_S0[:3]) - Sr**2

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

        # Compute intersection points directly in 4D homogeneous space
        pos = R0.copy()
        pos[:3] = R0[:3] + t * Rd_normalized
        
        pos2 = R0.copy()
        pos2[:3] = R0[:3] + t_ * Rd_normalized
        return pos, pos2


def intersect_ray_circle(R0, Rd, C0, Cr):
    """Finds intersection between ray and circle.



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

    pos = intersect_ray_plane(R0, Rd, P0, Pn) finds the intersection
    between a ray (specified by its origin 'R0' and direction 'Rd') and
    a plane (specified by point 'P0' on the plane and normal vector 'Pn').
    Returns None if the ray is parallel to the plane.

    The function solves the parametric ray equation:
        x = R0 + t * Rd
    with the plane equation:
        (x - P0) · Pn = 0

    Args:
        R0: Ray origin (4D homogeneous)
        Rd: Ray direction (4D homogeneous)
        P0: Point on plane (4D homogeneous)
        Pn: Plane normal vector (4D homogeneous)

    Returns:
        Intersection point (4D homogeneous), or None if ray is parallel to plane
    """
    # Normalize plane normal using spatial component
    Pn_normalized = Pn[:3] / np.linalg.norm(Pn[:3])

    # Check if ray is parallel to plane
    denom = np.dot(Rd[:3], Pn_normalized)

    if abs(denom) < 1e-15:  # Ray is parallel to plane
        return None

    # Work with 4D homogeneous vectors
    P0_to_R0 = P0 - R0
    
    # Solve for parameter t: (R0 + t*Rd - P0) · Pn = 0
    # t = (P0 - R0) · Pn / (Rd · Pn)
    t = np.dot(P0_to_R0[:3], Pn_normalized) / denom

    # Calculate intersection point directly in 4D homogeneous space
    intersection = R0.copy()
    intersection[:3] = R0[:3] + t * Rd[:3]
    
    return intersection


def intersect_ray_conic(R0, Rd, S0, r, k):
    """
    Find intersection between ray and conic section.

    Args:
        R0: Ray origin (4D homogeneous)
        Rd: Ray direction (4D homogeneous)
        S0: Conic center (4D homogeneous, typically corneal apex)
        r: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Tuple (pos1, pos2): Intersection points closer and farther from R0 (4D homogeneous).
        Returns (None, None) if no intersection.
    """
    # Normalize ray direction using spatial component
    Rd_normalized = Rd[:3] / np.linalg.norm(Rd[:3])

    # Translate coordinates so conic center is at origin using 4D operations
    R0_rel_vec = R0 - S0
    R0_rel = R0_rel_vec[:3]

    # Conic equation: x^2 + y^2 + (1+k)z^2 - r^2 = 0
    A = Rd_normalized[0] ** 2 + Rd_normalized[1] ** 2 + (1 + k) * Rd_normalized[2] ** 2
    B = 2 * (R0_rel[0] * Rd_normalized[0] + R0_rel[1] * Rd_normalized[1] + (1 + k) * R0_rel[2] * Rd_normalized[2])
    C = R0_rel[0] ** 2 + R0_rel[1] ** 2 + (1 + k) * R0_rel[2] ** 2 - r**2

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
        pos = R0.copy()
        pos[:3] = R0[:3] + ts[0] * Rd_normalized
        return pos, None
    else:
        ts.sort()
        pos1 = R0.copy()
        pos1[:3] = R0[:3] + ts[0] * Rd_normalized
        
        pos2 = R0.copy()
        pos2[:3] = R0[:3] + ts[1] * Rd_normalized
        return pos1, pos2


def conic_surface_normal(point, S0, r, k):
    """
    Calculate surface normal at a point on conic section surface.

    For conic equation: F(x,y,z) = x² + y² + (1+k)z² - r² = 0
    the normal vector is the gradient: ∇F = (2x, 2y, 2(1+k)z)

    Args:
        point: Point on conic surface (4D homogeneous)
        S0: Conic center (4D homogeneous, typically corneal apex)
        r: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Unit normal vector pointing outward from conic surface (3D)
    """
    # Work with 4D homogeneous coordinates, use spatial components for calculations
    point_to_S0 = point - S0
    x, y, z = point_to_S0[:3]

    # Calculate gradient: ∇F = (2x, 2y, 2(1+k)z)
    normal = np.array([2 * x, 2 * y, 2 * (1 + k) * z])

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
    intersects the conic surface.

    Args:
        center: Conic center (4D homogeneous)
        direction: Direction vector from center (3D, not homogeneous)
        r: Radius parameter (R in the formula, meters)
        k: Conic constant (k < 0 for prolate, k = 0 for sphere, k > 0 for oblate)

    Returns:
        Point on conic surface (4D homogeneous), or None if no intersection found
    """
    # Normalize direction using spatial component
    direction_normalized = direction[:3] / np.linalg.norm(direction[:3])

    # Conic equation: x^2 + y^2 + (1+k)z^2 = r^2
    # Ray from center: P(t) = t * direction
    # Substitute: t^2 * (dx^2 + dy^2 + (1+k)dz^2) = r^2
    dx, dy, dz = direction_normalized
    denominator = dx**2 + dy**2 + (1 + k) * dz**2

    if abs(denominator) < 1e-15:
        return None  # Degenerate case

    t = r / np.sqrt(denominator)

    # Calculate point on surface and return in 4D homogeneous coordinates
    result = center.copy()
    result[:3] = center[:3] + t * direction_normalized
    return result
