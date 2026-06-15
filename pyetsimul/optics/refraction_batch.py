"""Corneal refraction: where intraocular points appear to a camera through the cornea.

Given a camera, object points behind the cornea, and the corneal surface geometry, each function finds the
point on the cornea where an object's ray bends toward the camera under Snell's law. There is no closed
form, so the refraction direction (single surface) or launch direction (two surfaces) is found numerically.
All N object points are solved together as numpy arrays -- a bisection for the in-plane Snell root and a
Newton step for the out-of-plane (aspheric) and two-surface corrections -- so the whole pupil boundary
refracts in one pass.

Single-surface variants refract once (air <-> cornea); dual-surface variants refract at both the posterior
(aqueous -> cornea) and anterior (cornea -> air) surfaces. Both conic and spherical surface shapes are
supported.

Inputs are plain numpy: ``camera`` is ``(3,)``, ``objects`` is ``(N, 3)``, surface centres are ``(3,)``.
Each public function returns ``(points (N, 3), valid (N,))``; rows with no real refraction path hold NaN.
"""

import numpy as np

# A surface is (center (3,), radius, conic constant) -- the conic constant is None for a sphere.
_Surface = tuple[np.ndarray, float, float | None]
# Two-surface stack: posterior, anterior, then the three refractive indices (aqueous, cornea, outside).
_DualSurfaces = tuple[_Surface, _Surface, float, float, float]

_EPS_T = 1e-12  # smallest ray parameter accepted as a non-trivial surface crossing
_FD = 1e-7  # finite-difference step for the Newton Jacobians


def _unit(v: np.ndarray) -> np.ndarray:
    """Row-wise unit vectors of an ``(N, 3)`` array (zero rows left unchanged)."""
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.where(n > 0.0, n, 1.0)


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise 3-vector cross product (manual, avoids np.cross axis overhead)."""
    return np.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


# ---------------------------------------------------------------------------
# Vectorised surface primitives
# ---------------------------------------------------------------------------


def _conic_quadratic(
    origins: np.ndarray, dirs: np.ndarray, center: np.ndarray, radius: float, k: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Coefficients (a, b, c) and discriminant of the ray-conic intersection quadratic for ``(N, 3)`` rays.

    The rotationally symmetric conic with apex curvature ``radius`` (R) and conic constant ``k`` is
    ``x^2 + y^2 + (1 + k) z^2 - 2 R z = 0`` about an axis through ``center`` along +z (its centre of
    curvature sits at z = center_z - R / (1 + k)). Substituting a ray ``origin + t dir`` gives
    ``a t^2 + b t + c = 0``; the discriminant ``b^2 - 4 a c`` is non-negative where the ray meets it.
    """
    cz = center[2] - radius / (1.0 + k)
    ox, oy, oz = origins[:, 0] - center[0], origins[:, 1] - center[1], origins[:, 2] - cz
    dx, dy, dz = dirs[:, 0], dirs[:, 1], dirs[:, 2]
    a = dx * dx + dy * dy + (1.0 + k) * dz * dz
    b = 2.0 * (ox * dx + oy * dy + (1.0 + k) * oz * dz - radius * dz)
    c = ox * ox + oy * oy + (1.0 + k) * oz * oz - 2.0 * radius * oz
    return a, b, c, b * b - 4.0 * a * c


def _intersect_ray_conic(
    origins: np.ndarray, dirs: np.ndarray, center: np.ndarray, radius: float, k: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest non-negative-t hit of each ray with the conic: the first surface crossing. Returns (points, valid)."""
    d = _unit(dirs)
    a, b, _c, disc = _conic_quadratic(origins, d, center, radius, k)
    ok = disc >= 0.0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    tlo = np.minimum(t1, t2)
    thi = np.maximum(t1, t2)
    t = np.where(tlo >= 0.0, tlo, thi)  # smaller root if it is ahead of the origin, else the far one
    valid = ok & (thi >= 0.0)
    return origins + t[:, None] * d, valid


def _point_on_conic(center: np.ndarray, dirs: np.ndarray, radius: float, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Conic surface point reached from ``center`` along each direction in ``dirs``. Returns (points, valid).

    A ray from the conic centre crosses the surface twice; the point on the +dirs side is the larger root.
    """
    d = _unit(dirs)
    origins = np.broadcast_to(center, d.shape)
    a, b, _c, disc = _conic_quadratic(origins, d, center, radius, k)
    ok = disc >= 0.0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t1 = (-b + sq) / (2.0 * a)
    t2 = (-b - sq) / (2.0 * a)
    big, small = np.maximum(t1, t2), np.minimum(t1, t2)
    use_big = np.abs(big) > _EPS_T  # discard the trivial root at the centre itself
    t = np.where(use_big, big, small)
    valid = ok & (use_big | (np.abs(small) > _EPS_T))
    return center + t[:, None] * d, valid


def _conic_normal(points: np.ndarray, center: np.ndarray, radius: float, k: float) -> np.ndarray:
    """Outward unit normals at ``(N, 3)`` conic-surface points.

    The normal is the gradient of the conic's implicit function ``f = x^2 + y^2 + (1 + k) z^2 - 2 R z``,
    i.e. ``(2 x, 2 y, 2 (1 + k) z - 2 R)`` in surface-local coordinates. At the apex the gradient vanishes,
    so a degenerate normal falls back to the axis direction +z.
    """
    cz = center[2] - radius / (1.0 + k)
    nx = 2.0 * (points[:, 0] - center[0])
    ny = 2.0 * (points[:, 1] - center[1])
    nz = 2.0 * (1.0 + k) * (points[:, 2] - cz) - 2.0 * radius
    normal = np.stack([nx, ny, nz], axis=-1)
    mag = np.linalg.norm(normal, axis=-1, keepdims=True)
    degenerate = mag[:, 0] < 1e-15
    normal /= np.where(mag > 0.0, mag, 1.0)
    normal[degenerate] = np.array([0.0, 0.0, 1.0])  # apex fallback
    return normal


def _intersect_ray_sphere(
    origins: np.ndarray, dirs: np.ndarray, center: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray]:
    """Nearest non-negative-t hit of each ray with the sphere. Returns (points, valid)."""
    d = _unit(dirs)
    oc = origins - center
    b = 2.0 * np.einsum("ij,ij->i", oc, d)
    c = np.einsum("ij,ij->i", oc, oc) - radius * radius
    disc = b * b - 4.0 * c  # a == 1 for a unit direction
    ok = disc >= 0.0
    sq = np.sqrt(np.where(ok, disc, 0.0))
    t1 = (-b - sq) / 2.0
    t2 = (-b + sq) / 2.0
    tlo, thi = np.minimum(t1, t2), np.maximum(t1, t2)
    t = np.where(tlo >= 0.0, tlo, thi)
    valid = ok & (thi >= 0.0)
    return origins + t[:, None] * d, valid


def _sphere_normal(points: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Outward unit normals at ``(N, 3)`` sphere-surface points: the radius direction from the centre."""
    return _unit(points - center)


def _refract_direction(
    incident: np.ndarray, normal: np.ndarray, n_from: float, n_to: float
) -> tuple[np.ndarray, np.ndarray]:
    """Snell's law in vector form for ``(N, 3)`` rays. Returns (refracted unit dirs, valid).

    With ``eta = n_from / n_to`` the transmitted direction is
    ``eta i + (eta cos_i - cos_t) n``, where ``cos_t = sqrt(1 - eta^2 (1 - cos_i^2))``. The normal is first
    oriented to face the incident medium so the formula holds whether the ray enters or leaves the surface;
    a negative ``cos_t^2`` is total internal reflection and marks the row invalid.
    """
    n = normal.copy()
    cos_i = -np.einsum("ij,ij->i", incident, n)
    flip = cos_i < 0.0  # orient the normal to face the incident medium
    n[flip] = -n[flip]
    cos_i = np.abs(cos_i)
    eta = n_from / n_to
    cos_t_sq = 1.0 - eta * eta * (1.0 - cos_i * cos_i)
    valid = cos_t_sq >= 0.0  # else total internal reflection
    refracted = eta * incident + (eta * cos_i - np.sqrt(np.where(valid, cos_t_sq, 0.0)))[:, None] * n
    return _unit(refracted), valid


# ---------------------------------------------------------------------------
# Two-surface (dual) refraction: vectorised launch-direction shooting
# ---------------------------------------------------------------------------


def _trace_dual(
    launch: np.ndarray, objects: np.ndarray, surfaces: _DualSurfaces, conic: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Trace ``(N, 3)`` launch directions from ``objects`` out through both corneal surfaces.

    Each ray refracts first at the posterior surface (aqueous -> cornea) and then at the anterior surface
    (cornea -> air). Returns (anterior exit points, exit directions, valid). ``conic`` selects conic vs
    spherical primitives.
    """
    post, ant, n_aq, n_cor, n_out = surfaces
    d = _unit(launch)
    if conic:
        pp, v1 = _intersect_ray_conic(objects, d, *post)
        pn = _conic_normal(pp, *post)
    else:
        pp, v1 = _intersect_ray_sphere(objects, d, post[0], post[1])
        pn = _sphere_normal(pp, post[0])
    cdir, v2 = _refract_direction(d, pn, n_aq, n_cor)
    if conic:
        ap, v3 = _intersect_ray_conic(pp, cdir, *ant)
        an = _conic_normal(ap, *ant)
    else:
        ap, v3 = _intersect_ray_sphere(pp, cdir, ant[0], ant[1])
        an = _sphere_normal(ap, ant[0])
    edir, v4 = _refract_direction(cdir, an, n_cor, n_out)
    return ap, edir, v1 & v2 & v3 & v4


def _solve_two_surface_launch(
    camera: np.ndarray, objects: np.ndarray, surfaces: _DualSurfaces, conic: bool, iters: int = 40, tol: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """Find, for every object, the launch direction whose doubly refracted ray reaches the camera.

    Two surfaces have no closed-form refraction point, so the launch direction is searched. It is
    parameterised by a 2D offset in the plane around the straight object->camera line (basis perp_a,
    perp_b); damped Newton with a finite-difference Jacobian drives the exit ray's perpendicular miss to
    the camera to zero. Returns (anterior exit points, valid); unconverged rows hold NaN.
    """
    n = objects.shape[0]
    base = _unit(camera[None, :] - objects)
    up = np.broadcast_to(np.array([0.0, 0.0, 1.0]), base.shape)
    perp_a = _cross(base, up)
    deg = np.linalg.norm(perp_a, axis=-1) < 1e-9  # base parallel to +z: pick another reference axis
    if deg.any():
        perp_a[deg] = _cross(base[deg], np.array([0.0, 1.0, 0.0]))
    perp_a = _unit(perp_a)
    perp_b = _cross(base, perp_a)

    def trace(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        launch = base + offset[:, 0:1] * perp_a + offset[:, 1:2] * perp_b
        return _trace_dual(launch, objects, surfaces, conic)

    def residual(offset: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Two components of the exit ray's closest-approach miss to the camera, in a frame fixed to the ray.
        ap, edir, ok = trace(offset)
        rel = camera[None, :] - ap
        miss = rel - np.einsum("ij,ij->i", rel, edir)[:, None] * edir
        across = _unit(_cross(edir, perp_a))
        r = np.stack([np.einsum("ij,ij->i", miss, across), np.einsum("ij,ij->i", miss, _cross(edir, across))], -1)
        r = np.where(ok[:, None], r, 1e3)  # a ray that fails to get through is pushed far from the root
        return r, ok

    offset = np.zeros((n, 2))
    active = np.ones(n, dtype=bool)
    for _ in range(iters):
        r, _ok = residual(offset)
        done = np.linalg.norm(r, axis=-1) < tol
        active &= ~done
        if not active.any():
            break
        # Finite-difference Jacobian J[:, :, j] = d residual / d offset_j, then a 2x2 solve per object.
        jac = np.empty((n, 2, 2))
        for j in range(2):
            step = np.zeros((n, 2))
            step[:, j] = _FD
            jac[:, :, j] = (residual(offset + step)[0] - r) / _FD
        det = jac[:, 0, 0] * jac[:, 1, 1] - jac[:, 0, 1] * jac[:, 1, 0]
        solvable = active & (np.abs(det) > 1e-30)
        safe_det = np.where(solvable, det, 1.0)  # avoid 0/0 on singular rows; their step stays zero
        delta = np.zeros((n, 2))
        inv00, inv11 = jac[:, 1, 1], jac[:, 0, 0]
        inv01, inv10 = -jac[:, 0, 1], -jac[:, 1, 0]
        d0 = (inv00 * r[:, 0] + inv01 * r[:, 1]) / safe_det
        d1 = (inv10 * r[:, 0] + inv11 * r[:, 1]) / safe_det
        delta[solvable, 0] = -d0[solvable]
        delta[solvable, 1] = -d1[solvable]
        offset += delta

    ap, _edir, ok = trace(offset)
    r, _ = residual(offset)
    valid = ok & (np.linalg.norm(r, axis=-1) < 1e-6)
    ap = np.where(valid[:, None], ap, np.nan)
    return ap, valid


def find_refraction_dual_conic_batch(
    camera: np.ndarray,
    objects: np.ndarray,
    ant_center: np.ndarray,
    ant_radius: float,
    ant_k: float,
    post_center: np.ndarray,
    post_radius: float,
    post_k: float,
    n_outside: float,
    n_cornea: float,
    n_aqueous: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Anterior exit points of the two-surface conic refraction for ``(N, 3)`` object points."""
    surfaces = ((post_center, post_radius, post_k), (ant_center, ant_radius, ant_k), n_aqueous, n_cornea, n_outside)
    return _solve_two_surface_launch(camera, objects, surfaces, conic=True)


def find_refraction_dual_sphere_batch(
    camera: np.ndarray,
    objects: np.ndarray,
    ant_center: np.ndarray,
    ant_radius: float,
    post_center: np.ndarray,
    post_radius: float,
    n_outside: float,
    n_cornea: float,
    n_aqueous: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Anterior exit points of the two-surface spherical refraction for ``(N, 3)`` object points."""
    surfaces = ((post_center, post_radius, None), (ant_center, ant_radius, None), n_aqueous, n_cornea, n_outside)
    return _solve_two_surface_launch(camera, objects, surfaces, conic=False)


# ---------------------------------------------------------------------------
# Single-surface refraction
# ---------------------------------------------------------------------------


def _snell_residual_conic(
    direction: np.ndarray,
    camera: np.ndarray,
    objects: np.ndarray,
    center: np.ndarray,
    radius: float,
    k: float,
    n_out: float,
    n_conic: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Snell residual ``n_out sin θ_cam - n_conic sin θ_obj`` for candidate refraction directions.

    Each direction points from the conic centre to a trial surface point; the residual is zero when the
    incidence and transmission angles at that point satisfy Snell's law for the object-to-camera path.
    Also returns the surface point, normal, and the two ray directions for the caller's Newton step.
    """
    point, ok = _point_on_conic(center, direction, radius, k)
    normal = _conic_normal(point, center, radius, k)
    d_obj = _unit(point - objects)
    d_cam = _unit(camera[None, :] - point)
    cos_cam = np.einsum("ij,ij->i", normal, d_cam)
    cos_obj = -np.einsum("ij,ij->i", normal, d_obj)
    sin_cam = np.sqrt(np.maximum(0.0, 1.0 - cos_cam * cos_cam))
    sin_obj = np.sqrt(np.maximum(0.0, 1.0 - cos_obj * cos_obj))
    return n_out * sin_cam - n_conic * sin_obj, point, normal, d_obj, d_cam, ok


def find_refraction_conic_batch(
    camera: np.ndarray,
    objects: np.ndarray,
    center: np.ndarray,
    radius: float,
    k: float,
    n_outside: float,
    n_conic: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Refraction points on a single conic surface for ``(N, 3)`` object points.

    Two stages. Stage 1 finds the in-plane solution: the refraction direction is interpolated between the
    to-object and to-camera directions by ``alpha in [0, 1]``, and the Snell residual is monotonic and
    sign-changing in alpha, so a bracketed bisection finds the unique root directly on the conic (no
    spherical approximation). Stage 2 handles the asphere: when ``k != 0`` the refraction point leaves the
    camera-centre-object plane slightly, so a 2D Newton in ``(alpha, beta)`` adds an out-of-plane offset
    ``beta`` along ``perp``, driving both the Snell residual and the coplanarity residual
    ``normal . (d_obj x d_cam)`` to zero.
    """
    to_cam = _unit((camera - center)[None, :].repeat(objects.shape[0], 0))
    to_obj = _unit(objects - center)

    def direction(alpha: np.ndarray, beta: np.ndarray, perp: np.ndarray | float) -> np.ndarray:
        return to_cam * alpha[:, None] + to_obj * (1.0 - alpha)[:, None] + perp * beta[:, None]

    def snell(alpha: np.ndarray) -> np.ndarray:
        return _snell_residual_conic(
            direction(alpha, np.zeros_like(alpha), 0.0), camera, objects, center, radius, k, n_outside, n_conic
        )[0]

    # Stage 1: in-plane Snell root in alpha in [0, 1] by vectorised bisection.
    lo = np.zeros(objects.shape[0])
    hi = np.ones(objects.shape[0])
    f_lo = snell(lo)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = snell(mid)
        left = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(left, mid, lo)
        f_lo = np.where(left, f_mid, f_lo)
        hi = np.where(left, hi, mid)
    alpha = 0.5 * (lo + hi)

    # Stage 2: 2D Newton in (alpha, beta) for the out-of-plane correction. perp spans the direction normal
    # to the camera-centre-object plane (an arbitrary perpendicular when the two directions are collinear).
    perp = _cross(to_cam, to_obj)
    pn = np.linalg.norm(perp, axis=-1)
    bad = pn < 1e-15
    if bad.any():
        arb = np.where((np.abs(to_cam[:, 0]) < 0.9)[:, None], np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        perp[bad] = _cross(to_cam[bad], arb[bad])
    perp = _unit(perp)
    beta = np.zeros(objects.shape[0])

    def residual2(alpha: np.ndarray, beta: np.ndarray) -> np.ndarray:
        snell_r, _p, normal, d_obj, d_cam, _ok = _snell_residual_conic(
            direction(alpha, beta, perp), camera, objects, center, radius, k, n_outside, n_conic
        )
        coplanar = np.einsum("ij,ij->i", normal, _cross(d_obj, d_cam))
        return np.stack([snell_r, coplanar], -1)

    for _ in range(20):
        r = residual2(alpha, beta)
        if np.max(np.abs(r)) < 1e-12:
            break
        r_a = (residual2(alpha + _FD, beta) - r) / _FD
        r_b = (residual2(alpha, beta + _FD) - r) / _FD
        det = r_a[:, 0] * r_b[:, 1] - r_b[:, 0] * r_a[:, 1]
        ok = np.abs(det) > 1e-30
        d_alpha = np.where(ok, -(r_b[:, 1] * r[:, 0] - r_b[:, 0] * r[:, 1]) / det, 0.0)
        d_beta = np.where(ok, -(-r_a[:, 1] * r[:, 0] + r_a[:, 0] * r[:, 1]) / det, 0.0)
        alpha += d_alpha
        beta += d_beta

    point, ok = _point_on_conic(center, direction(alpha, beta, perp), radius, k)
    valid = ok & (np.max(np.abs(residual2(alpha, beta)), axis=-1) < 1e-6)
    return np.where(valid[:, None], point, np.nan), valid


def find_refraction_sphere_batch(
    camera: np.ndarray, objects: np.ndarray, center: np.ndarray, radius: float, n_outside: float, n_sphere: float
) -> tuple[np.ndarray, np.ndarray]:
    """Refraction points on a single sphere surface for ``(N, 3)`` object points.

    On a sphere the refraction point stays in the camera-centre-object plane, so a single 1D solve
    suffices: ``alpha in [0, 1]`` interpolates the surface normal between the to-object and to-camera
    directions, and the Snell residual is monotonic and sign-changing in alpha, so a bracketed bisection
    finds the unique root.
    """
    to_cam = _unit((camera - center)[None, :].repeat(objects.shape[0], 0))
    to_obj = _unit(objects - center)

    def snell(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        normal = _unit(to_cam * a[:, None] + to_obj * (1.0 - a)[:, None])
        surface = center + radius * normal
        c2s = _unit(camera[None, :] - surface)
        s2o = _unit(surface - objects)
        cos_c = np.einsum("ij,ij->i", normal, c2s)
        cos_o = np.einsum("ij,ij->i", normal, s2o)
        sin_c = np.sqrt(np.maximum(0.0, 1.0 - cos_c * cos_c))
        sin_o = np.sqrt(np.maximum(0.0, 1.0 - cos_o * cos_o))
        return n_outside * sin_c - n_sphere * sin_o, surface

    lo = np.zeros(objects.shape[0])
    hi = np.ones(objects.shape[0])
    f_lo = snell(lo)[0]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_mid = snell(mid)[0]
        left = np.sign(f_mid) == np.sign(f_lo)
        lo = np.where(left, mid, lo)
        f_lo = np.where(left, f_mid, f_lo)
        hi = np.where(left, hi, mid)
    a = 0.5 * (lo + hi)
    f, surface = snell(a)
    valid = np.abs(f) < 1e-6
    return np.where(valid[:, None], surface, np.nan), valid
