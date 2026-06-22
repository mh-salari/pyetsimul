"""The vectorised refraction (refraction_batch) must obey the physics it solves.

Single-surface refraction points must lie on the corneal surface and satisfy Snell's law; the
dual-surface solver must invert a known forward ray-trace (a launch traced through both surfaces,
with the camera placed on the resulting exit ray, is recovered by the solver).
"""

import numpy as np
import numpy.testing as npt

from pyetsimul.core.cornea import ConicCornea, SphericalCornea
from pyetsimul.optics import refraction_batch as rb
from pyetsimul.types import Position3D

N_AIR, N_CORNEA, N_AQ = 1.0, 1.376, 1.336
CAMERA = np.array([5.0, 10.0, -500.0])  # well in front of the cornea, off-axis


def _xyz(p: Position3D) -> np.ndarray:
    return np.array([p.x, p.y, p.z], dtype=float)


def _objects(n: int = 40, seed: int = 0) -> np.ndarray:
    """Random pupil-like points behind the cornea (z = -8.79 mm, radius <= 3 mm)."""
    rng = np.random.default_rng(seed)
    r = 3.0 * np.sqrt(rng.random(n))
    th = 2 * np.pi * rng.random(n)
    return np.column_stack([r * np.cos(th), r * np.sin(th), np.full(n, -8.79)])


def _snell(
    points: np.ndarray, objects: np.ndarray, camera: np.ndarray, normal: np.ndarray, n_inside: float, n_outside: float
) -> tuple[np.ndarray, np.ndarray]:
    """(n_outside * sin θ_camera, n_inside * sin θ_object) at the refraction points -- equal under Snell."""
    d_cam = rb._unit(camera[None, :] - points)
    d_obj = rb._unit(points - objects)
    cos_cam = np.einsum("ij,ij->i", normal, d_cam)
    cos_obj = -np.einsum("ij,ij->i", normal, d_obj)
    sin_cam = np.sqrt(np.maximum(0.0, 1.0 - cos_cam**2))
    sin_obj = np.sqrt(np.maximum(0.0, 1.0 - cos_obj**2))
    return n_outside * sin_cam, n_inside * sin_obj


def test_single_conic_on_surface_and_snell() -> None:
    """Single conic refraction points lie on the conic surface and satisfy Snell's law."""
    c = ConicCornea()
    c.setup_eye_geometry(24.75)
    center, radius, k = _xyz(c.center), c.anterior_radius, c.anterior_k
    objs = _objects()
    pts, valid = rb.find_refraction_conic_batch(CAMERA, objs, center, radius, k, N_AIR, N_CORNEA)
    assert valid.sum() >= 35
    p, o = pts[valid], objs[valid]
    cz = center[2] - radius / (1 + k)
    x, y, z = p[:, 0] - center[0], p[:, 1] - center[1], p[:, 2] - cz
    npt.assert_allclose(x * x + y * y + (1 + k) * z * z - 2 * radius * z, 0.0, atol=1e-6)  # on the conic
    lhs, rhs = _snell(p, o, CAMERA, rb._conic_normal(p, center, radius, k), N_CORNEA, N_AIR)
    npt.assert_allclose(lhs, rhs, atol=1e-6)


def test_single_sphere_on_surface_and_snell() -> None:
    """Single sphere refraction points lie on the sphere surface and satisfy Snell's law."""
    c = SphericalCornea()
    c.setup_eye_geometry(24.75)
    center, radius = _xyz(c.center), c.anterior_radius
    objs = _objects()
    pts, valid = rb.find_refraction_sphere_batch(CAMERA, objs, center, radius, N_AIR, N_CORNEA)
    assert valid.sum() >= 35
    p, o = pts[valid], objs[valid]
    npt.assert_allclose(np.linalg.norm(p - center, axis=1), radius, atol=1e-6)  # on the sphere
    lhs, rhs = _snell(p, o, CAMERA, rb._sphere_normal(p, center), N_CORNEA, N_AIR)
    npt.assert_allclose(lhs, rhs, atol=1e-6)


def _dual_forward_inverse(conic: bool) -> None:
    if conic:
        c = ConicCornea()
        c.setup_eye_geometry(24.75)
        ant = (_xyz(c.center), c.anterior_radius, c.anterior_k)
        post = (_xyz(c.get_posterior_center()), c.posterior_radius, c.posterior_k)

        def solve(cam: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return rb.find_refraction_dual_conic_batch(
                cam, obj, ant[0], ant[1], ant[2], post[0], post[1], post[2], N_AIR, N_CORNEA, N_AQ
            )

    else:
        c = SphericalCornea()
        c.setup_eye_geometry(24.75)
        ant = (_xyz(c.center), c.anterior_radius, None)
        post = (_xyz(c.get_posterior_center()), c.posterior_radius, None)

        def solve(cam: np.ndarray, obj: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            return rb.find_refraction_dual_sphere_batch(
                cam, obj, ant[0], ant[1], post[0], post[1], N_AIR, N_CORNEA, N_AQ
            )

    stack = [(post, N_CORNEA), (ant, N_AIR)]  # object side out: aqueous -> posterior -> cornea -> anterior -> air
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(60):
        radius = 3.0 * np.sqrt(rng.random())  # pupil-sized object, like the real boundary
        theta = 2 * np.pi * rng.random()
        obj = np.array([[radius * np.cos(theta), radius * np.sin(theta), -8.79]])
        front = np.array([[3.0 * (rng.random() - 0.5), 3.0 * (rng.random() - 0.5), -200.0]])  # small-angle
        exit_pt, exit_dir, valid = rb._trace_stack(front - obj, obj, stack, N_AQ, conic)
        if not valid[0]:
            continue
        camera = exit_pt[0] + 400.0 * exit_dir[0]  # camera on the traced exit ray
        pts, found = solve(camera, obj)
        assert found[0], "dual solver failed to invert a valid forward trace"
        npt.assert_allclose(pts[0], exit_pt[0], atol=1e-5)
        checked += 1
    assert checked >= 20, f"only {checked} forward-inverse cases succeeded"


def test_dual_conic_forward_inverse() -> None:
    """The dual conic solver inverts a known forward ray-trace through both corneal surfaces."""
    _dual_forward_inverse(conic=True)


def test_dual_sphere_forward_inverse() -> None:
    """The dual sphere solver inverts a known forward ray-trace through both corneal surfaces."""
    _dual_forward_inverse(conic=False)
