"""Unit tests for the two-surface (posterior + anterior) inverse refraction functions."""

import pytest

from pyetsimul.optics.refractions import (
    find_refraction_conic,
    find_refraction_dual_conic,
    find_refraction_dual_sphere,
    find_refraction_sphere,
)
from pyetsimul.types import Position3D

# Cornea-like geometry shared by the tests: each surface apex sits on the z-axis (anterior apex at z=0,
# posterior apex at z=0.5), the pupil object lies behind both surfaces in the aqueous, and a distant
# on-axis camera observes from the air side.
CAMERA = Position3D(0.0, 0.0, -100.0)
ANTERIOR_CENTER = Position3D(0.0, 0.0, 8.0)
ANTERIOR_RADIUS = 8.0
POSTERIOR_CENTER = Position3D(0.0, 0.0, 7.0)
POSTERIOR_RADIUS = 6.5
N_AIR = 1.0
N_CORNEA = 1.376
N_AQUEOUS = 1.336

# Anterior conic with a realistic prolate constant whose apex is also placed at z=0 (center at R/(1+k)).
CONIC_ANTERIOR_K = -0.26
CONIC_ANTERIOR_CENTER = Position3D(0.0, 0.0, ANTERIOR_RADIUS / (1.0 + CONIC_ANTERIOR_K))


def test_dual_sphere_reduces_to_single_when_posterior_is_no_op() -> None:
    """A no-op posterior surface (n_aqueous == n_cornea) reduces the two-surface result to single-surface."""
    obj = Position3D(0.5, 0.0, 3.6)
    single = find_refraction_sphere(CAMERA, obj, ANTERIOR_CENTER, ANTERIOR_RADIUS, N_AIR, N_CORNEA)
    dual = find_refraction_dual_sphere(
        CAMERA, obj, ANTERIOR_CENTER, ANTERIOR_RADIUS, POSTERIOR_CENTER, POSTERIOR_RADIUS, N_AIR, N_CORNEA, N_CORNEA
    )
    assert single is not None
    assert dual is not None
    assert dual.x == pytest.approx(single.x, abs=1e-3)
    assert dual.y == pytest.approx(single.y, abs=1e-3)
    assert dual.z == pytest.approx(single.z, abs=1e-3)


def test_dual_conic_reduces_to_single_when_posterior_is_no_op() -> None:
    """The conic two-surface path with a no-op posterior matches the single conic refraction."""
    obj = Position3D(0.5, 0.0, 3.6)
    single = find_refraction_conic(
        CAMERA, obj, CONIC_ANTERIOR_CENTER, ANTERIOR_RADIUS, CONIC_ANTERIOR_K, N_AIR, N_CORNEA
    )
    dual = find_refraction_dual_conic(
        CAMERA,
        obj,
        CONIC_ANTERIOR_CENTER,
        ANTERIOR_RADIUS,
        CONIC_ANTERIOR_K,
        POSTERIOR_CENTER,
        POSTERIOR_RADIUS,
        0.0,
        N_AIR,
        N_CORNEA,
        N_CORNEA,
    )
    assert single is not None
    assert dual is not None
    assert dual.x == pytest.approx(single.x, abs=1e-3)
    assert dual.z == pytest.approx(single.z, abs=1e-3)


def test_dual_conic_with_zero_constant_matches_dual_sphere() -> None:
    """A conic with k=0 reproduces the independent spherical two-surface refraction."""
    obj = Position3D(1.5, 0.0, 3.6)
    sphere = find_refraction_dual_sphere(
        CAMERA, obj, ANTERIOR_CENTER, ANTERIOR_RADIUS, POSTERIOR_CENTER, POSTERIOR_RADIUS, N_AIR, N_CORNEA, N_AQUEOUS
    )
    conic = find_refraction_dual_conic(
        CAMERA,
        obj,
        ANTERIOR_CENTER,
        ANTERIOR_RADIUS,
        0.0,
        POSTERIOR_CENTER,
        POSTERIOR_RADIUS,
        0.0,
        N_AIR,
        N_CORNEA,
        N_AQUEOUS,
    )
    assert sphere is not None
    assert conic is not None
    assert conic.x == pytest.approx(sphere.x, abs=1e-6)
    assert conic.y == pytest.approx(sphere.y, abs=1e-6)
    assert conic.z == pytest.approx(sphere.z, abs=1e-6)


def test_dual_on_axis_refracts_to_apex() -> None:
    """An on-axis object refracts straight through both surfaces to the anterior apex (here z=0)."""
    obj = Position3D(0.0, 0.0, 3.6)
    sphere = find_refraction_dual_sphere(
        CAMERA, obj, ANTERIOR_CENTER, ANTERIOR_RADIUS, POSTERIOR_CENTER, POSTERIOR_RADIUS, N_AIR, N_CORNEA, N_AQUEOUS
    )
    conic = find_refraction_dual_conic(
        CAMERA,
        obj,
        CONIC_ANTERIOR_CENTER,
        ANTERIOR_RADIUS,
        CONIC_ANTERIOR_K,
        POSTERIOR_CENTER,
        POSTERIOR_RADIUS,
        0.0,
        N_AIR,
        N_CORNEA,
        N_AQUEOUS,
    )
    for result in (sphere, conic):
        assert result is not None
        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)
        assert result.z == pytest.approx(0.0, abs=1e-6)


def test_posterior_surface_changes_the_refraction() -> None:
    """A real aqueous->cornea index step shifts the apparent point from the anterior-only result."""
    obj = Position3D(1.5, 0.0, 3.6)
    anterior_only = find_refraction_dual_sphere(
        CAMERA, obj, ANTERIOR_CENTER, ANTERIOR_RADIUS, POSTERIOR_CENTER, POSTERIOR_RADIUS, N_AIR, N_CORNEA, N_CORNEA
    )
    two_surface = find_refraction_dual_sphere(
        CAMERA, obj, ANTERIOR_CENTER, ANTERIOR_RADIUS, POSTERIOR_CENTER, POSTERIOR_RADIUS, N_AIR, N_CORNEA, N_AQUEOUS
    )
    assert anterior_only is not None
    assert two_surface is not None
    assert abs(two_surface.x - anterior_only.x) + abs(two_surface.z - anterior_only.z) > 1e-4
