"""Unit tests for find_reflection_conic function."""

import numpy as np
from et_simul.optics.reflections import find_reflection_sphere, find_reflection_conic
from et_simul.types import Position3D, Point3D


def test_conic_equals_sphere_reflection():
    """Test conic reflection with k=0 gives sphere behavior."""
    # Realistic eye tracking geometry based on example.py
    light_pos = Position3D(200e-3, 0, 0)  # Light position (200mm in x)
    camera_pos = Position3D(0, 0, -50e-3)  # Camera position (50mm in front, facing -z)
    radius = 7.98e-3  # Realistic corneal radius

    # For k=0 conic to match sphere at origin, conic center must be at (0,0,-R)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere at origin
    conic_center = Position3D(0.0, 0.0, 0)  # Conic center adjusted for k=0 equivalence

    # Sphere function with proper center
    sphere_glint = find_reflection_sphere(light_pos, camera_pos, sphere_center, radius)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_glint = find_reflection_conic(light_pos, camera_pos, conic_center, r_param, k)

    # Should match exactly since both create identical spheres
    assert sphere_glint is not None, "Sphere reflection should not be None for this camera position"
    assert conic_glint is not None, "Conic reflection should not be None for this camera position"
    sphere_glint.assert_close(conic_glint, rtol=1e-12, atol=1e-14)


def test_conic_output_properties():
    """Test that conic reflection output has correct properties."""
    light_pos = Position3D(200e-3, 0, 0)  # Realistic light position
    camera_pos = Position3D(0, 0, -50e-3)  # Realistic camera position
    conic_center = Position3D(0.0, 0.0, -10e-3)  # Conic center at realistic cornea position
    r_param = 7.98e-3  # Radius parameter
    k = -0.18  # Typical prolate cornea

    glint_pos = find_reflection_conic(light_pos, camera_pos, conic_center, r_param, k)
    assert glint_pos is not None, "glint_pos should not be None for this camera position"

    # Check types and shapes
    assert isinstance(glint_pos, Point3D)

    # Should be finite values
    assert np.all(np.isfinite(np.array(glint_pos)))


def test_prolate_vs_oblate_conic():
    """Test that prolate and oblate conics give different results."""
    light_pos = Position3D(200e-3, 0, 0)  # Realistic light position
    camera_pos = Position3D(0, 0, -50e-3)  # Realistic camera position
    conic_center = Position3D(0.0, 0.0, -10e-3)  # Conic center at realistic cornea position
    r_param = 7.98e-3  # Radius parameter

    # Prolate conic (typical cornea)
    k_prolate = -0.18
    glint_prolate = find_reflection_conic(light_pos, camera_pos, conic_center, r_param, k_prolate)

    # Oblate conic
    k_oblate = 0.18
    glint_oblate = find_reflection_conic(light_pos, camera_pos, conic_center, r_param, k_oblate)

    # Should give different results (unless by coincidence)
    assert glint_prolate is not None, "glint_prolate should not be None for this camera position"
    assert glint_oblate is not None, "glint_oblate should not be None for this camera position"

    # Results should be different for different asphericity
    assert not np.allclose(np.array(glint_prolate), np.array(glint_oblate), rtol=1e-10)
