"""Unit tests for find_reflection_sphere function."""

import numpy as np
from et_simul.optics.reflections import find_reflection_sphere
from et_simul.types import Position3D, Point3D


def test_specific_reflection_case():
    """Test reflection calculation with MATLAB reference values."""
    # Test parameters from MATLAB
    light_pos = Position3D(0.25, 0, 0)  # Light position
    camera_pos = Position3D(0.1, -0.1, 0)  # Camera position
    sphere_center = Position3D(0.05, 0, 1.0)  # Sphere center
    sphere_radius = 0.8  # Sphere radius

    glint_pos = find_reflection_sphere(light_pos, camera_pos, sphere_center, sphere_radius)

    # MATLAB reference values
    expected_glint_pos = Point3D(0.1472592734935877, -0.0408954315633531, 0.2069878958069135)

    glint_pos.assert_close(expected_glint_pos, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    light_pos = Position3D(0.25, 0, 0)
    camera_pos = Position3D(0.1, -0.1, 0)
    sphere_center = Position3D(0.05, 0, 1.0)
    sphere_radius = 0.8

    glint_pos = find_reflection_sphere(light_pos, camera_pos, sphere_center, sphere_radius)

    assert glint_pos is not None, "glint_pos should not be None for these inputs"
    # Check types and shapes
    assert isinstance(glint_pos, Point3D)

    # Should be finite values
    assert np.all(np.isfinite(np.array(glint_pos)))
