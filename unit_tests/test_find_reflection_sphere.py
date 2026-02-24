"""Unit tests for find_reflection_sphere function."""

import numpy as np

from pyetsimul.optics.reflections import find_reflection_sphere
from pyetsimul.types import Point3D, Position3D


def test_specific_reflection_case() -> None:
    """Test reflection calculation with MATLAB reference values."""
    # Test parameters from MATLAB
    light_pos = Position3D(250, 0, 0)  # Light position
    camera_pos = Position3D(100, -100, 0)  # Camera position
    sphere_center = Position3D(50, 0, 1000)  # Sphere center
    sphere_radius = 800  # Sphere radius

    glint_pos = find_reflection_sphere(light_pos, camera_pos, sphere_center, sphere_radius)

    # MATLAB reference values
    expected_glint_pos = Point3D(147.2592734935877, -40.8954315633531, 206.9878958069135)

    glint_pos.assert_close(expected_glint_pos, rtol=1e-14, atol=1e-15)


def test_output_properties() -> None:
    """Test that output has correct properties."""
    light_pos = Position3D(250, 0, 0)
    camera_pos = Position3D(100, -100, 0)
    sphere_center = Position3D(50, 0, 1000)
    sphere_radius = 800

    glint_pos = find_reflection_sphere(light_pos, camera_pos, sphere_center, sphere_radius)

    assert glint_pos is not None, "glint_pos should not be None for these inputs"
    # Check types and shapes
    assert isinstance(glint_pos, Point3D)

    # Should be finite values
    assert np.all(np.isfinite(np.array(glint_pos)))
