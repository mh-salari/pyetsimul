"""Unit tests for find_refraction_conic function."""

import numpy as np
from et_simul.optics.refractions import find_refraction_sphere, find_refraction_conic
from et_simul.types import Position3D, Point3D


def test_conic_equals_sphere_refraction():
    """Test conic refraction with k=0 gives sphere behavior."""
    # Realistic eye tracking geometry based on example.py
    radius = 7.98e-3  # Realistic corneal radius
    n_outside = 1.0  # Air
    n_inside = 1.376  # Cornea

    # Object behind cornea (pupil), camera in front
    object_pos = Position3D(0.0, 0.0, -6e-3)  # Object 6mm behind origin (pupil)
    camera_pos = Position3D(0.0, 0.0, 50e-3)  # Camera 50mm in front

    # For k=0 conic to match sphere at origin, conic center must be at (0,0,-R)
    sphere_center = Position3D(0.0, 0.0, 0.0)  # Sphere at origin
    conic_center = Position3D(0.0, 0.0, 0)  # Conic center adjusted for k=0 equivalence

    # Sphere function with proper center
    sphere_result = find_refraction_sphere(camera_pos, object_pos, sphere_center, radius, n_outside, n_inside)

    # Conic function with k=0
    r_param = radius
    k = 0.0  # k=0 represents a perfect sphere
    conic_result = find_refraction_conic(camera_pos, object_pos, conic_center, r_param, k, n_outside, n_inside)

    # Should match exactly since both create identical spheres
    assert sphere_result is not None, "sphere_result should not be None for this test case"
    assert conic_result is not None, "conic_result should not be None for this test case"
    sphere_result.assert_close(conic_result, rtol=1e-12, atol=1e-14)


def test_conic_refraction_basic():
    """Test basic conic refraction scenario."""
    # Realistic conic parameters
    conic_center = Position3D(0.0, 0.0, 0.0)  # Conic center at origin
    r_param = 7.98e-3  # Radius parameter
    k = -0.18  # Typical prolate cornea
    n_outside = 1.0  # Air
    n_conic = 1.376  # Cornea

    # Realistic eye tracking geometry
    object_pos = Position3D(0.0, 0.0, -6e-3)  # Object behind cornea (pupil)
    camera_pos = Position3D(0.0, 0.0, 50e-3)  # Camera in front

    intersection_point = find_refraction_conic(camera_pos, object_pos, conic_center, r_param, k, n_outside, n_conic)
    assert intersection_point is not None, "intersection_point should not be None for this basic refraction scenario"

    # Check types and shapes
    assert isinstance(intersection_point, Point3D)
    intersection_array = np.array(intersection_point)
    assert intersection_array.shape == (3,)
    assert intersection_array.dtype == np.float64

    # Should be finite values
    assert np.all(np.isfinite(intersection_array))


def test_different_k_values():
    """Test that different k values give different refraction results."""
    conic_center = Position3D(0.0, 0.0, 0.0)  # Conic center at origin
    r_param = 7.98e-3  # Radius parameter
    n_outside = 1.0
    n_conic = 1.376
    object_pos = Position3D(0.0, 0.0, -6e-3)  # Realistic object position (pupil)
    camera_pos = Position3D(0.0, 0.0, 50e-3)  # Realistic camera position

    # Prolate conic (typical cornea)
    k_prolate = -0.18
    result_prolate = find_refraction_conic(
        camera_pos, object_pos, conic_center, r_param, k_prolate, n_outside, n_conic
    )

    # Oblate conic
    k_oblate = 0.18
    result_oblate = find_refraction_conic(camera_pos, object_pos, conic_center, r_param, k_oblate, n_outside, n_conic)

    # Sphere (k=0)
    k_sphere = 0.0
    result_sphere = find_refraction_conic(camera_pos, object_pos, conic_center, r_param, k_sphere, n_outside, n_conic)

    # All should give different results (unless by coincidence)
    assert result_prolate is not None, "result_prolate should not be None"
    assert result_oblate is not None, "result_oblate should not be None"
    assert result_sphere is not None, "result_sphere should not be None"

    # Results should be different for different asphericity
    assert not np.allclose(np.array(result_prolate), np.array(result_oblate), rtol=1e-10)
    assert not np.allclose(np.array(result_prolate), np.array(result_sphere), rtol=1e-10)
    assert not np.allclose(np.array(result_oblate), np.array(result_sphere), rtol=1e-10)


def test_conic_output_properties():
    """Test that conic refraction output has correct properties."""
    conic_center = Position3D(0.0, 0.0, 0.0)  # Conic center at origin
    r_param = 7.98e-3  # Radius parameter
    k = -0.18
    n_outside = 1.0
    n_conic = 1.376
    object_pos = Position3D(0.0, 0.0, -6e-3)  # Realistic object position
    camera_pos = Position3D(0.0, 0.0, 50e-3)  # Realistic camera position

    intersection_point = find_refraction_conic(camera_pos, object_pos, conic_center, r_param, k, n_outside, n_conic)
    assert intersection_point is not None, "intersection_point should not be None for these inputs"

    # Check types and shapes
    assert isinstance(intersection_point, Point3D)
    intersection_array = np.array(intersection_point)
    assert intersection_array.dtype == np.float64
    assert intersection_array.shape == (3,)

    # Should be finite values
    assert np.all(np.isfinite(intersection_array))


def test_realistic_corneal_parameters():
    """Test with realistic corneal parameters."""
    # Realistic corneal geometry
    conic_center = Position3D(0.0, 0.0, 0.0)  # Conic center at origin
    r_param = 7.98e-3  # 7.98mm radius parameter
    k = -0.18  # Typical anterior cornea conic constant
    n_outside = 1.0  # Air
    n_cornea = 1.376  # Cornea

    # Object behind cornea (pupil), camera in front
    object_pos = Position3D(0.0, 0.0, -0.006)  # Object 6mm behind origin
    camera_pos = Position3D(0.0, 0.0, 0.05)  # Camera 5cm in front

    intersection_point = find_refraction_conic(camera_pos, object_pos, conic_center, r_param, k, n_outside, n_cornea)
    assert intersection_point is not None, "intersection_point should not be None for realistic corneal parameters"

    # Should be somewhere on the corneal surface
    assert isinstance(intersection_point, Point3D)
    intersection_array = np.array(intersection_point)
    assert intersection_array.shape == (3,)
    assert np.all(np.isfinite(intersection_array))

    # Z-coordinate should be near corneal surface
    # For original conic equation, vertex is at z = R/(1+k) from center
    vertex_z = conic_center.z + r_param / (1 + k)  # Expected vertex position
    # The intersection should be reasonably close to the corneal surface
    assert abs(intersection_point.z - vertex_z) < 0.05  # Within 5cm of vertex (very loose bounds)
    # For typical corneal parameters, intersection should be in positive z range
    assert -0.02 < intersection_point.z < 0.05  # Reasonable bounds for corneal surface
