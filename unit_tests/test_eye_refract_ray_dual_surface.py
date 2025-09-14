"""Unit tests for refract_ray_dual_surface function."""

import numpy as np

from pyetsimul.core.eye import Eye
from pyetsimul.optics.refractions import refract_ray_dual_surface
from pyetsimul.types.geometry import Direction3D, Point3D


def test_optical_axis_ray() -> None:
    """Test ray along optical axis with MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # Ray along optical axis
    ray_origin = Point3D(0.0, 0.0, 25.0)
    ray_direction = Direction3D(0.0, 0.0, -1.0)
    object_origin, intersection_point, intersection_direction = refract_ray_dual_surface(e, ray_origin, ray_direction)

    # MATLAB reference values
    expected_object_origin = np.array([0.0, 0.0, 0.0036299999993510, 1.0])
    expected_intersection_point = np.array([0.0, 0.0, 0.0012600000, 1.0])
    expected_intersection_direction = np.array([0.0, 0.0, -1.0, 0.0])

    assert object_origin is not None
    assert intersection_point is not None
    assert intersection_direction is not None
    object_origin.assert_close(Point3D.from_array(expected_object_origin[:3]), rtol=1e-12, atol=1e-15)
    intersection_point.assert_close(Point3D.from_array(expected_intersection_point[:3]), rtol=1e-12, atol=1e-15)
    intersection_direction.assert_close(
        Direction3D.from_array(expected_intersection_direction[:3]), rtol=1e-12, atol=1e-15
    )

    # Test that final direction is normalized
    assert abs(intersection_direction.magnitude() - 1.0) < 1e-15


def test_ray_not_completing_path() -> None:
    """Test ray that doesn't complete path through cornea with MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # Ray from outsintersection_directione eye that doesn't complete path
    ray_origin = Point3D(4.0, 1.5, 45.0)
    ray_direction = Direction3D(-0.08, -0.03, -1.0).normalize()
    object_origin, intersection_point, intersection_direction = refract_ray_dual_surface(e, ray_origin, ray_direction)

    # MATLAB reference: Ray does not complete path (empty)
    assert object_origin is None
    assert intersection_point is None
    assert intersection_direction is None


def test_ray_missing_eye() -> None:
    """Test ray that misses eye completely with MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # Ray that misses the eye completely
    ray_origin = Point3D(15.0, 15.0, 40.0)
    ray_direction = Direction3D(0.0, 0.0, -1.0)
    object_origin, intersection_point, intersection_direction = refract_ray_dual_surface(e, ray_origin, ray_direction)

    # MATLAB reference: Ray misses eye (empty)
    assert object_origin is None
    assert intersection_point is None
    assert intersection_direction is None


def test_output_properties() -> None:
    """Test that output has correct properties when valintersection_direction."""
    e = Eye(fovea_displacement=False)

    # Use optical axis case that produces valintersection_direction results
    ray_origin = Point3D(0.0, 0.0, 25.0)
    ray_direction = Direction3D(0.0, 0.0, -1.0)
    object_origin, intersection_point, intersection_direction = refract_ray_dual_surface(e, ray_origin, ray_direction)
    assert object_origin is not None, "object_origin should not be None for these inputs"
    assert intersection_point is not None, "intersection_point should not be None for these inputs"
    assert intersection_direction is not None, "intersection_direction should not be None for these inputs"

    # Check types and shapes - Point3D returns 3D, Direction3D returns 3D
    arr_object_origin = np.array(object_origin)
    arr_intersection_point = np.array(intersection_point)
    arr_intersection_direction = np.array(intersection_direction)
    assert arr_object_origin.shape == (3,)  # Point3D returns 3D cooray_directioninates
    assert arr_intersection_point.shape == (3,)  # Point3D returns 3D cooray_directioninates
    assert arr_intersection_direction.shape == (4,)  # Direction3D returns 4D homogenous cooray_directioninates

    # Final direction should be normalized
    assert np.isclose(intersection_direction.magnitude(), 1.0, rtol=1e-12)
