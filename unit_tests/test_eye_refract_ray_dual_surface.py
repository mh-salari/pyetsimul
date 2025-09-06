"""Unit tests for refract_ray_dual_surface function."""

import numpy as np
from pyetsimul.core.eye import Eye
from pyetsimul.types.geometry import Point3D, Direction3D
from pyetsimul.optics.refractions import refract_ray_dual_surface


def test_optical_axis_ray():
    """Test ray along optical axis with MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # Ray along optical axis
    R0 = Point3D(0.0, 0.0, 25.0)
    Rd = Direction3D(0.0, 0.0, -1.0)
    O0, I0, Id = refract_ray_dual_surface(e, R0, Rd)

    # MATLAB reference values
    expected_O0 = np.array([0.0, 0.0, 0.0036299999993510, 1.0])
    expected_I0 = np.array([0.0, 0.0, 0.0012600000, 1.0])
    expected_Id = np.array([0.0, 0.0, -1.0, 0.0])

    assert O0 is not None
    assert I0 is not None
    assert Id is not None
    O0.assert_close(Point3D.from_array(expected_O0[:3]), rtol=1e-12, atol=1e-15)
    I0.assert_close(Point3D.from_array(expected_I0[:3]), rtol=1e-12, atol=1e-15)
    Id.assert_close(Direction3D.from_array(expected_Id[:3]), rtol=1e-12, atol=1e-15)

    # Test that final direction is normalized
    assert abs(Id.magnitude() - 1.0) < 1e-15


def test_ray_not_completing_path():
    """Test ray that doesn't complete path through cornea with MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # Ray from outside eye that doesn't complete path
    R0 = Point3D(4.0, 1.5, 45.0)
    Rd = Direction3D(-0.08, -0.03, -1.0).normalize()
    O0, I0, Id = refract_ray_dual_surface(e, R0, Rd)

    # MATLAB reference: Ray does not complete path (empty)
    assert O0 is None
    assert I0 is None
    assert Id is None


def test_ray_missing_eye():
    """Test ray that misses eye completely with MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # Ray that misses the eye completely
    R0 = Point3D(15.0, 15.0, 40.0)
    Rd = Direction3D(0.0, 0.0, -1.0)
    O0, I0, Id = refract_ray_dual_surface(e, R0, Rd)

    # MATLAB reference: Ray misses eye (empty)
    assert O0 is None
    assert I0 is None
    assert Id is None


def test_output_properties():
    """Test that output has correct properties when valid."""
    e = Eye(fovea_displacement=False)

    # Use optical axis case that produces valid results
    R0 = Point3D(0.0, 0.0, 25.0)
    Rd = Direction3D(0.0, 0.0, -1.0)
    O0, I0, Id = refract_ray_dual_surface(e, R0, Rd)
    assert O0 is not None, "O0 should not be None for these inputs"
    assert I0 is not None, "I0 should not be None for these inputs"
    assert Id is not None, "Id should not be None for these inputs"

    # Check types and shapes - Point3D returns 3D, Direction3D returns 3D
    arr_O0 = np.array(O0)
    arr_I0 = np.array(I0)
    arr_Id = np.array(Id)
    assert arr_O0.shape == (3,)  # Point3D returns 3D coordinates
    assert arr_I0.shape == (3,)  # Point3D returns 3D coordinates
    assert arr_Id.shape == (4,)  # Direction3D returns 4D homogenous coordinates

    # Final direction should be normalized
    assert np.isclose(Id.magnitude(), 1.0, rtol=1e-12)
