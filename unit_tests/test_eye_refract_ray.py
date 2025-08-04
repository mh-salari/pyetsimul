"""Unit tests for Eye.refract_ray method."""

import numpy as np
from et_simul.core.eye import Eye
from et_simul.optics.eye_optics import refract_ray_at_cornea
from et_simul.types import Position3D, Vector3D


def test_basic_ray_refraction():
    """Test basic ray refraction with MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    # Ray from outside eye - origin and direction
    R0 = Position3D(5.0, 2.0, 50.0)
    Rd = Vector3D(-0.1, -0.04, -1.0).normalize()

    U0, Ud = refract_ray_at_cornea(e, R0, Rd)

    # MATLAB reference values
    expected_U0 = np.array([0.0003620467382754, 0.0001448186953101, 0.0036204673827527, 1.0])
    expected_Ud = np.array([-0.0846693339523489, -0.0338677335809380, -0.9958333598098410, 0.0])

    assert U0 is not None
    assert Ud is not None
    U0.assert_close(Position3D.from_array(expected_U0[:3]), rtol=1e-12, atol=1e-15)
    Ud.assert_close(Vector3D.from_array(expected_Ud[:3]), rtol=1e-12, atol=1e-15)

    # Test that refracted direction is normalized
    assert abs(Ud.magnitude() - 1.0) < 1e-15


def test_optical_axis_ray():
    """Test ray along optical axis with MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # Ray along optical axis
    R0 = Position3D(0.0, 0.0, 30.0)
    Rd = Vector3D(0.0, 0.0, -1.0)

    U0, Ud = refract_ray_at_cornea(e, R0, Rd)

    # MATLAB reference values
    expected_U0 = np.array([0.0, 0.0, 0.0036299999993510, 1.0])
    expected_Ud = np.array([0.0, 0.0, -1.0, 0.0])

    assert U0 is not None
    assert Ud is not None
    U0.assert_close(Position3D.from_array(expected_U0[:3]), rtol=1e-12, atol=1e-15)
    Ud.assert_close(Vector3D.from_array(expected_Ud[:3]), rtol=1e-12, atol=1e-15)

    # Test that refracted direction is normalized
    assert abs(Ud.magnitude() - 1.0) < 1e-15


def test_ray_missing_eye():
    """Test ray that misses the eye - should return None."""
    e = Eye(fovea_displacement=False)

    # Ray that misses the eye completely
    R0 = Position3D(20.0, 20.0, 50.0)
    Rd = Vector3D(0.0, 0.0, -1.0)

    U0, Ud = refract_ray_at_cornea(e, R0, Rd)

    # Should return None for both when ray misses eye
    assert U0 is None
    assert Ud is None


def test_homogeneous_coordinates():
    """Test with homogeneous coordinates and MATLAB reference values."""
    e = Eye(fovea_displacement=False)

    # 4D homogeneous coordinates
    R0 = Position3D(3.0, 1.5, 40.0)
    Rd = Vector3D(-0.075, -0.0375, -1.0).normalize()

    U0, Ud = refract_ray_at_cornea(e, R0, Rd)

    # MATLAB reference values (4D)
    expected_U0 = np.array([0.0002718158445933, 0.0001359079222967, 0.0036242112612399, 1.0])
    expected_Ud = np.array(
        [
            -0.0636298985212294,
            -0.0318149492606147,
            -0.9974663127232531,
            0.0,
        ]
    )

    assert U0 is not None
    assert Ud is not None
    U0.assert_close(Position3D.from_array(expected_U0[:3]), rtol=1e-12, atol=1e-15)
    Ud.assert_close(Vector3D.from_array(expected_Ud[:3]), rtol=1e-12, atol=1e-15)

    # Test that 3D part of refracted direction is normalized
    assert abs(Ud.magnitude() - 1.0) < 1e-15


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye(fovea_displacement=False)

    R0 = Position3D(5.0, 2.0, 50.0)
    Rd = Vector3D(-0.1, -0.04, -1.0).normalize()

    U0, Ud = refract_ray_at_cornea(e, R0, Rd)
    assert U0 is not None, "U0 should not be None for these inputs"
    assert Ud is not None, "Ud should not be None for these inputs"

    # Check types and shapes - Position3D returns 4D homogeneous, Vector3D returns 3D
    arr_U0 = np.array(U0)
    arr_Ud = np.array(Ud)
    assert arr_U0.shape == (4,)  # Position3D returns 4D homogeneous coordinates
    assert arr_Ud.shape == (3,)  # Vector3D returns 3D coordinates

    # Refracted direction should be normalized
    assert np.isclose(Ud.magnitude(), 1.0, rtol=1e-12)
