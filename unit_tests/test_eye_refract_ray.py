"""Unit tests for Eye.refract_ray method."""

import numpy as np
from et_simul.core.eye import Eye


def test_basic_ray_refraction():
    """Test basic ray refraction with MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    
    # Ray from outside eye - origin and direction
    R0 = np.array([5.0, 2.0, 50.0])
    Rd = np.array([-0.1, -0.04, -1.0])
    
    # Normalize direction vector (as done in MATLAB)
    Rd = Rd / np.linalg.norm(Rd)
    
    U0, Ud = e.refract_ray(R0, Rd)
    
    # MATLAB reference values
    expected_U0 = np.array([0.0003620467382754, 0.0001448186953101, 0.0036204673827527])
    expected_Ud = np.array([-0.0846693339523489, -0.0338677335809380, -0.9958333598098410])
    
    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-12, atol=1e-15)
    
    # Test that refracted direction is normalized
    assert abs(np.linalg.norm(Ud) - 1.0) < 1e-15


def test_optical_axis_ray():
    """Test ray along optical axis with MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    
    # Ray along optical axis
    R0 = np.array([0.0, 0.0, 30.0])
    Rd = np.array([0.0, 0.0, -1.0])
    
    U0, Ud = e.refract_ray(R0, Rd)
    
    # MATLAB reference values
    expected_U0 = np.array([0.0000000000000000, 0.0000000000000000, 0.0036299999993510])
    expected_Ud = np.array([0.0000000000000000, 0.0000000000000000, -1.0000000000000000])
    
    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-12, atol=1e-15)
    
    # Test that refracted direction is normalized
    assert abs(np.linalg.norm(Ud) - 1.0) < 1e-15


def test_ray_missing_eye():
    """Test ray that misses the eye - should return None."""
    e = Eye(fovea_displacement=False)
    
    # Ray that misses the eye completely
    R0 = np.array([20.0, 20.0, 50.0])
    Rd = np.array([0.0, 0.0, -1.0])
    
    U0, Ud = e.refract_ray(R0, Rd)
    
    # Should return None for both when ray misses eye
    assert U0 is None
    assert Ud is None


def test_homogeneous_coordinates():
    """Test with homogeneous coordinates and MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    
    # 4D homogeneous coordinates
    R0 = np.array([3.0, 1.5, 40.0, 1.0])
    Rd = np.array([-0.075, -0.0375, -1.0, 0.0])
    
    # Normalize direction (3D part)
    Rd[:3] = Rd[:3] / np.linalg.norm(Rd[:3])
    
    U0, Ud = e.refract_ray(R0, Rd)
    
    # MATLAB reference values (4D)
    expected_U0 = np.array([0.0002718158445933, 0.0001359079222967, 0.0036242112612399, 1.0000000000000000])
    expected_Ud = np.array([-0.0636298985212294, -0.0318149492606147, -0.9974663127232531, 0.0000000000000000])
    
    assert U0 is not None
    assert Ud is not None
    np.testing.assert_allclose(U0, expected_U0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(Ud, expected_Ud, rtol=1e-12, atol=1e-15)
    
    # Test that 3D part of refracted direction is normalized
    assert abs(np.linalg.norm(Ud[:3]) - 1.0) < 1e-15


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye(fovea_displacement=False)
    
    R0 = np.array([5.0, 2.0, 50.0])
    Rd = np.array([-0.1, -0.04, -1.0])
    Rd = Rd / np.linalg.norm(Rd)
    
    U0, Ud = e.refract_ray(R0, Rd)
    
    if U0 is not None and Ud is not None:
        # Check types and shapes
        assert isinstance(U0, np.ndarray)
        assert isinstance(Ud, np.ndarray)
        assert U0.dtype == np.float64
        assert Ud.dtype == np.float64
        assert U0.shape == (3,) or U0.shape == (4,)  # Can be 3D or 4D
        assert Ud.shape == (3,) or Ud.shape == (4,)  # Can be 3D or 4D
        
        # Refracted direction should be normalized
        if Ud.shape == (3,):
            assert np.isclose(np.linalg.norm(Ud), 1.0, rtol=1e-12)
        else:  # 4D homogeneous
            assert np.isclose(np.linalg.norm(Ud[:3]), 1.0, rtol=1e-12)