"""Unit tests for Eye.refract_ray_advanced method."""

import numpy as np
from et_simul.core.eye import Eye


def test_optical_axis_ray():
    """Test ray along optical axis with MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    
    # Ray along optical axis
    R0 = np.array([0.0, 0.0, 25.0])
    Rd = np.array([0.0, 0.0, -1.0])
    
    O0, I0, Id = e.refract_ray_advanced(R0, Rd)
    
    # MATLAB reference values
    expected_O0 = np.array([0.0000000000000000, 0.0000000000000000, 0.0036299999993510])
    expected_I0 = np.array([0.0000000000000000, 0.0000000000000000, 0.0012600000000000])
    expected_Id = np.array([0.0000000000000000, 0.0000000000000000, -1.0000000000000000])
    
    assert O0 is not None
    assert I0 is not None
    assert Id is not None
    np.testing.assert_allclose(O0, expected_O0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(I0, expected_I0, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(Id, expected_Id, rtol=1e-12, atol=1e-15)
    
    # Test that final direction is normalized
    assert abs(np.linalg.norm(Id) - 1.0) < 1e-15


def test_ray_not_completing_path():
    """Test ray that doesn't complete path through cornea with MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    
    # Ray from outside eye that doesn't complete path
    R0 = np.array([4.0, 1.5, 45.0])
    Rd = np.array([-0.08, -0.03, -1.0])
    
    # Normalize direction vector (as done in MATLAB)
    Rd = Rd / np.linalg.norm(Rd)
    
    O0, I0, Id = e.refract_ray_advanced(R0, Rd)
    
    # MATLAB reference: Ray does not complete path (empty)
    assert O0 is None
    assert I0 is None
    assert Id is None


def test_ray_missing_eye():
    """Test ray that misses eye completely with MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    
    # Ray that misses the eye completely
    R0 = np.array([15.0, 15.0, 40.0])
    Rd = np.array([0.0, 0.0, -1.0])
    
    O0, I0, Id = e.refract_ray_advanced(R0, Rd)
    
    # MATLAB reference: Ray misses eye (empty)
    assert O0 is None
    assert I0 is None
    assert Id is None


def test_homogeneous_coordinates():
    """Test with homogeneous coordinates and MATLAB reference values."""
    e = Eye(fovea_displacement=False)
    
    # Test with 4D homogeneous coordinates
    R0 = np.array([2.5, 1.0, 35.0, 1.0])
    Rd = np.array([-0.07, -0.025, -1.0, 0.0])
    
    # Normalize direction (3D part only, as done in MATLAB)
    Rd[:3] = Rd[:3] / np.linalg.norm(Rd[:3])
    
    O0, I0, Id = e.refract_ray_advanced(R0, Rd)
    
    # MATLAB reference: Ray does not complete path (empty)
    assert O0 is None
    assert I0 is None
    assert Id is None


def test_output_properties():
    """Test that output has correct properties when valid."""
    e = Eye(fovea_displacement=False)
    
    # Use optical axis case that produces valid results
    R0 = np.array([0.0, 0.0, 25.0])
    Rd = np.array([0.0, 0.0, -1.0])
    
    O0, I0, Id = e.refract_ray_advanced(R0, Rd)
    
    if O0 is not None and I0 is not None and Id is not None:
        # Check types and shapes
        assert isinstance(O0, np.ndarray)
        assert isinstance(I0, np.ndarray)
        assert isinstance(Id, np.ndarray)
        assert O0.dtype == np.float64
        assert I0.dtype == np.float64
        assert Id.dtype == np.float64
        assert O0.shape == (3,)
        assert I0.shape == (3,)
        assert Id.shape == (3,)
        
        # Final direction should be normalized
        assert np.isclose(np.linalg.norm(Id), 1.0, rtol=1e-12)