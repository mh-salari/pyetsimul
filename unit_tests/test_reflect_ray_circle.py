"""Unit tests for reflect_ray_circle function."""

import numpy as np
from et_simul.optics.reflections import reflect_ray_circle


def test_basic_center_reflection():
    """Test basic center front reflection with MATLAB reference values."""
    # Ray hitting circle dead center from front
    R0 = np.array([0.0, -3.0])         # Ray origin
    Rd = np.array([0.0, 1.0])          # Ray direction (unit vector)
    C0 = np.array([0.0, 0.0])          # Circle center
    Cr = 2.0                           # Circle radius
    
    S0, Sd = reflect_ray_circle(R0, Rd, C0, Cr)
    
    # MATLAB reference values
    expected_S0 = np.array([0.0000000000000000, -2.0000000000000000])
    expected_Sd = np.array([0.0000000000000000, -1.0000000000000000])
    
    assert S0 is not None
    assert Sd is not None
    np.testing.assert_allclose(S0, expected_S0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(Sd, expected_Sd, rtol=1e-14, atol=1e-15)


def test_angled_reflection():
    """Test angled reflection with MATLAB reference values."""
    # Angled ray with non-normalized input direction
    R0 = np.array([-2.0, -2.0])       # Ray origin
    Rd = np.array([1.0, 1.0])         # Ray direction (not normalized)
    C0 = np.array([0.0, 0.0])         # Circle center
    Cr = 1.5                          # Circle radius
    
    S0, Sd = reflect_ray_circle(R0, Rd, C0, Cr)
    
    # MATLAB reference values
    expected_S0 = np.array([-1.0606601717798214, -1.0606601717798214])
    expected_Sd = np.array([-0.7071067811865472, -0.7071067811865472])
    
    assert S0 is not None
    assert Sd is not None
    np.testing.assert_allclose(S0, expected_S0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(Sd, expected_Sd, rtol=1e-14, atol=1e-15)
    
    # Test that reflected direction is normalized
    assert abs(np.linalg.norm(Sd) - 1.0) < 1e-14


def test_offset_circle():
    """Test reflection with offset circle and MATLAB reference values."""
    # Circle not centered at origin
    R0 = np.array([0.0, 0.0])         # Ray origin
    Rd = np.array([0.6, 0.8])         # Ray direction (not unit)
    C0 = np.array([2.0, 2.0])         # Circle center
    Cr = 1.0                          # Circle radius
    
    S0, Sd = reflect_ray_circle(R0, Rd, C0, Cr)
    
    # MATLAB reference values
    expected_S0 = np.array([1.1300909166052995, 1.5067878888070660])
    expected_Sd = np.array([-0.9945696889543475, -0.1040727332842388])
    
    assert S0 is not None
    assert Sd is not None
    np.testing.assert_allclose(S0, expected_S0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(Sd, expected_Sd, rtol=1e-14, atol=1e-15)


def test_grazing_incidence():
    """Test grazing incidence with MATLAB reference values."""
    # Ray parallel to tangent
    R0 = np.array([-3.0, 0.0])        # Ray origin
    Rd = np.array([1.0, 0.0])         # Ray direction (horizontal)
    C0 = np.array([0.0, 0.0])         # Circle center
    Cr = 1.0                          # Circle radius
    
    S0, Sd = reflect_ray_circle(R0, Rd, C0, Cr)
    
    # MATLAB reference values
    expected_S0 = np.array([-1.0000000000000000, 0.0000000000000000])
    expected_Sd = np.array([-1.0000000000000000, 0.0000000000000000])
    
    assert S0 is not None
    assert Sd is not None
    np.testing.assert_allclose(S0, expected_S0, rtol=1e-14, atol=1e-15)
    np.testing.assert_allclose(Sd, expected_Sd, rtol=1e-14, atol=1e-15)


def test_ray_missing_circle():
    """Test ray that misses circle - should return None."""
    # Ray that doesn't intersect circle
    R0 = np.array([0.0, 0.0])         # Ray origin
    Rd = np.array([1.0, 0.0])         # Ray direction
    C0 = np.array([0.0, 3.0])         # Circle center (away from ray)
    Cr = 1.0                          # Circle radius
    
    S0, Sd = reflect_ray_circle(R0, Rd, C0, Cr)
    
    # Should return None for both when ray misses circle
    assert S0 is None
    assert Sd is None


def test_output_properties():
    """Test that output has correct properties."""
    R0 = np.array([0.0, -3.0])
    Rd = np.array([0.0, 1.0])
    C0 = np.array([0.0, 0.0])
    Cr = 2.0
    
    S0, Sd = reflect_ray_circle(R0, Rd, C0, Cr)
    
    if S0 is not None and Sd is not None:
        # Check types and shapes
        assert isinstance(S0, np.ndarray)
        assert isinstance(Sd, np.ndarray)
        assert S0.dtype == np.float64
        assert Sd.dtype == np.float64
        assert S0.shape == (2,)
        assert Sd.shape == (2,)
        
        # Reflected direction should be normalized
        assert np.isclose(np.linalg.norm(Sd), 1.0, rtol=1e-12)