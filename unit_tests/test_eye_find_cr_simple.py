"""Unit tests for Eye.find_cr_simple method."""

import numpy as np
from et_simul.core.eye import Eye
from et_simul.core.light import Light
from et_simul.core.camera import Camera


def test_basic_simple_corneal_reflex():
    """Test basic simple corneal reflex with MATLAB reference values."""
    e = Eye()
    l = Light()
    c = Camera()
    
    # Position light in front of eye
    l.position = np.array([0, 0, -50, 1])
    
    # Position camera to the side
    c.trans[0:3, 3] = np.array([30, 0, -40])
    
    cr = e.find_cr_simple(l, c)
    
    # MATLAB reference values
    expected_cr = np.array([0.0000000000000000, 0.0000000000000000, -0.0093376952886582, 1.0000000000000000])
    
    assert cr is not None
    np.testing.assert_allclose(cr, expected_cr, rtol=1e-10, atol=1e-12)
    
    # Test 4D homogeneous coordinates
    assert cr.shape == (4,)
    assert cr[3] == 1.0


def test_angled_positions():
    """Test simple corneal reflex with angled light and camera positions."""
    e = Eye()
    l = Light()
    c = Camera()
    
    # Light at an angle
    l.position = np.array([15, -10, -45, 1])
    
    # Camera at different position
    c.trans[0:3, 3] = np.array([-20, 8, -35])
    
    cr = e.find_cr_simple(l, c)
    
    # MATLAB reference values
    expected_cr = np.array([0.0020587253340034, -0.0013724835560023, -0.0105255789716633, 1.0000000000000000])
    
    assert cr is not None
    np.testing.assert_allclose(cr, expected_cr, rtol=1e-10, atol=1e-12)
    
    # Test 4D homogeneous coordinates
    assert cr.shape == (4,)
    assert cr[3] == 1.0


def test_reflex_outside_cornea_boundary():
    """Test case where simple reflex falls outside cornea boundary - should return None."""
    e = Eye()
    l = Light()
    c = Camera()
    
    # Eye looking down, light and camera positioned to create reflex outside boundary
    target_down = np.array([0, -100, 0])
    e.look_at(target_down)
    
    # Extreme positioning to force reflex outside cornea
    l.position = np.array([10, 50, -30, 1])
    c.trans[0:3, 3] = np.array([20, 80, -20])
    
    cr = e.find_cr_simple(l, c)
    
    # Should return None when reflex is outside cornea boundary
    assert cr is None


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye()
    l = Light()
    c = Camera()
    
    l.position = np.array([0, 0, -50, 1])
    c.trans[0:3, 3] = np.array([30, 0, -40])
    
    cr = e.find_cr_simple(l, c)
    
    if cr is not None:
        # Check types and shapes
        assert isinstance(cr, np.ndarray)
        assert cr.dtype == np.float64
        assert cr.shape == (4,)
        assert cr[3] == 1.0  # Homogeneous coordinate