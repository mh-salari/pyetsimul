"""Unit tests for Eye.find_cr_simple method."""

import numpy as np
from pyetsimul.types import Position3D
from pyetsimul.core.eye import Eye
from pyetsimul.core.light import Light
from pyetsimul.core.camera import Camera
from pyetsimul.optics.reflections import find_corneal_reflection_simple


def test_basic_simple_corneal_reflex():
    """Test basic simple corneal reflex with MATLAB reference values."""
    e = Eye()
    l = Light(position=Position3D(0, 0, -50))
    c = Camera()

    # Position camera to the side
    c.trans[0:3, 3] = np.array([30, 0, -40])

    cr = find_corneal_reflection_simple(e, l, c)

    # MATLAB reference values
    expected_cr = Position3D(0.0, 0.0, -0.0093376952886582)

    assert cr is not None
    cr.assert_close(expected_cr, rtol=1e-10, atol=1e-12)

    # Test 4D homogeneous coordinates
    arr = np.array(cr)
    assert arr.shape == (4,)
    assert arr[3] == 1.0


def test_angled_positions():
    """Test simple corneal reflex with angled light and camera positions."""
    e = Eye()
    l = Light(position=Position3D(15, -10, -45))
    c = Camera()

    # Camera at different position
    c.trans[0:3, 3] = np.array([-20, 8, -35])

    cr = find_corneal_reflection_simple(e, l, c)

    # MATLAB reference values
    expected_cr = Position3D(0.0020587253340034, -0.0013724835560023, -0.0105255789716633)

    assert cr is not None
    cr.assert_close(expected_cr, rtol=1e-10, atol=1e-12)

    # Test 4D homogeneous coordinates
    arr = np.array(cr)
    assert arr.shape == (4,)
    assert arr[3] == 1.0


def test_reflex_outside_cornea_boundary():
    """Test case where simple reflex falls outside cornea boundary - should return None."""
    e = Eye()
    l = Light(position=Position3D(10, 50, -30))
    c = Camera()

    # Eye looking down, light and camera positioned to create reflex outside boundary
    target_down = Position3D(0, -100, 0)
    e.look_at(target_down)
    c.trans[0:3, 3] = np.array([20, 80, -20])

    cr = find_corneal_reflection_simple(e, l, c)

    # Should return None when reflex is outside cornea boundary
    assert cr is None


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye()
    l = Light(position=Position3D(0, 0, -50))
    c = Camera()
    c.trans[0:3, 3] = np.array([30, 0, -40])

    cr = find_corneal_reflection_simple(e, l, c)
    assert cr is not None, "cr should not be None for these inputs"

    # Check types and shapes
    arr = np.array(cr)
    assert arr.shape == (4,)
    assert arr[3] == 1.0  # Homogeneous coordinate
