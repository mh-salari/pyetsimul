"""Unit tests for Eye.find_cr method."""

import numpy as np
from et_simul.core.eye import Eye
from et_simul.core.light import Light
from et_simul.core.camera import Camera


def test_basic_corneal_reflex():
    """Test basic corneal reflex with MATLAB reference values."""
    e = Eye()
    l = Light(position=np.array([0, 0, -50, 1]))
    c = Camera()

    # Position camera to the side
    c.trans[0:3, 3] = np.array([30, 0, -40])

    cr = e.find_cr(l, c)

    # MATLAB reference values
    expected_cr = np.array(
        [
            0.0025236951722993,
            0.0,
            -0.0119204268490828,
            1.0,
        ]
    )

    assert cr is not None
    np.testing.assert_allclose(cr, expected_cr, rtol=1e-10, atol=1e-12)

    # Test CR is within cornea
    assert e.point_within_cornea(cr)


def test_different_positions():
    """Test corneal reflex with different light and camera positions."""
    e = Eye()
    l = Light(position=np.array([20, 15, -60, 1]))
    c = Camera()

    # Camera at different position
    c.trans[0:3, 3] = np.array([-25, 10, -35])

    cr = e.find_cr(l, c)

    # MATLAB reference values
    expected_cr = np.array(
        [
            -0.0011496765821543,
            0.0020336905185480,
            -0.0119803831247988,
            1.0,
        ]
    )

    assert cr is not None
    np.testing.assert_allclose(cr, expected_cr, rtol=1e-10, atol=1e-12)
    assert e.point_within_cornea(cr)


def test_light_behind_eye():
    """Test case where light is behind eye - should return None."""
    e = Eye()
    l = Light(position=np.array([0, 0, 50, 1]))
    c = Camera()

    # Camera in front
    c.trans[0:3, 3] = np.array([20, 0, -30])

    cr = e.find_cr(l, c)

    # Should return None when light is behind eye
    assert cr is None


def test_reflex_outside_cornea_boundary():
    """Test case where reflex falls outside cornea boundary - should return None."""
    e = Eye()
    l = Light(position=np.array([5, 30, -40, 1]))
    c = Camera()

    # Eye looking straight down
    target_down = np.array([0, -100, 0, 1])
    e.look_at(target_down)
    c.trans[0:3, 3] = np.array([10, 60, -25])

    cr = e.find_cr(l, c)

    # Should return None when reflection is outside cornea boundary
    assert cr is None


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye()
    l = Light(position=np.array([0, 0, -50, 1]))
    c = Camera()
    c.trans[0:3, 3] = np.array([30, 0, -40])

    cr = e.find_cr(l, c)
    assert cr is not None, "cr should not be None for these inputs"

    # Check types and shapes
    assert isinstance(cr, np.ndarray)
    assert cr.dtype == np.float64
    assert cr.shape == (4,)
    assert cr[3] == 1.0  # Homogeneous coordinate
