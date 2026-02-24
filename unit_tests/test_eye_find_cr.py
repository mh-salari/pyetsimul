"""Unit tests for Eye.find_cr method."""

import numpy as np

from pyetsimul.core.camera import Camera
from pyetsimul.core.eye import Eye
from pyetsimul.core.light import Light
from pyetsimul.types.geometry import Position3D


def test_basic_corneal_reflex() -> None:
    """Test basic corneal reflex with MATLAB reference values."""
    e = Eye()

    l = Light(position=Position3D(0, 0, -50000))
    c = Camera()

    # Position camera to the side
    c.trans[0:3, 3] = np.array([30000, 0, -40000])

    cr = e.find_cr(l, c)

    # MATLAB reference values
    expected_cr = np.array([
        2.5236951722993,
        0.0,
        -11.9204268490828,
        1.0,
    ])

    assert cr is not None
    cr.assert_close(Position3D.from_array(expected_cr), rtol=1e-10, atol=1e-12)

    # Test CR is within cornea
    assert e.point_within_cornea(cr)


def test_different_positions() -> None:
    """Test corneal reflex with different light and camera positions."""
    e = Eye()
    l = Light(position=Position3D(20000, 15000, -60000))
    c = Camera()

    # Camera at different position
    c.trans[0:3, 3] = np.array([-25000, 10000, -35000])

    cr = e.find_cr(l, c)

    # MATLAB reference values
    expected_cr = np.array([
        -1.1496765821543,
        2.0336905185480,
        -11.9803831247988,
        1.0,
    ])

    assert cr is not None
    cr.assert_close(Position3D.from_array(expected_cr), rtol=1e-10, atol=1e-12)
    assert e.point_within_cornea(cr)


def test_light_behind_eye() -> None:
    """Test case where light is behind eye - should return None."""
    e = Eye()
    l = Light(position=Position3D(0, 0, 50000))
    c = Camera()

    # Camera in front
    c.trans[0:3, 3] = np.array([20000, 0, -30000])

    cr = e.find_cr(l, c)

    # Should return None when light is behind eye
    assert cr is None


def test_reflex_outside_cornea_boundary() -> None:
    """Test case where reflex falls outside cornea boundary - should return None."""
    e = Eye()
    l = Light(position=Position3D(5000, 30000, -40000))
    c = Camera()

    # Eye looking straight down
    target_down = Position3D(0, -100000, 0)
    e.look_at(target_down)
    c.trans[0:3, 3] = np.array([10000, 60000, -25000])

    cr = e.find_cr(l, c)

    # Should return None when reflection is outside cornea boundary
    assert cr is None


def test_output_properties() -> None:
    """Test that output has correct properties."""
    e = Eye()
    l = Light(position=Position3D(0, 0, -50000))
    c = Camera()
    c.trans[0:3, 3] = np.array([30000, 0, -40000])

    cr = e.find_cr(l, c)
    assert cr is not None, "cr should not be None for these inputs"

    # Check types and shapes
    arr = np.array(cr)
    assert arr.shape == (4,)
    assert arr[3] == 1.0  # Homogeneous coordinate
