"""Unit tests for angle2gaze function."""

import numpy as np
from pyetsimul.geometry.conversions import angle2gaze
from pyetsimul.types import Direction3D, Point2D


def test_zero_angles():
    """Test with zero angles - should point in -y direction."""
    result = angle2gaze(Point2D(0.0, 0.0))
    expected = Direction3D(0.0, -1.0, 0.0)
    result.assert_close(expected, rtol=1e-10)


def test_positive_angles():
    """Test with positive angles."""
    result = angle2gaze(Point2D(0.1, 0.2))
    expected = Direction3D(0.0978433950072557, -0.9751703272018158, 0.1986693307950612)
    result.assert_close(expected, rtol=1e-10)


def test_negative_angles():
    """Test with negative angles."""
    result = angle2gaze(Point2D(-0.15, -0.25))
    expected = Direction3D(-0.1447924628309112, -0.9580325796404553, -0.2474039592545229)
    result.assert_close(expected, rtol=1e-10)


def test_custom_rest_position():
    """Test with custom rest position matrix."""
    angles = Point2D(0.1, 0.1)
    rest_pos = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    result = angle2gaze(angles, rest_pos)
    expected = Direction3D(0.0998334166468282, -0.9900332889206207, 0.0993346653975306)
    result.assert_close(expected, rtol=1e-10)


def test_output_properties():
    """Test that output has correct properties."""
    result = angle2gaze(Point2D(0.1, 0.2))

    # Should be Direction3D object
    assert isinstance(result, Direction3D)

    # Should have correct attributes
    assert hasattr(result, "x")
    assert hasattr(result, "y")
    assert hasattr(result, "z")

    # The to_array() method should return a 4D homogeneous vector with the last component as 0
    result_array = np.array(result)
    assert result_array.shape == (4,)
    assert result_array[3] == 0.0
