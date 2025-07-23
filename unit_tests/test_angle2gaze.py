"""Unit tests for angle2gaze function."""

import numpy as np
from et_simul.geometry.conversions import angle2gaze


def test_zero_angles():
    """Test with zero angles - should point in -y direction."""
    result = angle2gaze([0.0, 0.0])
    expected = np.array([0.0, -1.0, 0.0, 0.0])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_positive_angles():
    """Test with positive angles."""
    result = angle2gaze([0.1, 0.2])
    expected = np.array([0.0978433950072557, -0.9751703272018158, 0.1986693307950612, 0.0])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_negative_angles():
    """Test with negative angles."""
    result = angle2gaze([-0.15, -0.25])
    expected = np.array([-0.1447924628309112, -0.9580325796404553, -0.2474039592545229, 0.0])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_custom_rest_position():
    """Test with custom rest position matrix."""
    angles = [0.1, 0.1]
    rest_pos = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    result = angle2gaze(angles, rest_pos)
    expected = np.array([0.0998334166468282, -0.9900332889206207, 0.0993346653975306, 0.0])
    np.testing.assert_allclose(result, expected, rtol=1e-10)


def test_output_properties():
    """Test that output has correct properties."""
    result = angle2gaze([0.1, 0.2])
    
    # Should be 4D homogeneous vector
    assert result.shape == (4,)
    
    # Fourth component should be 0 (direction vector)
    assert result[3] == 0.0
    
    # Should be numpy array
    assert isinstance(result, np.ndarray)