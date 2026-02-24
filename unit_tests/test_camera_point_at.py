"""Unit tests for camera point_at function."""

import numpy as np

from pyetsimul.core.camera import Camera
from pyetsimul.types import Position3D


def test_point_at_basic() -> None:
    """Test basic point_at functionality."""
    c = Camera()
    original_trans = c.trans.copy()
    target = Position3D(x=5000, y=3000, z=-10000)

    c.point_at(target)

    # After point_at, both trans and rest_trans should be identical
    np.testing.assert_allclose(c.trans, c.rest_trans, rtol=1e-15)

    # Trans should have changed from original
    assert not np.allclose(c.trans, original_trans, rtol=1e-10)

    # Should be a valid transformation matrix
    rotation = c.orientation
    assert np.allclose(np.linalg.det(rotation), 1.0, rtol=1e-12)
    assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-12)


def test_point_at_vs_pan_tilt() -> None:
    """Test difference between point_at and pan_tilt regarding rest_trans."""
    c1 = Camera()
    c2 = Camera()
    target = Position3D(x=2000, y=4000, z=-8000)

    # Test pan_tilt (should not change rest_trans)
    original_rest = c1.rest_trans.copy()
    c1.pan_tilt(target)
    np.testing.assert_allclose(c1.rest_trans, original_rest, rtol=1e-15)

    # Test point_at (should change rest_trans to match trans)
    c2.point_at(target)
    np.testing.assert_allclose(c2.trans, c2.rest_trans, rtol=1e-15)

    # Both should have same final trans matrix
    np.testing.assert_allclose(c1.trans, c2.trans, rtol=1e-15)


def test_output_properties() -> None:
    """Test that point_at maintains proper camera properties."""
    c = Camera()
    original_focal_length = c.camera_matrix.focal_length
    target = Position3D(x=1000, y=-2000, z=-5000)

    c.point_at(target)

    # Should maintain camera properties
    assert c.camera_matrix.focal_length == original_focal_length
    assert c.trans.shape == (4, 4)
    assert c.trans.dtype == np.float64
    assert c.rest_trans.shape == (4, 4)
    assert c.rest_trans.dtype == np.float64

    # Transformation matrices should be orthogonal
    for matrix in [c.trans, c.rest_trans]:
        rotation = matrix[:3, :3]
        assert np.allclose(np.linalg.det(rotation), 1.0, rtol=1e-12)
        assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-12)
        assert np.allclose(matrix[3, :], [0, 0, 0, 1])
