"""Unit tests for coordinate system validation in Eye and Camera classes."""

import numpy as np
import pytest
from et_simul.core.eye import Eye
from et_simul.core.camera import Camera


def test_eye_valid_right_handed_orientation():
    """Test that valid right-handed eye orientation is accepted."""
    eye = Eye()
    identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    eye.set_rest_orientation(identity)  # Should not raise


def test_eye_left_handed_orientation_rejected():
    """Test that left-handed eye orientation is rejected."""
    eye = Eye()
    left_handed = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    
    with pytest.raises(ValueError, match="Left-handed coordinate system detected"):
        eye.set_rest_orientation(left_handed)


def test_eye_invalid_matrix_rejected():
    """Test that invalid eye rotation matrix is rejected."""
    eye = Eye()
    scaled = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    with pytest.raises(ValueError, match="Invalid rotation matrix"):
        eye.set_rest_orientation(scaled)


def test_camera_valid_right_handed_orientation():
    """Test that valid right-handed camera orientation is accepted."""
    camera = Camera()
    identity = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    camera.orientation = identity  # Should not raise


def test_camera_left_handed_orientation_rejected():
    """Test that left-handed camera orientation is rejected."""
    camera = Camera()
    left_handed = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])
    
    with pytest.raises(ValueError, match="Left-handed coordinate system detected"):
        camera.orientation = left_handed


def test_camera_invalid_matrix_rejected():
    """Test that invalid camera rotation matrix is rejected."""
    camera = Camera()
    scaled = np.array([[2, 0, 0], [0, 1, 0], [0, 0, 1]])
    
    with pytest.raises(ValueError, match="Invalid rotation matrix"):
        camera.orientation = scaled