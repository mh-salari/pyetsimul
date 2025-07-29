"""Unit tests for coordinate system validation in Eye and Camera classes."""

import numpy as np
import pytest
from et_simul.core.eye import Eye
from et_simul.core.camera import Camera
from et_simul.core import enforce_right_handed_coordinates, is_right_handed_enforced


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


def test_coordinate_system_enforcement_flag():
    """Test that the coordinate system enforcement flag works correctly."""
    # Store original state
    original_state = is_right_handed_enforced()

    try:
        # Test default behavior (right-handed enforced)
        enforce_right_handed_coordinates(True)
        assert is_right_handed_enforced()

        eye = Eye()
        camera = Camera()

        # Valid matrices
        right_handed = np.eye(3)  # Identity matrix (det = +1)
        left_handed = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 1]])  # det = -1

        # Right-handed should work when enforcement is enabled
        eye.set_rest_orientation(right_handed)
        camera.orientation = right_handed

        # Left-handed should fail when enforcement is enabled
        with pytest.raises(ValueError, match="Left-handed coordinate system detected"):
            eye.set_rest_orientation(left_handed)

        with pytest.raises(ValueError, match="Left-handed coordinate system detected"):
            camera.orientation = left_handed

        # Disable enforcement
        enforce_right_handed_coordinates(False)
        assert not is_right_handed_enforced()

        # Both should work when enforcement is disabled
        eye.set_rest_orientation(left_handed)
        eye.set_rest_orientation(right_handed)
        camera.orientation = left_handed
        camera.orientation = right_handed

        # Re-enable enforcement
        enforce_right_handed_coordinates(True)
        assert is_right_handed_enforced()

        # Left-handed should fail again
        with pytest.raises(ValueError, match="Left-handed coordinate system detected"):
            eye.set_rest_orientation(left_handed)

        with pytest.raises(ValueError, match="Left-handed coordinate system detected"):
            camera.orientation = left_handed

    finally:
        # Restore original state
        enforce_right_handed_coordinates(original_state)
