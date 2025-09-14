"""Unit tests for listings_law function."""

import numpy as np

from pyetsimul.geometry.listings_law import calculate_eye_rotation
from pyetsimul.types import Vector3D


def test_3d_rotation() -> None:
    """Test complex 3D rotation with MATLAB reference values."""
    rest_dir = Vector3D(1.0, 1.0, 1.0)  # Diagonal direction (matching MATLAB)
    target_dir = Vector3D(-1.0, 1.0, -1.0)  # Different diagonal

    rotation_matrix = calculate_eye_rotation(rest_dir, target_dir)

    # MATLAB reference values
    expected_rotation_matrix = np.array([
        [0.3333333333333334, -0.6666666666666669, -0.6666666666666667],
        [0.6666666666666669, -0.3333333333333334, 0.6666666666666669],
        [-0.6666666666666667, -0.6666666666666669, 0.3333333333333334],
    ])

    np.testing.assert_allclose(rotation_matrix, expected_rotation_matrix, rtol=1e-14, atol=1e-15)

    # Verify rotation achieves target direction
    rest_normalized = np.array(rest_dir.normalize())
    target_normalized = np.array(target_dir.normalize())
    result_dir = rotation_matrix @ rest_normalized
    np.testing.assert_allclose(result_dir, target_normalized, rtol=1e-14, atol=1e-15)


def test_small_angle_rotation() -> None:
    """Test small angle rotation with MATLAB reference values."""
    rest_dir = Vector3D(0.0, 0.0, 1.0)  # Forward (matching MATLAB)
    target_dir = Vector3D(0.1, 0.0, 0.995)  # Slightly right
    target_dir = target_dir.normalize()  # Normalize

    rotation_matrix = calculate_eye_rotation(rest_dir, target_dir)

    # MATLAB reference values
    expected_rotation_matrix = np.array([
        [0.9949875627331983, 0.0, 0.0999987500234370],
        [0.0, 1.0, 0.0],
        [-0.0999987500234370, 0.0, 0.9949875627331983],
    ])

    np.testing.assert_allclose(rotation_matrix, expected_rotation_matrix, rtol=1e-14, atol=1e-15)

    # Verify rotation achieves target direction
    result_dir = rotation_matrix @ np.array(rest_dir)
    np.testing.assert_allclose(result_dir, np.array(target_dir), rtol=1e-14, atol=1e-15)


def test_no_rotation_same_direction() -> None:
    """Test case where no rotation is needed (same direction)."""
    rest_dir = Vector3D(0.0, 0.0, 1.0)  # Forward direction (matching MATLAB)
    target_dir = Vector3D(0.0, 0.0, 1.0)  # Same forward direction

    rotation_matrix = calculate_eye_rotation(rest_dir, target_dir)

    # Should be identity matrix
    expected_rotation_matrix = np.eye(3)
    np.testing.assert_allclose(rotation_matrix, expected_rotation_matrix, rtol=1e-14, atol=1e-15)


def test_output_properties() -> None:
    """Test that rotation matrices have proper mathematical properties."""
    test_cases = [
        (Vector3D(0, 0, -1), Vector3D(0.1, 0.2, -0.975)),  # Small rotation
        (Vector3D(0, 0, -1), Vector3D(0.3, 0.4, -0.8)),  # 3D rotation
        (Vector3D(1, 0, 0), Vector3D(0, 1, 0)),  # 90-degree rotation
    ]

    for rest_dir, target_dir in test_cases:
        normalize_rest_dir = rest_dir.normalize()
        normalize_target_dir = target_dir.normalize()

        rotation_matrix = calculate_eye_rotation(normalize_rest_dir, normalize_target_dir)

        # Check properties
        assert isinstance(rotation_matrix, np.ndarray)
        assert rotation_matrix.shape == (3, 3)
        assert rotation_matrix.dtype == np.float64

        # Check orthogonality (rotation_matrix @ rotation_matrix.T = I)
        np.testing.assert_allclose(rotation_matrix @ rotation_matrix.T, np.eye(3), rtol=1e-12, atol=1e-15)

        # Check determinant = 1 (proper rotation)
        det = np.linalg.det(rotation_matrix)
        np.testing.assert_allclose(det, 1.0, rtol=1e-12, atol=1e-15)

        # Check that it rotates normalize_rest_dir to normalize_target_dir
        result_dir = rotation_matrix @ np.array(normalize_rest_dir)
        np.testing.assert_allclose(result_dir, np.array(normalize_target_dir), rtol=1e-12, atol=1e-15)
