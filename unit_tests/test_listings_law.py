"""Unit tests for listings_law function."""

import numpy as np
from et_simul.core.eye import Eye


def test_3d_rotation():
    """Test complex 3D rotation with MATLAB reference values."""
    rest_dir = np.array([1.0, 1.0, 1.0])  # Diagonal direction (matching MATLAB)
    target_dir = np.array([-1.0, 1.0, -1.0])  # Different diagonal

    R = Eye.listings_law(rest_dir, target_dir)

    # MATLAB reference values
    expected_R = np.array(
        [
            [0.3333333333333334, -0.6666666666666669, -0.6666666666666667],
            [0.6666666666666669, -0.3333333333333334, 0.6666666666666669],
            [-0.6666666666666667, -0.6666666666666669, 0.3333333333333334],
        ]
    )

    np.testing.assert_allclose(R, expected_R, rtol=1e-14, atol=1e-15)

    # Verify rotation achieves target direction
    rest_normalized = rest_dir / np.linalg.norm(rest_dir)
    target_normalized = target_dir / np.linalg.norm(target_dir)
    result_dir = R @ rest_normalized
    np.testing.assert_allclose(result_dir, target_normalized, rtol=1e-14, atol=1e-15)


def test_small_angle_rotation():
    """Test small angle rotation with MATLAB reference values."""
    rest_dir = np.array([0.0, 0.0, 1.0])  # Forward (matching MATLAB)
    target_dir = np.array([0.1, 0.0, 0.995])  # Slightly right
    target_dir = target_dir / np.linalg.norm(target_dir)  # Normalize

    R = Eye.listings_law(rest_dir, target_dir)

    # MATLAB reference values
    expected_R = np.array(
        [
            [0.9949875627331983, 0.0, 0.0999987500234370],
            [0.0, 1.0, 0.0],
            [-0.0999987500234370, 0.0, 0.9949875627331983],
        ]
    )

    np.testing.assert_allclose(R, expected_R, rtol=1e-14, atol=1e-15)

    # Verify rotation achieves target direction
    result_dir = R @ rest_dir
    np.testing.assert_allclose(result_dir, target_dir, rtol=1e-14, atol=1e-15)


def test_no_rotation_same_direction():
    """Test case where no rotation is needed (same direction)."""
    rest_dir = np.array([0.0, 0.0, 1.0])  # Forward direction (matching MATLAB)
    target_dir = np.array([0.0, 0.0, 1.0])  # Same forward direction

    R = Eye.listings_law(rest_dir, target_dir)

    # Should be identity matrix
    expected_R = np.eye(3)
    np.testing.assert_allclose(R, expected_R, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that rotation matrices have proper mathematical properties."""
    test_cases = [
        ([0, 0, -1], [0.1, 0.2, -0.975]),  # Small rotation
        ([0, 0, -1], [0.3, 0.4, -0.8]),  # 3D rotation
        ([1, 0, 0], [0, 1, 0]),  # 90-degree rotation
    ]

    for rest_dir, target_dir in test_cases:
        rest_dir = np.array(rest_dir) / np.linalg.norm(rest_dir)
        target_dir = np.array(target_dir) / np.linalg.norm(target_dir)

        R = Eye.listings_law(rest_dir, target_dir)

        # Check properties
        assert isinstance(R, np.ndarray)
        assert R.shape == (3, 3)
        assert R.dtype == np.float64

        # Check orthogonality (R @ R.T = I)
        np.testing.assert_allclose(R @ R.T, np.eye(3), rtol=1e-12, atol=1e-15)

        # Check determinant = 1 (proper rotation)
        det = np.linalg.det(R)
        np.testing.assert_allclose(det, 1.0, rtol=1e-12, atol=1e-15)

        # Check that it rotates rest_dir to target_dir
        result_dir = R @ rest_dir
        np.testing.assert_allclose(result_dir, target_dir, rtol=1e-12, atol=1e-15)
