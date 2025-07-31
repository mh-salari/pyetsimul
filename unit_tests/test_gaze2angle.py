"""Unit tests for gaze2angle function."""

import numpy as np
from et_simul.geometry.conversions import gaze2angle, angle2gaze


def test_round_trip_conversion():
    """Test that gaze->angle->gaze conversion preserves the original 4D vector."""
    observer_pos = np.array([0, 0.5, 0.2, 1])

    # Test a few key points from the grid
    test_points = [[-0.2, 0.05], [0.0, 0.2], [0.2, 0.35], [-0.1, 0.15]]

    for x, y in test_points:
        # Create a 4D homogeneous gaze vector
        target_point = np.array([x, 0, y, 1])
        gaze = target_point - observer_pos
        gaze /= np.linalg.norm(gaze)
        gaze[3] = 0  # Ensure it's a direction vector

        # Round-trip conversion
        angles = gaze2angle(gaze)
        gaze_new = angle2gaze(angles)

        # Check accuracy
        error = np.linalg.norm(gaze - gaze_new)
        assert error < 1e-8, f"Round-trip error {error:.2e} exceeds tolerance at ({x}, {y})"


def test_forward_gaze():
    """Test conversion of a forward-looking 4D homogeneous gaze vector."""
    forward_gaze = np.array([0, 0, -1, 0])
    angles = gaze2angle(forward_gaze)
    gaze_back = angle2gaze(angles)

    np.testing.assert_allclose(forward_gaze, gaze_back, atol=1e-10)


def test_side_gaze():
    """Test conversion of a side-looking 4D homogeneous gaze vector."""
    right_gaze = np.array([1, 0, 0, 0])
    angles = gaze2angle(right_gaze)
    gaze_back = angle2gaze(angles)

    np.testing.assert_allclose(right_gaze, gaze_back, atol=1e-10)


def test_grid_conversion_accuracy():
    """Test conversion accuracy over a representative grid subset."""
    observer_pos = np.array([0, 0.5, 0.2, 1])
    tolerance = 1e-8

    # Test smaller representative subset of the full grid
    x_values = [-0.2, -0.1, 0.0, 0.1, 0.2]
    y_values = [0.05, 0.15, 0.25, 0.35]

    max_error = 0.0
    for x in x_values:
        for y in y_values:
            target_point = np.array([x, 0, y, 1])
            gaze = target_point - observer_pos
            gaze /= np.linalg.norm(gaze)
            gaze[3] = 0  # Ensure it's a direction vector

            angles = gaze2angle(gaze)
            gaze_new = angle2gaze(angles)

            error = np.linalg.norm(gaze - gaze_new)
            max_error = max(max_error, error)

            assert error < tolerance, f"Error {error:.2e} exceeds tolerance at ({x}, {y})"

    assert max_error < tolerance


def test_output_properties():
    """Test that outputs have correct properties."""
    gaze = np.array([0.1, 0.2, -0.9, 0])
    gaze /= np.linalg.norm(gaze)
    gaze[3] = 0

    angles = gaze2angle(gaze)
    gaze_back = angle2gaze(angles)

    # Check types and shapes
    assert isinstance(angles, np.ndarray)
    assert angles.shape == (2,)
    assert isinstance(gaze_back, np.ndarray)
    assert gaze_back.shape == (4,)  # Should be 4D homogeneous vector

