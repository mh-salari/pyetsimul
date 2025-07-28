"""Unit tests for gaze2angle function."""

import numpy as np
from et_simul.geometry.conversions import gaze2angle, angle2gaze


def test_round_trip_conversion():
    """Test that gaze->angle->gaze conversion preserves original vector."""
    observer_pos = np.array([0, 0.5, 0.2, 1])

    # Test a few key points from the grid
    test_points = [[-0.2, 0.05], [0.0, 0.2], [0.2, 0.35], [-0.1, 0.15]]

    for x, y in test_points:
        # Create gaze vector
        target_point = np.array([x, 0, y, 1])
        gaze = target_point - observer_pos
        gaze = gaze / np.linalg.norm(gaze)

        # Round-trip conversion
        angles = gaze2angle(gaze)
        gaze_new = angle2gaze(angles)

        # Check accuracy
        error = np.linalg.norm(gaze[:3] - gaze_new[:3])
        assert error < 1e-8, f"Round-trip error {error:.2e} exceeds tolerance at ({x}, {y})"


def test_forward_gaze():
    """Test conversion of forward-looking gaze."""
    forward_gaze = np.array([0, 0, -1])
    angles = gaze2angle(forward_gaze)
    gaze_back = angle2gaze(angles)

    np.testing.assert_allclose(forward_gaze, gaze_back[:3], atol=1e-10)


def test_side_gaze():
    """Test conversion of side-looking gaze."""
    right_gaze = np.array([1, 0, 0])
    angles = gaze2angle(right_gaze)
    gaze_back = angle2gaze(angles)

    np.testing.assert_allclose(right_gaze, gaze_back[:3], atol=1e-10)


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
            gaze = gaze / np.linalg.norm(gaze)

            angles = gaze2angle(gaze)
            gaze_new = angle2gaze(angles)

            error = np.linalg.norm(gaze[:3] - gaze_new[:3])
            max_error = max(max_error, error)

            assert error < tolerance, f"Error {error:.2e} exceeds tolerance at ({x}, {y})"

    assert max_error < tolerance


def test_output_properties():
    """Test that outputs have correct properties."""
    gaze = np.array([0.1, 0.2, -0.9])
    gaze = gaze / np.linalg.norm(gaze)

    angles = gaze2angle(gaze)
    gaze_back = angle2gaze(angles)

    # Check types and shapes
    assert isinstance(angles, (list, np.ndarray))
    assert len(angles) == 2
    assert isinstance(gaze_back, np.ndarray)
    assert len(gaze_back) in [3, 4]  # May include homogeneous coordinate
