"""Unit tests for gaze2angle function."""

import numpy as np
from pyetsimul.geometry.conversions import gaze2angle, angle2gaze
from pyetsimul.types import Point2D, Direction3D


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
        angles = gaze2angle(Direction3D.from_array(gaze))
        gaze_new = angle2gaze(angles)

        # Convert Direction3D back to homogeneous array for comparison
        gaze_new_array = np.array(gaze_new)

        # Check accuracy
        error = np.linalg.norm(gaze - gaze_new_array)
        assert error < 1e-8, f"Round-trip error {error:.2e} exceeds tolerance at ({x}, {y})"


def test_forward_gaze():
    """Test conversion of a forward-looking 4D homogeneous gaze vector."""
    forward_gaze = np.array([0, 0, -1, 0])
    angles = gaze2angle(Direction3D.from_array(forward_gaze))
    gaze_back = angle2gaze(angles)

    # Convert Direction3D back to homogeneous array for comparison
    gaze_back_array = np.array(gaze_back)
    np.testing.assert_allclose(forward_gaze, gaze_back_array, atol=1e-10)


def test_side_gaze():
    """Test conversion of a side-looking 4D homogeneous gaze vector."""
    right_gaze = np.array([1, 0, 0, 0])
    angles = gaze2angle(Direction3D.from_array(right_gaze))
    gaze_back = angle2gaze(angles)

    # Convert Direction3D back to homogeneous array for comparison
    gaze_back_array = np.array(gaze_back)
    np.testing.assert_allclose(right_gaze, gaze_back_array, atol=1e-10)


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

            angles = gaze2angle(Direction3D.from_array(gaze))
            gaze_new = angle2gaze(angles)

            # Convert Direction3D back to homogeneous array for comparison
            gaze_new_array = np.array(gaze_new)
            error = np.linalg.norm(gaze - gaze_new_array)
            max_error = max(max_error, float(error))

            assert error < tolerance, f"Error {error:.2e} exceeds tolerance at ({x}, {y})"

    assert max_error < tolerance


def test_output_properties():
    """Test that outputs have correct properties."""
    gaze = np.array([0.1, 0.2, -0.9, 0])
    gaze /= np.linalg.norm(gaze)
    gaze[3] = 0

    angles = gaze2angle(Direction3D.from_array(gaze))
    gaze_back = angle2gaze(angles)

    # Check types and properties
    assert isinstance(angles, Point2D)
    assert isinstance(gaze_back, Direction3D)

    # Check that we can convert back to arrays with correct shapes
    angles_array = np.array(angles)
    gaze_back_array = np.array(gaze_back)
    assert angles_array.shape == (2,)
    assert gaze_back_array.shape == (4,)  # Should be 4D homogeneous vector
