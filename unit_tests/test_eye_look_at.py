"""Unit tests for eye look_at function."""

import numpy as np
import numpy.testing as npt
from et_simul.core.eye import Eye
from et_simul.types import Position3D, RotationMatrix


def test_look_at_with_foveal_displacement():
    """Test look_at with foveal displacement enabled (default)."""
    e = Eye()
    target = Position3D(x=12.0, y=-7.0, z=-30.0)
    initial_position = np.array(e.position)

    e.look_at(target)

    # MATLAB reference result with foveal displacement
    expected_matrix: RotationMatrix = np.array(
        [
            [0.8880588066153353, 0.0529342540962807, -0.4566722246165374],
            [0.0657514591983408, 0.9685125121295987, 0.2401255077281524],
            [0.4550036281242554, -0.2432724369748543, 0.8566155612654254],
        ],
        dtype=np.float64,
    )

    # Position should remain unchanged
    e.position.assert_close(Position3D.from_array(initial_position), rtol=1e-12)

    # Orientation matrix should match MATLAB exactly
    npt.assert_allclose(e.orientation, expected_matrix, rtol=1e-12)


def test_look_at_with_no_foveal_displacement():
    """Test look_at with foveal displacement disabled."""
    e = Eye(fovea_displacement=False)  # Disable foveal displacement
    target = Position3D(x=12.0, y=-7.0, z=-30.0)
    initial_position = np.array(e.position)

    e.look_at(target)

    # MATLAB reference result without foveal displacement (verified with debug_look_at.m)
    expected_matrix: RotationMatrix = np.array(
        [
            [0.93092917340976, 0.04029131551097, -0.36297036241134],
            [0.04029131551097, 0.97649673261860, 0.21173271140662],
            [0.36297036241134, -0.21173271140662, 0.90742590602836],
        ],
        dtype=np.float64,
    )

    # Position should remain unchanged
    e.position.assert_close(Position3D.from_array(initial_position), rtol=1e-12)

    # Orientation matrix should match MATLAB exactly
    npt.assert_allclose(e.orientation, expected_matrix, rtol=1e-12)


def test_output_properties():
    """Test that look_at output has correct properties."""
    e = Eye()
    target = Position3D(x=10.0, y=5.0, z=-30.0)

    # look_at doesn't return anything (void method)
    e.look_at(target)

    # Should have proper transformation matrix properties
    assert e.trans.shape == (4, 4)
    assert e.trans.dtype == np.float64

    # Rotation matrix should be orthogonal with determinant 1
    rotation = e.orientation
    assert np.allclose(np.linalg.det(rotation), 1.0, rtol=1e-12)
    assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-12)

    # Bottom row should be [0, 0, 0, 1]
    assert np.allclose(e.trans[3, :], [0, 0, 0, 1])
