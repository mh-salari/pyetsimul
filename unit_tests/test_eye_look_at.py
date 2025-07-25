"""Unit tests for eye look_at function."""

import numpy as np
from et_simul.core.eye import Eye


def test_look_at_with_foveal_displacement():
    """Test look_at with foveal displacement enabled (default)."""
    e = Eye()
    target = np.array([12, -7, -30])
    initial_position = e.position.copy()

    e.look_at(target)

    # MATLAB reference result with foveal displacement
    expected_matrix = np.array(
        [
            [0.8880588066153353, 0.0529342540962807, -0.4566722246165374],
            [0.0657514591983408, 0.9685125121295987, 0.2401255077281524],
            [0.4550036281242554, -0.2432724369748543, 0.8566155612654254],
        ]
    )

    # Position should remain unchanged
    np.testing.assert_allclose(e.position, initial_position, rtol=1e-12)

    # Orientation matrix should match MATLAB exactly
    np.testing.assert_allclose(e.orientation, expected_matrix, rtol=1e-12)


def test_look_at_without_foveal_displacement():
    """Test look_at with foveal displacement disabled."""
    e = Eye(fovea_displacement=False)
    target = np.array([12, -7, -30])
    initial_position = e.position.copy()

    e.look_at(target)

    # MATLAB reference result without foveal displacement
    expected_matrix = np.array(
        [
            [0.9309291734097576, 0.0402913155109749, -0.3629703624113421],
            [0.0402913155109748, 0.9764967326185980, 0.2117327114066163],
            [0.3629703624113422, -0.2117327114066163, 0.9074259060283554],
        ]
    )

    # Position should remain unchanged
    np.testing.assert_allclose(e.position, initial_position, rtol=1e-12)

    # Orientation matrix should match MATLAB exactly
    np.testing.assert_allclose(e.orientation, expected_matrix, rtol=1e-12)


def test_output_properties():
    """Test that look_at output has correct properties."""
    e = Eye()
    target = np.array([10, 5, -30])

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
