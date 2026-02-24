"""Unit tests for camera pan_tilt function."""

import numpy as np

from pyetsimul.core.camera import Camera
from pyetsimul.types import Position3D, TransformationMatrix


def test_basic_pan_tilt() -> None:
    """Test basic pan/tilt operation with MATLAB reference values."""
    c = Camera()

    # Target point as Position3D
    look_at = Position3D(x=10000, y=5000, z=-20000)

    # Apply pan/tilt
    c.pan_tilt(look_at)

    # MATLAB reference values
    expected_trans = np.array([
        [
            0.8944271909999159,
            -0.0975900072948533,
            -0.4364357804719848,
            0.0,
        ],
        [
            0.0,
            0.9759000729485332,
            -0.2182178902359924,
            0.0,
        ],
        [
            0.4472135954999580,
            0.1951800145897066,
            0.8728715609439696,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ])

    # Verify transformation matrix matches MATLAB exactly
    np.testing.assert_allclose(c.trans, expected_trans, rtol=1e-14, atol=1e-15)


def test_modified_rest_trans() -> None:
    """Test pan/tilt with camera that has been translated."""
    c = Camera()

    # Translate camera position
    position = Position3D(x=2000, y=1000, z=3000)
    translation_matrix = TransformationMatrix.from_translation(position)
    c.trans @= translation_matrix
    c.rest_trans = c.trans.copy()  # Update rest position

    # Target point as Position3D
    look_at = Position3D(x=8000, y=4000, z=-15000)

    # Apply pan/tilt
    c.pan_tilt(look_at)

    # MATLAB reference values
    expected_trans = np.array([
        [
            0.9486832980505138,
            -0.0493864798324795,
            -0.3123475237772121,
            2000,
        ],
        [
            0.0,
            0.9877295966495896,
            -0.1561737618886061,
            1000,
        ],
        [
            0.3162277660168379,
            0.1481594394974385,
            0.9370425713316364,
            3000,
        ],
        [
            0.0,
            0.0,
            0.0,
            1.0,
        ],
    ])

    # Verify transformation matrix matches MATLAB exactly
    np.testing.assert_allclose(c.trans, expected_trans, rtol=1e-14, atol=1e-15)


def test_output_properties() -> None:
    """Test that pan_tilt maintains proper camera properties."""
    c = Camera()
    original_focal_length = c.camera_matrix.focal_length
    target = Position3D(x=10000, y=5000, z=-20000)

    c.pan_tilt(target)

    # Should maintain camera properties
    assert c.camera_matrix.focal_length == original_focal_length
    assert c.trans.shape == (4, 4)
    assert c.trans.dtype == np.float64

    # Transformation matrix should be orthogonal
    rotation = c.orientation
    assert np.allclose(np.linalg.det(rotation), 1.0, rtol=1e-12)
    assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-12)

    # Bottom row should remain [0, 0, 0, 1]
    assert np.allclose(c.trans[3, :], [0, 0, 0, 1])
