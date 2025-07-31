"""Unit tests for camera pan_tilt function."""

import numpy as np
from et_simul.core.camera import Camera


def test_basic_pan_tilt():
    """Test basic pan/tilt operation with MATLAB reference values."""
    c = Camera()

    # Target point (4D homogeneous coordinates)
    look_at = np.array([10.0, 5.0, -20.0, 1.0])

    # Apply pan/tilt
    c.pan_tilt(look_at)

    # MATLAB reference values
    expected_trans = np.array(
        [
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
        ]
    )

    # Verify transformation matrix matches MATLAB exactly
    np.testing.assert_allclose(c.trans, expected_trans, rtol=1e-14, atol=1e-15)


def test_modified_rest_trans():
    """Test pan/tilt with camera that has been translated."""
    c = Camera()

    # Translate camera position
    translation_matrix = np.eye(4)
    translation_matrix[:3, 3] = [2.0, 1.0, 3.0]  # Move camera
    c.trans = c.trans @ translation_matrix
    c.rest_trans = c.trans.copy()  # Update rest position

    # Target point (4D homogeneous coordinates)
    look_at = np.array([8.0, 4.0, -15.0, 1.0])

    # Apply pan/tilt
    c.pan_tilt(look_at)

    # MATLAB reference values
    expected_trans = np.array(
        [
            [
                0.9486832980505138,
                -0.0493864798324795,
                -0.3123475237772121,
                2.0,
            ],
            [
                0.0,
                0.9877295966495896,
                -0.1561737618886061,
                1.0,
            ],
            [
                0.3162277660168379,
                0.1481594394974385,
                0.9370425713316364,
                3.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
            ],
        ]
    )

    # Verify transformation matrix matches MATLAB exactly
    np.testing.assert_allclose(c.trans, expected_trans, rtol=1e-14, atol=1e-15)


def test_output_properties():
    """Test that pan_tilt maintains proper camera properties."""
    c = Camera()
    original_focal_length = c.focal_length
    target = np.array([10.0, 5.0, -20.0, 1.0])

    c.pan_tilt(target)

    # Should maintain camera properties
    assert c.focal_length == original_focal_length
    assert c.trans.shape == (4, 4)
    assert c.trans.dtype == np.float64

    # Transformation matrix should be orthogonal
    rotation = c.orientation
    assert np.allclose(np.linalg.det(rotation), 1.0, rtol=1e-12)
    assert np.allclose(rotation @ rotation.T, np.eye(3), rtol=1e-12)

    # Bottom row should remain [0, 0, 0, 1]
    assert np.allclose(c.trans[3, :], [0, 0, 0, 1])
