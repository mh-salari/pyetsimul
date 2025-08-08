"""Unit tests for Eye.find_refracted_position method."""

import numpy as np
from pyetsimul.core.eye import Eye
from pyetsimul.types.geometry import Position3D


def test_basic_refraction():
    """Test basic refraction scenario with MATLAB reference values."""
    e = Eye()

    # Camera position (50mm in front, slightly offset)
    C = Position3D(1.0, 0.5, 50.0)

    # Object position inside eye (slightly off-center)
    O = Position3D(0.5, 0.2, -4.0)

    I = e.find_refracted_position(C, O)

    # MATLAB reference values
    expected_I = np.array(
        [
            0.0036221057296969,
            0.0014867621985687,
            -0.0113034371535094,
            1.0,
        ]
    )

    assert I is not None
    arr = np.append(np.array(I), 1.0)
    np.testing.assert_allclose(arr, expected_I, rtol=1e-10, atol=1e-12)


def test_close_camera():
    """Test refraction with camera close to eye and MATLAB reference values."""
    e = Eye()

    # Camera very close (15mm instead of 50mm)
    C = Position3D(1.0, 0.5, 15.0)

    # Object position inside eye
    O = Position3D(0.5, 0.2, -4.0)

    I = e.find_refracted_position(C, O)

    # MATLAB reference values
    expected_I = np.array(
        [
            0.0042042117071795,
            0.0017993479882829,
            -0.0108896751248267,
            1.0,
        ]
    )

    assert I is not None
    arr = np.append(np.array(I), 1.0)
    np.testing.assert_allclose(arr, expected_I, rtol=1e-10, atol=1e-12)


def test_refraction_impossible_geometry():
    """Test case where refraction is impossible - should return None."""
    e = Eye()

    # Put object outside the eye sphere entirely (beyond cornea radius)
    # This causes the underlying find_refraction to succeed but point_within_cornea to fail
    C = Position3D(1.0, 0.5, 50.0)  # Normal camera position
    O = Position3D(0.0, 0.0, 5.0)  # Object in front of eye (outside)

    I = e.find_refracted_position(C, O)

    # Should return None when refraction is impossible
    assert I is None


def test_output_properties():
    """Test that output has correct properties when valid."""
    e = Eye()
    C = Position3D(1.0, 0.5, 50.0)
    O = Position3D(0.5, 0.2, -4.0)

    I = e.find_refracted_position(C, O)
    assert I is not None, "I should not be None for these inputs"

    # Check types and shapes
    arr = np.append(np.array(I), 1.0)
    assert arr.shape == (4,)
    assert arr[3] == 1.0  # Homogeneous coordinate
