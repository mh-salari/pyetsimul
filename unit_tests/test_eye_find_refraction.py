"""Unit tests for Eye.find_refracted_position method."""

import numpy as np

from pyetsimul.core.eye import Eye
from pyetsimul.types.geometry import Position3D


def test_basic_refraction() -> None:
    """Test basic refraction scenario with MATLAB reference values."""
    e = Eye()

    # Camera position (50mm in front, slightly offset)
    center_point = Position3D(1.0, 0.5, 50.0)

    # Object position inside eye (slightly off-center)
    origin_point = Position3D(0.5, 0.2, -4.0)

    intersection_point = e.find_refracted_position(center_point, origin_point)

    # MATLAB reference values
    expected_i = np.array([
        0.0036221057296969,
        0.0014867621985687,
        -0.0113034371535094,
        1.0,
    ])

    assert intersection_point is not None
    np.testing.assert_allclose(intersection_point, expected_i, rtol=1e-10, atol=1e-12)


def test_close_camera() -> None:
    """Test refraction with camera close to eye and MATLAB reference values."""
    e = Eye()

    # Camera very close (15mm instead of 50mm)
    center_point = Position3D(1.0, 0.5, 15.0)

    # Object position inside eye
    origin_point = Position3D(0.5, 0.2, -4.0)

    intersection_point = e.find_refracted_position(center_point, origin_point)

    # MATLAB reference values
    expected_i = np.array([
        0.0042042117071795,
        0.0017993479882829,
        -0.0108896751248267,
        1.0,
    ])

    assert intersection_point is not None
    np.testing.assert_allclose(intersection_point, expected_i, rtol=1e-10, atol=1e-12)


def test_refraction_impossible_geometry() -> None:
    """Test case where refraction is impossible - should return None."""
    e = Eye()

    # Put object outside the eye sphere entirely (beyond cornea radius)
    # This causes the underlying find_refraction to succeed but point_within_cornea to fail
    center_point = Position3D(1.0, 0.5, 50.0)  # Normal camera position
    origin_point = Position3D(0.0, 0.0, 5.0)  # Object in front of eye (outside)

    intersection_point = e.find_refracted_position(center_point, origin_point)

    # Should return None when refraction is impossible
    assert intersection_point is None


def test_output_properties() -> None:
    """Test that output has correct properties when valid."""
    e = Eye()
    center_point = Position3D(1.0, 0.5, 50.0)
    origin_point = Position3D(0.5, 0.2, -4.0)

    intersection_point = e.find_refracted_position(center_point, origin_point)
    assert isinstance(intersection_point, Position3D), "intersection_point should not be None for these inputs"
