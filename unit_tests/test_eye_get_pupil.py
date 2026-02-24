"""Unit tests for Eye.get_pupil method."""

import numpy as np

from pyetsimul.core.cornea import SphericalCornea
from pyetsimul.core.eye import Eye


def test_default_n() -> None:
    """Test get_pupil with default N=20 and MATLAB reference values."""
    e = Eye()
    pupil_boundary_points = e.get_pupil().boundary_points

    # MATLAB reference values for first 5 points
    expected_first_5 = np.array([
        [
            3.0,
            0.0,
            -8.79,
            1.0,
        ],
        [
            2.8531695488855,
            0.9270509831248,
            -8.79,
            1.0,
        ],
        [
            2.4270509831248,
            1.7633557568774,
            -8.79,
            1.0,
        ],
        [
            1.7633557568774,
            2.4270509831248,
            -8.79,
            1.0,
        ],
        [
            0.9270509831248,
            2.8531695488855,
            -8.79,
            1.0,
        ],
    ]).T

    # Test shape and first 5 points
    assert pupil_boundary_points.shape == (4, 20)
    np.testing.assert_allclose(pupil_boundary_points[:, :5], expected_first_5, rtol=1e-12)

    # Test homogeneous coordinates
    assert np.allclose(pupil_boundary_points[3, :], 1.0, rtol=1e-12)

    # Test circle properties
    pupil_center = np.array(e.get_pupil_center_in_world())[:3]
    radii = np.linalg.norm(pupil_boundary_points[:3, :] - pupil_center.reshape(-1, 1), axis=0)
    expected_radius = 3.0
    assert np.allclose(radii, expected_radius, rtol=1e-12)


def test_custom_n_8() -> None:
    """Test get_pupil with N=8 and MATLAB reference values."""
    e = Eye()
    e.pupil.n = 8  # Set pupil resolution to 8
    pupil_boundary_points = e.get_pupil().boundary_points

    # MATLAB reference values for all 8 points
    expected_points = np.array([
        [
            3.0,
            0.0,
            -8.79,
            1.0,
        ],
        [
            2.1213203435596,
            2.1213203435596,
            -8.79,
            1.0,
        ],
        [
            0.0,
            3.0,
            -8.79,
            1.0,
        ],
        [
            -2.1213203435596,
            2.1213203435596,
            -8.79,
            1.0,
        ],
        [
            -3.0,
            0.0,
            -8.79,
            1.0,
        ],
        [
            -2.1213203435596,
            -2.1213203435596,
            -8.79,
            1.0,
        ],
        [
            -0.0,
            -3.0,
            -8.79,
            1.0,
        ],
        [
            2.1213203435596,
            -2.1213203435596,
            -8.79,
            1.0,
        ],
    ]).T

    assert pupil_boundary_points.shape == (4, 8)
    np.testing.assert_allclose(pupil_boundary_points, expected_points, rtol=1e-12, atol=1e-15)

    # Test homogeneous coordinates
    assert np.allclose(pupil_boundary_points[3, :], 1.0, rtol=1e-12)


def test_custom_corneal_radius() -> None:
    """Test get_pupil with custom corneal radius and MATLAB reference values."""
    r_cornea_custom = 10  # 10mm corneal radius
    e = Eye(cornea=SphericalCornea(anterior_radius=r_cornea_custom))
    e.pupil.n = 12  # Set pupil resolution to 12
    pupil_boundary_points = e.get_pupil().boundary_points

    # MATLAB reference values for scaled eye
    expected_points = np.array([
        [
            3.7593984962406,
            0.0,
            -11.0150375939850,
            1.0,
        ],
        [
            3.2557346006934,
            1.8796992481203,
            -11.0150375939850,
            1.0,
        ],
        [
            1.8796992481203,
            3.2557346006934,
            -11.0150375939850,
            1.0,
        ],
        [
            0.0,
            3.7593984962406,
            -11.0150375939850,
            1.0,
        ],
        [
            -1.8796992481203,
            3.2557346006934,
            -11.0150375939850,
            1.0,
        ],
        [
            -3.2557346006934,
            1.8796992481203,
            -11.0150375939850,
            1.0,
        ],
        [
            -3.7593984962406,
            0.0,
            -11.0150375939850,
            1.0,
        ],
        [
            -3.2557346006934,
            -1.8796992481203,
            -11.0150375939850,
            1.0,
        ],
        [
            -1.8796992481203,
            -3.2557346006934,
            -11.0150375939850,
            1.0,
        ],
        [
            -0.0,
            -3.7593984962406,
            -11.0150375939850,
            1.0,
        ],
        [
            1.8796992481203,
            -3.2557346006934,
            -11.0150375939850,
            1.0,
        ],
        [
            3.2557346006934,
            -1.8796992481203,
            -11.0150375939850,
            1.0,
        ],
    ]).T

    assert pupil_boundary_points.shape == (4, 12)
    np.testing.assert_allclose(pupil_boundary_points, expected_points, rtol=1e-12, atol=1e-15)


def test_custom_rest_position() -> None:
    """Test get_pupil with custom rest position and MATLAB reference values."""
    # Create eye with 45-degree rotation around Y-axis
    theta = 45 / 180 * np.pi
    custom_rest = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)],
    ])
    e = Eye(cornea=SphericalCornea(anterior_radius=7.98))
    e.set_rest_orientation(custom_rest)
    e.pupil.n = 6  # Set pupil resolution to 6
    pupil_boundary_points = e.get_pupil().boundary_points

    # MATLAB reference values for rotated eye
    expected_points = np.array([
        [
            -4.0941482630701,
            0.0,
            -8.3367889501894,
            1.0,
        ],
        [
            -5.1548084348499,
            2.5980762113533,
            -7.2761287784096,
            1.0,
        ],
        [
            -7.2761287784096,
            2.5980762113533,
            -5.1548084348499,
            1.0,
        ],
        [
            -8.3367889501894,
            0.0,
            -4.0941482630701,
            1.0,
        ],
        [
            -7.2761287784096,
            -2.5980762113533,
            -5.1548084348499,
            1.0,
        ],
        [
            -5.1548084348499,
            -2.5980762113533,
            -7.2761287784096,
            1.0,
        ],
    ]).T

    assert pupil_boundary_points.shape == (4, 6)
    np.testing.assert_allclose(pupil_boundary_points, expected_points, rtol=1e-12, atol=1e-15)


def test_output_properties() -> None:
    """Test that output has correct properties."""
    e = Eye()
    e.pupil.n = 16  # Set pupil resolution to 16
    pupil_boundary_points = e.get_pupil().boundary_points

    # Check types and shapes
    assert isinstance(pupil_boundary_points, np.ndarray)
    assert pupil_boundary_points.shape == (4, 16)
    assert pupil_boundary_points.dtype == np.float64

    # Test homogeneous coordinates
    assert np.allclose(pupil_boundary_points[3, :], 1.0, rtol=1e-12)

    # Test that points are finite
    assert np.all(np.isfinite(pupil_boundary_points))

    # Test circle properties
    pupil_center = np.array(e.get_pupil_center_in_world())[:3]
    radii = np.linalg.norm(pupil_boundary_points[:3, :] - pupil_center.reshape(-1, 1), axis=0)
    expected_radius = np.linalg.norm(np.array(e.pupil.x_pupil)[:3])
    assert np.allclose(radii, expected_radius, rtol=1e-12)
