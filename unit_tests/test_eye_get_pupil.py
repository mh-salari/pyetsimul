"""Unit tests for Eye.get_pupil method."""

import numpy as np
from et_simul.core.eye import Eye


def test_default_n():
    """Test get_pupil with default N=20 and MATLAB reference values."""
    e = Eye()
    X = e.get_pupil()

    # MATLAB reference values for first 5 points
    expected_first_5 = np.array(
        [
            [
                0.0030000000000000,
                0.0000000000000000,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                0.0028531695488855,
                0.0009270509831248,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                0.0024270509831248,
                0.0017633557568774,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                0.0017633557568774,
                0.0024270509831248,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                0.0009270509831248,
                0.0028531695488855,
                -0.0087900000000000,
                1.0000000000000000,
            ],
        ]
    ).T

    # Test shape and first 5 points
    assert X.shape == (4, 20)
    np.testing.assert_allclose(X[:, :5], expected_first_5, rtol=1e-12)

    # Test homogeneous coordinates
    assert np.allclose(X[3, :], 1.0, rtol=1e-12)

    # Test circle properties
    pupil_center = e.pupil.pos_pupil[:3]
    radii = np.linalg.norm(X[:3, :] - pupil_center.reshape(-1, 1), axis=0)
    expected_radius = 0.0030000000000000
    assert np.allclose(radii, expected_radius, rtol=1e-12)


def test_custom_n_8():
    """Test get_pupil with N=8 and MATLAB reference values."""
    e = Eye()
    e.pupil.N = 8  # Set pupil resolution to 8
    X = e.get_pupil()

    # MATLAB reference values for all 8 points
    expected_points = np.array(
        [
            [
                0.0030000000000000,
                0.0000000000000000,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                0.0021213203435596,
                0.0021213203435596,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                0.0000000000000000,
                0.0030000000000000,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                -0.0021213203435596,
                0.0021213203435596,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                -0.0030000000000000,
                0.0000000000000000,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                -0.0021213203435596,
                -0.0021213203435596,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                -0.0000000000000000,
                -0.0030000000000000,
                -0.0087900000000000,
                1.0000000000000000,
            ],
            [
                0.0021213203435596,
                -0.0021213203435596,
                -0.0087900000000000,
                1.0000000000000000,
            ],
        ]
    ).T

    assert X.shape == (4, 8)
    np.testing.assert_allclose(X, expected_points, rtol=1e-12, atol=1e-15)

    # Test homogeneous coordinates
    assert np.allclose(X[3, :], 1.0, rtol=1e-12)


def test_custom_corneal_radius():
    """Test get_pupil with custom corneal radius and MATLAB reference values."""
    r_cornea_custom = 10e-3  # 10mm corneal radius
    e = Eye(r_cornea=r_cornea_custom)
    e.pupil.N = 12  # Set pupil resolution to 12
    X = e.get_pupil()

    # MATLAB reference values for scaled eye
    expected_points = np.array(
        [
            [
                0.0037593984962406,
                0.0000000000000000,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                0.0032557346006934,
                0.0018796992481203,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                0.0018796992481203,
                0.0032557346006934,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                0.0000000000000000,
                0.0037593984962406,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                -0.0018796992481203,
                0.0032557346006934,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                -0.0032557346006934,
                0.0018796992481203,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                -0.0037593984962406,
                0.0000000000000000,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                -0.0032557346006934,
                -0.0018796992481203,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                -0.0018796992481203,
                -0.0032557346006934,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                -0.0000000000000000,
                -0.0037593984962406,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                0.0018796992481203,
                -0.0032557346006934,
                -0.0110150375939850,
                1.0000000000000000,
            ],
            [
                0.0032557346006934,
                -0.0018796992481203,
                -0.0110150375939850,
                1.0000000000000000,
            ],
        ]
    ).T

    assert X.shape == (4, 12)
    np.testing.assert_allclose(X, expected_points, rtol=1e-12, atol=1e-15)


def test_custom_rest_position():
    """Test get_pupil with custom rest position and MATLAB reference values."""
    # Create eye with 45-degree rotation around Y-axis
    theta = 45 / 180 * np.pi
    custom_rest = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ]
    )
    e = Eye(r_cornea=7.98e-3)
    e.set_rest_orientation(custom_rest)
    e.pupil.N = 6  # Set pupil resolution to 6
    X = e.get_pupil()

    # MATLAB reference values for rotated eye
    expected_points = np.array(
        [
            [
                -0.0040941482630701,
                0.0000000000000000,
                -0.0083367889501894,
                1.0000000000000000,
            ],
            [
                -0.0051548084348499,
                0.0025980762113533,
                -0.0072761287784096,
                1.0000000000000000,
            ],
            [
                -0.0072761287784096,
                0.0025980762113533,
                -0.0051548084348499,
                1.0000000000000000,
            ],
            [
                -0.0083367889501894,
                0.0000000000000000,
                -0.0040941482630701,
                1.0000000000000000,
            ],
            [
                -0.0072761287784096,
                -0.0025980762113533,
                -0.0051548084348499,
                1.0000000000000000,
            ],
            [
                -0.0051548084348499,
                -0.0025980762113533,
                -0.0072761287784096,
                1.0000000000000000,
            ],
        ]
    ).T

    assert X.shape == (4, 6)
    np.testing.assert_allclose(X, expected_points, rtol=1e-12, atol=1e-15)


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye()
    e.pupil.N = 16  # Set pupil resolution to 16
    X = e.get_pupil()

    # Check types and shapes
    assert isinstance(X, np.ndarray)
    assert X.shape == (4, 16)
    assert X.dtype == np.float64

    # Test homogeneous coordinates
    assert np.allclose(X[3, :], 1.0, rtol=1e-12)

    # Test that points are finite
    assert np.all(np.isfinite(X))

    # Test circle properties
    pupil_center = e.pupil.pos_pupil[:3]
    radii = np.linalg.norm(X[:3, :] - pupil_center.reshape(-1, 1), axis=0)
    expected_radius = np.linalg.norm(e.pupil.x_pupil[:3])
    assert np.allclose(radii, expected_radius, rtol=1e-12)
