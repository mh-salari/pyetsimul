"""Unit tests for Eye.point_within_cornea method."""

import numpy as np
from et_simul.core.eye import Eye
from et_simul.core.cornea import SphericalCornea


def test_basic_cases():
    """Test basic cases with MATLAB reference values."""
    e = Eye()

    # Case 1: Point at apex (should be within)
    p1 = e.pos_apex
    expected_p1 = np.array([0.000000, 0.000000, -0.012330, 1.000000])
    np.testing.assert_allclose(p1, expected_p1, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p1)

    # Case 2: Point at cornea center (should be outside)
    p2 = e.cornea.center
    expected_p2 = np.array([0.000000, 0.000000, -0.004350, 1.000000])
    np.testing.assert_allclose(p2, expected_p2, rtol=1e-6, atol=1e-6)
    assert not e.point_within_cornea(p2)

    # Case 3: Point beyond cornea depth (should be outside)
    direction = (e.cornea.center - e.pos_apex) / np.linalg.norm(e.cornea.center - e.pos_apex)
    p3 = e.pos_apex + 2 * e.depth_cornea * direction
    expected_p3 = np.array([0.000000, 0.000000, -0.005250, 1.000000])
    np.testing.assert_allclose(p3, expected_p3, rtol=1e-6, atol=1e-6)
    assert not e.point_within_cornea(p3)


def test_boundary_cases():
    """Test boundary and edge cases with MATLAB reference values."""
    e = Eye()

    # Case 4: Point just inside boundary (should be within)
    direction = (e.cornea.center - e.pos_apex) / np.linalg.norm(e.cornea.center - e.pos_apex)
    p4 = e.pos_apex + (e.depth_cornea - 1e-6) * direction
    expected_p4 = np.array([0.000000, 0.000000, -0.008791, 1.000000])
    np.testing.assert_allclose(p4, expected_p4, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p4)

    # Case 6: Point perpendicular to cornea direction (should be within)
    perp_direction = np.array([1, 0, 0, 0])
    p6 = e.pos_apex + 0.001 * perp_direction
    expected_p6 = np.array([0.001000, 0.000000, -0.012330, 1.000000])
    np.testing.assert_allclose(p6, expected_p6, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p6)

    # Case 7: Point behind apex (should be within)
    direction = (e.cornea.center - e.pos_apex) / np.linalg.norm(e.cornea.center - e.pos_apex)
    p7 = e.pos_apex - 0.5 * e.depth_cornea * direction
    expected_p7 = np.array([0.000000, 0.000000, -0.014100, 1.000000])
    np.testing.assert_allclose(p7, expected_p7, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p7)


def test_custom_eye():
    """Test with custom eye parameters and MATLAB reference values."""
    # Case 5: Custom eye with different cornea radius
    r_cornea_custom = 9e-3
    # Create custom cornea with different radius (center=None for auto-positioning)
    custom_cornea = SphericalCornea(radius=r_cornea_custom)
    e5 = Eye(cornea=custom_cornea)
    p5 = e5.pos_apex
    expected_p5 = np.array([0.000000, 0.000000, -0.013906, 1.000000])
    np.testing.assert_allclose(p5, expected_p5, rtol=1e-6, atol=1e-6)
    assert e5.point_within_cornea(p5)

    # Verify custom radius was applied
    assert np.isclose(e5.cornea.radius, r_cornea_custom)


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye()
    result = e.point_within_cornea(e.pos_apex)

    # Check return type
    assert isinstance(result, (bool, np.bool_))

    # Test consistency across multiple calls
    result2 = e.point_within_cornea(e.pos_apex)
    assert result == result2
