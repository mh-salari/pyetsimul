"""Unit tests for Eye.point_within_cornea method."""

import numpy as np
from et_simul.core.eye import Eye
from et_simul.core.cornea import SphericalCornea
from et_simul.types.geometry import Position3D, Vector3D


def test_basic_cases():
    """Test basic cases with MATLAB reference values."""
    e = Eye()

    # Case 1: Point at apex (should be within)
    p1 = e.cornea.get_apex_position()
    expected_p1 = Position3D(0.0, 0.0, -0.012330)
    p1.assert_close(expected_p1, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p1)

    # Case 2: Point at cornea center (should be outside)
    p2 = e.cornea.center
    expected_p2 = Position3D(0.0, 0.0, -0.004350)
    p2.assert_close(expected_p2, rtol=1e-6, atol=1e-6)
    assert not e.point_within_cornea(p2)

    # Case 3: Point beyond cornea depth (should be outside)
    apex = e.cornea.get_apex_position()
    center = e.cornea.center
    direction = Vector3D(center.x - apex.x, center.y - apex.y, center.z - apex.z).normalize()
    p3 = Position3D(
        apex.x + 2 * e.cornea.get_corneal_depth() * direction.x,
        apex.y + 2 * e.cornea.get_corneal_depth() * direction.y,
        apex.z + 2 * e.cornea.get_corneal_depth() * direction.z,
    )
    expected_p3 = Position3D(0.0, 0.0, -0.005250)
    p3.assert_close(expected_p3, rtol=1e-6, atol=1e-6)
    assert not e.point_within_cornea(p3)


def test_boundary_cases():
    """Test boundary and edge cases with MATLAB reference values."""
    e = Eye()

    # Case 4: Point just inside boundary (should be within)
    apex = e.cornea.get_apex_position()
    center = e.cornea.center
    direction = Vector3D(center.x - apex.x, center.y - apex.y, center.z - apex.z).normalize()
    p4 = Position3D(
        apex.x + (e.cornea.get_corneal_depth() - 1e-6) * direction.x,
        apex.y + (e.cornea.get_corneal_depth() - 1e-6) * direction.y,
        apex.z + (e.cornea.get_corneal_depth() - 1e-6) * direction.z,
    )
    expected_p4 = Position3D(0.0, 0.0, -0.008791)
    p4.assert_close(expected_p4, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p4)

    # Case 6: Point perpendicular to cornea direction (should be within)
    perp_direction = Vector3D(1, 0, 0)
    apex = e.cornea.get_apex_position()
    p6 = Position3D(
        apex.x + 0.001 * perp_direction.x, apex.y + 0.001 * perp_direction.y, apex.z + 0.001 * perp_direction.z
    )
    expected_p6 = Position3D(0.001000, 0.0, -0.012330)
    p6.assert_close(expected_p6, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p6)

    # Case 7: Point behind apex (should be within)
    apex = e.cornea.get_apex_position()
    center = e.cornea.center
    direction = Vector3D(center.x - apex.x, center.y - apex.y, center.z - apex.z).normalize()
    p7 = Position3D(
        apex.x - 0.5 * e.cornea.get_corneal_depth() * direction.x,
        apex.y - 0.5 * e.cornea.get_corneal_depth() * direction.y,
        apex.z - 0.5 * e.cornea.get_corneal_depth() * direction.z,
    )
    expected_p7 = Position3D(0.0, 0.0, -0.014100)
    p7.assert_close(expected_p7, rtol=1e-6, atol=1e-6)
    assert e.point_within_cornea(p7)


def test_custom_eye():
    """Test with custom eye parameters and MATLAB reference values."""
    # Case 5: Custom eye with different cornea radius
    r_cornea_custom = 9e-3
    # Create custom cornea with different radius (center=None for auto-positioning)
    custom_cornea = SphericalCornea(anterior_radius=r_cornea_custom)
    e5 = Eye(cornea=custom_cornea)
    p5 = e5.cornea.get_apex_position()
    expected_p5 = Position3D(0.0, 0.0, -0.013906)
    p5.assert_close(expected_p5, rtol=1e-6, atol=1e-6)
    assert e5.point_within_cornea(p5)

    # Verify custom radius was applied
    assert np.isclose(e5.cornea.anterior_radius, r_cornea_custom)


def test_output_properties():
    """Test that output has correct properties."""
    e = Eye()
    result = e.point_within_cornea(e.cornea.get_apex_position())

    # Check return type
    assert isinstance(result, (bool, np.bool_))

    # Test consistency across multiple calls
    result2 = e.point_within_cornea(e.cornea.get_apex_position())
    assert result == result2
