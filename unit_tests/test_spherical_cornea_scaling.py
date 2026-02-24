"""Unit tests for SphericalCornea scaling behavior."""

import pytest

from pyetsimul.core.cornea import SphericalCornea
from pyetsimul.types import Position3D


def test_default_scaling() -> None:
    """Test that default cornea has scale factor 1.0."""
    cornea = SphericalCornea()
    cornea.center = Position3D(x=0.0, y=0.0, z=-4.35)  # Set center for testing

    assert cornea.get_scale_factor() == 1.0
    assert cornea.anterior_radius == 7.98
    assert cornea.posterior_radius == 6.22
    assert cornea.thickness_offset == 1.15


def test_custom_radius_scaling() -> None:
    """Test scaling with custom anterior radius."""
    custom_radius = 10  # 10mm
    cornea = SphericalCornea(anterior_radius=custom_radius)
    cornea.center = Position3D(x=0.0, y=0.0, z=-5)  # Set center for testing

    expected_scale = custom_radius / 7.98  # ~1.2531

    # Test scale factor
    assert abs(cornea.get_scale_factor() - expected_scale) < 1e-6

    # Test anterior radius (not scaled - it's the input)
    assert cornea.anterior_radius == custom_radius

    # Test posterior radius (should be scaled)
    expected_posterior = expected_scale * 6.22
    assert abs(cornea.posterior_radius - expected_posterior) < 1e-9

    # Test thickness offset (should be scaled)
    expected_offset = expected_scale * 1.15
    assert abs(cornea.thickness_offset - expected_offset) < 1e-9


def test_small_radius_scaling() -> None:
    """Test scaling with smaller anterior radius."""
    custom_radius = 6.98  # 6.98mm (from eye_anatomy example)
    cornea = SphericalCornea(anterior_radius=custom_radius)
    cornea.center = Position3D(x=0.0, y=0.0, z=-4)  # Set center for testing

    expected_scale = custom_radius / 7.98  # ~0.8747

    # Test scale factor
    assert abs(cornea.get_scale_factor() - expected_scale) < 1e-6

    # Test scaled parameters
    expected_posterior = expected_scale * 6.22
    expected_offset = expected_scale * 1.15

    assert abs(cornea.posterior_radius - expected_posterior) < 1e-9
    assert abs(cornea.thickness_offset - expected_offset) < 1e-9

    # Verify specific values for regression testing
    assert abs(cornea.posterior_radius - 5.441) < 0.001  # 5.441mm
    assert abs(cornea.thickness_offset - 1.006) < 0.001  # 1.006mm


def test_thickness_calculation() -> None:
    """Test that thickness calculation uses scaled parameters."""
    # Default cornea
    default_cornea = SphericalCornea()
    default_thickness = abs(7.98 - 6.22 - 1.15)
    assert abs(default_cornea.thickness - default_thickness) < 1e-9

    # Custom cornea
    custom_radius = 10
    custom_cornea = SphericalCornea(anterior_radius=custom_radius)
    scale = custom_radius / 7.98
    expected_thickness = abs(custom_radius - scale * 6.22 - scale * 1.15)
    assert abs(custom_cornea.thickness - expected_thickness) < 1e-9


def test_posterior_center_scaling() -> None:
    """Test that posterior center calculation uses scaled parameters."""
    custom_radius = 8.5
    cornea = SphericalCornea(anterior_radius=custom_radius)
    cornea.center = Position3D(x=0.0, y=0.0, z=-5)

    # Calculate expected posterior center
    scale = custom_radius / 7.98
    expected_posterior_radius = scale * 6.22
    expected_thickness_offset = scale * 1.15
    thickness_term = custom_radius - expected_posterior_radius - expected_thickness_offset

    # Expected posterior center position
    expected_posterior = Position3D(
        x=0.0,  # x stays same
        y=0.0,  # y stays same
        z=-5 - thickness_term,  # z shifts by thickness
    )

    actual_posterior = cornea.get_posterior_center()
    actual_posterior.assert_close(expected_posterior, rtol=1e-12)


def test_corneal_depth_scaling() -> None:
    """Test that corneal depth scales with anterior radius."""
    # Default cornea
    default_cornea = SphericalCornea()
    assert abs(default_cornea.get_corneal_depth() - 3.54) < 1e-9

    # Custom cornea
    custom_radius = 9
    custom_cornea = SphericalCornea(anterior_radius=custom_radius)
    expected_scale = custom_radius / 7.98
    expected_depth = expected_scale * 3.54
    assert abs(custom_cornea.get_corneal_depth() - expected_depth) < 1e-9


def test_apex_position_scaling() -> None:
    """Test that apex position calculation works with any radius."""
    custom_radius = 11
    cornea = SphericalCornea(anterior_radius=custom_radius)
    cornea.center = Position3D(x=0.0, y=0.0, z=-6)

    # Expected apex position (same x,y, but z shifted by radius)
    expected_apex = Position3D(
        x=0.0,  # x stays same
        y=0.0,  # y stays same
        z=-6 - custom_radius,  # z shifts by radius
    )

    actual_apex = cornea.get_apex_position()
    actual_apex.assert_close(expected_apex, rtol=1e-12)


def test_scaling_consistency() -> None:
    """Test that all scaled parameters maintain proper relationships."""
    custom_radius = 7.5
    cornea = SphericalCornea(anterior_radius=custom_radius)

    scale = cornea.get_scale_factor()

    # All scaled parameters should use the same scale factor
    expected_posterior = scale * 6.22
    expected_offset = scale * 1.15
    expected_depth = scale * 3.54

    assert abs(cornea.posterior_radius - expected_posterior) < 1e-9
    assert abs(cornea.thickness_offset - expected_offset) < 1e-9
    assert abs(cornea.get_corneal_depth() - expected_depth) < 1e-9


def test_reference_values() -> None:
    """Test that reference values are preserved."""
    cornea = SphericalCornea()

    # Check that reference values match Boff and Lincoln [1988] constants
    assert cornea._r_cornea_default == 7.98  # noqa: SLF001
    assert cornea._posterior_radius_default == 6.22  # noqa: SLF001
    assert cornea._thickness_offset_default == 1.15  # noqa: SLF001
    assert cornea._cornea_depth_default == 3.54  # noqa: SLF001
    assert cornea._cornea_center_to_rotation_center_default == 10.20  # noqa: SLF001


@pytest.mark.parametrize("radius", [5, 6.5, 7.98, 9, 12])
def test_multiple_radii(radius: float) -> None:
    """Test scaling with various corneal radii."""
    cornea = SphericalCornea(anterior_radius=radius)
    cornea.center = Position3D(x=0.0, y=0.0, z=-5)

    expected_scale = radius / 7.98

    # Test that scale factor is correct
    assert abs(cornea.get_scale_factor() - expected_scale) < 1e-10

    # Test that all parameters scale consistently
    assert abs(cornea.posterior_radius - expected_scale * 6.22) < 1e-12
    assert abs(cornea.thickness_offset - expected_scale * 1.15) < 1e-12
    assert abs(cornea.get_corneal_depth() - expected_scale * 3.54) < 1e-12

    # Test anatomical constraint: anterior > posterior
    assert cornea.anterior_radius > cornea.posterior_radius

    # Test that thickness is positive
    assert cornea.thickness > 0
