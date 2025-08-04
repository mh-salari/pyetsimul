"""Unit tests for SphericalCornea scaling behavior."""

import pytest
from et_simul.core.cornea import SphericalCornea
from et_simul.types import Position3D


class TestSphericalCorneaScaling:
    """Test scaling behavior of SphericalCornea parameters."""

    def test_default_scaling(self):
        """Test that default cornea has scale factor 1.0."""
        cornea = SphericalCornea()
        cornea.center = Position3D(x=0.0, y=0.0, z=-0.00435)  # Set center for testing

        assert cornea.get_scale_factor() == 1.0
        assert cornea.anterior_radius == 7.98e-3
        assert cornea.posterior_radius == 6.22e-3
        assert cornea.thickness_offset == 1.15e-3

    def test_custom_radius_scaling(self):
        """Test scaling with custom anterior radius."""
        custom_radius = 10e-3  # 10mm
        cornea = SphericalCornea(anterior_radius=custom_radius)
        cornea.center = Position3D(x=0.0, y=0.0, z=-0.005)  # Set center for testing

        expected_scale = custom_radius / 7.98e-3  # ~1.2531

        # Test scale factor
        assert abs(cornea.get_scale_factor() - expected_scale) < 1e-6

        # Test anterior radius (not scaled - it's the input)
        assert cornea.anterior_radius == custom_radius

        # Test posterior radius (should be scaled)
        expected_posterior = expected_scale * 6.22e-3
        assert abs(cornea.posterior_radius - expected_posterior) < 1e-9

        # Test thickness offset (should be scaled)
        expected_offset = expected_scale * 1.15e-3
        assert abs(cornea.thickness_offset - expected_offset) < 1e-9

    def test_small_radius_scaling(self):
        """Test scaling with smaller anterior radius."""
        custom_radius = 6.98e-3  # 6.98mm (from eye_anatomy example)
        cornea = SphericalCornea(anterior_radius=custom_radius)
        cornea.center = Position3D(x=0.0, y=0.0, z=-0.004)  # Set center for testing

        expected_scale = custom_radius / 7.98e-3  # ~0.8747

        # Test scale factor
        assert abs(cornea.get_scale_factor() - expected_scale) < 1e-6

        # Test scaled parameters
        expected_posterior = expected_scale * 6.22e-3
        expected_offset = expected_scale * 1.15e-3

        assert abs(cornea.posterior_radius - expected_posterior) < 1e-9
        assert abs(cornea.thickness_offset - expected_offset) < 1e-9

        # Verify specific values for regression testing
        assert abs(cornea.posterior_radius * 1000 - 5.441) < 0.001  # 5.441mm
        assert abs(cornea.thickness_offset * 1000 - 1.006) < 0.001  # 1.006mm

    def test_thickness_calculation(self):
        """Test that thickness calculation uses scaled parameters."""
        # Default cornea
        default_cornea = SphericalCornea()
        default_thickness = abs(7.98e-3 - 6.22e-3 - 1.15e-3)
        assert abs(default_cornea.thickness - default_thickness) < 1e-9

        # Custom cornea
        custom_radius = 10e-3
        custom_cornea = SphericalCornea(anterior_radius=custom_radius)
        scale = custom_radius / 7.98e-3
        expected_thickness = abs(custom_radius - scale * 6.22e-3 - scale * 1.15e-3)
        assert abs(custom_cornea.thickness - expected_thickness) < 1e-9

    def test_posterior_center_scaling(self):
        """Test that posterior center calculation uses scaled parameters."""
        custom_radius = 8.5e-3
        cornea = SphericalCornea(anterior_radius=custom_radius)
        cornea.center = Position3D(x=0.0, y=0.0, z=-0.005)

        # Calculate expected posterior center
        scale = custom_radius / 7.98e-3
        expected_posterior_radius = scale * 6.22e-3
        expected_thickness_offset = scale * 1.15e-3
        thickness_term = custom_radius - expected_posterior_radius - expected_thickness_offset

        # Expected posterior center position
        expected_posterior = Position3D(
            x=0.0,  # x stays same
            y=0.0,  # y stays same
            z=-0.005 - thickness_term,  # z shifts by thickness
        )

        actual_posterior = cornea.get_posterior_center()
        actual_posterior.assert_close(expected_posterior, rtol=1e-12)

    def test_corneal_depth_scaling(self):
        """Test that corneal depth scales with anterior radius."""
        # Default cornea
        default_cornea = SphericalCornea()
        assert abs(default_cornea.get_corneal_depth() - 3.54e-3) < 1e-9

        # Custom cornea
        custom_radius = 9e-3
        custom_cornea = SphericalCornea(anterior_radius=custom_radius)
        expected_scale = custom_radius / 7.98e-3
        expected_depth = expected_scale * 3.54e-3
        assert abs(custom_cornea.get_corneal_depth() - expected_depth) < 1e-9

    def test_apex_position_scaling(self):
        """Test that apex position calculation works with any radius."""
        custom_radius = 11e-3
        cornea = SphericalCornea(anterior_radius=custom_radius)
        cornea.center = Position3D(x=0.0, y=0.0, z=-0.006)

        # Expected apex position (same x,y, but z shifted by radius)
        expected_apex = Position3D(
            x=0.0,  # x stays same
            y=0.0,  # y stays same
            z=-0.006 - custom_radius,  # z shifts by radius
        )

        actual_apex = cornea.get_apex_position()
        actual_apex.assert_close(expected_apex, rtol=1e-12)

    def test_scaling_consistency(self):
        """Test that all scaled parameters maintain proper relationships."""
        custom_radius = 7.5e-3
        cornea = SphericalCornea(anterior_radius=custom_radius)

        scale = cornea.get_scale_factor()

        # All scaled parameters should use the same scale factor
        expected_posterior = scale * 6.22e-3
        expected_offset = scale * 1.15e-3
        expected_depth = scale * 3.54e-3

        assert abs(cornea.posterior_radius - expected_posterior) < 1e-9
        assert abs(cornea.thickness_offset - expected_offset) < 1e-9
        assert abs(cornea.get_corneal_depth() - expected_depth) < 1e-9

    def test_reference_values(self):
        """Test that reference values are preserved."""
        cornea = SphericalCornea()

        # Check that reference values match Boff and Lincoln [1988] constants
        assert cornea._r_cornea_default == 7.98e-3
        assert cornea._posterior_radius_default == 6.22e-3
        assert cornea._thickness_offset_default == 1.15e-3
        assert cornea._cornea_depth_default == 3.54e-3
        assert cornea._cornea_center_to_rotation_center_default == 10.20e-3

    @pytest.mark.parametrize("radius", [5e-3, 6.5e-3, 7.98e-3, 9e-3, 12e-3])
    def test_multiple_radii(self, radius):
        """Test scaling with various corneal radii."""
        cornea = SphericalCornea(anterior_radius=radius)
        cornea.center = Position3D(x=0.0, y=0.0, z=-0.005)

        expected_scale = radius / 7.98e-3

        # Test that scale factor is correct
        assert abs(cornea.get_scale_factor() - expected_scale) < 1e-10

        # Test that all parameters scale consistently
        assert abs(cornea.posterior_radius - expected_scale * 6.22e-3) < 1e-12
        assert abs(cornea.thickness_offset - expected_scale * 1.15e-3) < 1e-12
        assert abs(cornea.get_corneal_depth() - expected_scale * 3.54e-3) < 1e-12

        # Test anatomical constraint: anterior > posterior
        assert cornea.anterior_radius > cornea.posterior_radius

        # Test that thickness is positive
        assert cornea.thickness > 0
