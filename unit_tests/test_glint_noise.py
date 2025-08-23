"""Unit tests for glint noise functionality."""

import unittest
from pyetsimul.camera_noise import GlintNoiseConfig, apply_glint_noise
from pyetsimul.types import Point2D


class TestGlintNoise(unittest.TestCase):
    """Simplified tests for glint noise."""

    def test_no_noise(self):
        """Test that no noise is applied when config is None or noise_type is None."""
        original_pos = Point2D(10.0, 20.0)

        # Test with config=None
        pos_after_none_config = apply_glint_noise(original_pos, None)
        self.assertEqual(original_pos, pos_after_none_config)

        # Test with noise_type=None
        config = GlintNoiseConfig(noise_type=None)
        pos_after_none_noise_type = apply_glint_noise(original_pos, config)
        self.assertEqual(original_pos, pos_after_none_noise_type)

    def test_gaussian_noise(self):
        """Test that Gaussian noise is applied."""
        original_pos = Point2D(10.0, 20.0)
        config = GlintNoiseConfig(noise_type="gaussian", std=1.0, seed=42)

        noisy_pos = apply_glint_noise(original_pos, config)

        self.assertNotEqual(original_pos.x, noisy_pos.x)
        self.assertNotEqual(original_pos.y, noisy_pos.y)

    def test_uniform_noise(self):
        """Test that uniform noise is applied."""
        original_pos = Point2D(10.0, 20.0)
        config = GlintNoiseConfig(noise_type="uniform", std=1.0, seed=42)

        noisy_pos = apply_glint_noise(original_pos, config)

        self.assertNotEqual(original_pos.x, noisy_pos.x)
        self.assertNotEqual(original_pos.y, noisy_pos.y)

    def test_constant_offset_noise(self):
        """Test that constant offset noise is applied correctly."""
        original_pos = Point2D(10.0, 20.0)
        config = GlintNoiseConfig(noise_type="constant_offset", offset_x=5.0, offset_y=-5.0)

        noisy_pos = apply_glint_noise(original_pos, config)

        self.assertAlmostEqual(noisy_pos.x, 15.0)
        self.assertAlmostEqual(noisy_pos.y, 15.0)

    def test_config_validation(self):
        """Test that GlintNoiseConfig raises errors for invalid configurations."""
        with self.assertRaises(ValueError):
            GlintNoiseConfig(noise_type="gaussian")  # Missing std

        with self.assertRaises(ValueError):
            GlintNoiseConfig(noise_type="uniform")  # Missing std

        with self.assertRaises(ValueError):
            GlintNoiseConfig(noise_type="constant_offset", offset_x=1.0)  # Missing offset_y

        with self.assertRaises(ValueError):
            GlintNoiseConfig(noise_type="constant_offset", offset_y=1.0)  # Missing offset_x

        with self.assertRaises(ValueError):
            GlintNoiseConfig(noise_type="unknown_noise")  # Unknown noise type

    def test_advanced_noise(self):
        """Test advanced noise mode with mean and covariance."""
        original_pos = Point2D(10.0, 20.0)
        config = GlintNoiseConfig(mean=[1.0, -1.0], covariance=[[2.0, 0.5], [0.5, 1.5]], seed=42)

        # Should automatically set noise_type to 'advanced'
        self.assertEqual(config.noise_type, "advanced")

        noisy_pos = apply_glint_noise(original_pos, config)

        self.assertNotEqual(original_pos.x, noisy_pos.x)
        self.assertNotEqual(original_pos.y, noisy_pos.y)

    def test_advanced_mode_validation(self):
        """Test validation for advanced mode parameters."""
        # Missing covariance
        with self.assertRaises(ValueError):
            GlintNoiseConfig(mean=[0.0, 0.0])

        # Missing mean
        with self.assertRaises(ValueError):
            GlintNoiseConfig(covariance=[[1.0, 0.0], [0.0, 1.0]])

        # Wrong mean size
        with self.assertRaises(ValueError):
            GlintNoiseConfig(mean=[0.0], covariance=[[1.0, 0.0], [0.0, 1.0]])

        # Wrong covariance size
        with self.assertRaises(ValueError):
            GlintNoiseConfig(mean=[0.0, 0.0], covariance=[[1.0]])

        # Negative definite covariance
        with self.assertRaises(ValueError):
            GlintNoiseConfig(mean=[0.0, 0.0], covariance=[[-1.0, 0.0], [0.0, 1.0]])
