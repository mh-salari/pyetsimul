"""Unit tests for glint noise functionality."""

import numpy as np
import unittest
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, Point2D
from pyetsimul.camera_noise import GlintNoiseConfig, apply_glint_noise


class TestGlintNoise(unittest.TestCase):
    """Test glint noise functionality in camera."""

    def setUp(self):
        """Set up test eye, light, and camera."""
        # Create eye (same as example.py)
        rotation_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        self.eye = Eye(fovea_displacement=False)
        self.eye.set_rest_orientation(rotation_matrix)
        self.eye.position = Position3D(0, 500e-3, 200e-3)

        # Create light
        self.light = Light(position=Position3D(200e-3, 0, 0))

    def test_no_noise_deterministic(self):
        """Test that camera with no noise produces identical glint positions."""
        camera = Camera()
        camera.point_at(self.eye.position)

        positions = []
        for _ in range(10):
            image = camera.take_image(self.eye, [self.light])
            if image.corneal_reflections[0] is not None:
                cr = image.corneal_reflections[0]
                positions.append((cr.x, cr.y))

        # All positions should be identical
        self.assertGreater(len(positions), 0, "Should detect glint")
        first_pos = positions[0]
        for pos in positions:
            self.assertAlmostEqual(pos[0], first_pos[0], places=6)
            self.assertAlmostEqual(pos[1], first_pos[1], places=6)

    def test_gaussian_noise_distribution(self):
        """Test that Gaussian noise produces appropriate statistics."""
        noise_std = 1.0
        camera = Camera(glint_noise_config=GlintNoiseConfig(std=noise_std, noise_type="gaussian"))
        camera.point_at(self.eye.position)

        # Get reference position (no noise)
        ref_camera = Camera()
        ref_camera.point_at(self.eye.position)
        ref_image = ref_camera.take_image(self.eye, [self.light])
        ref_pos = ref_image.corneal_reflections[0]

        # Collect noisy positions
        positions = []
        for _ in range(100):
            image = camera.take_image(self.eye, [self.light])
            if image.corneal_reflections[0] is not None:
                cr = image.corneal_reflections[0]
                positions.append((cr.x, cr.y))

        self.assertGreater(len(positions), 50, "Should detect glint in most trials")

        # Calculate statistics
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]
        x_mean, x_std = np.mean(x_coords), np.std(x_coords)
        y_mean, y_std = np.mean(y_coords), np.std(y_coords)

        # Mean should be close to reference position
        self.assertAlmostEqual(x_mean, ref_pos.x, delta=0.2)
        self.assertAlmostEqual(y_mean, ref_pos.y, delta=0.2)

        # Standard deviation should be close to specified noise
        self.assertAlmostEqual(x_std, noise_std, delta=0.2)
        self.assertAlmostEqual(y_std, noise_std, delta=0.2)

    def test_uniform_noise_distribution(self):
        """Test that uniform noise produces appropriate range."""
        noise_std = 1.0
        camera = Camera(glint_noise_config=GlintNoiseConfig(std=noise_std, noise_type="uniform"))
        camera.point_at(self.eye.position)

        # Collect noisy positions
        positions = []
        for _ in range(100):
            image = camera.take_image(self.eye, [self.light])
            if image.corneal_reflections[0] is not None:
                cr = image.corneal_reflections[0]
                positions.append((cr.x, cr.y))

        self.assertGreater(len(positions), 50, "Should detect glint in most trials")

        # Check that positions are within expected range
        expected_range = noise_std * np.sqrt(3)  # uniform range for matching variance
        x_coords = [pos[0] for pos in positions]
        y_coords = [pos[1] for pos in positions]

        x_range = max(x_coords) - min(x_coords)
        y_range = max(y_coords) - min(y_coords)

        # Range should be reasonable (not too small, not too large)
        self.assertGreater(x_range, expected_range * 0.5)
        self.assertLess(x_range, expected_range * 2.5)
        self.assertGreater(y_range, expected_range * 0.5)
        self.assertLess(y_range, expected_range * 2.5)

    def test_invalid_noise_type(self):
        """Test that invalid noise type falls back to no noise."""
        camera = Camera(glint_noise_config=GlintNoiseConfig(std=1.0, noise_type="invalid"))
        camera.point_at(self.eye.position)

        positions = []
        for _ in range(5):
            image = camera.take_image(self.eye, [self.light])
            if image.corneal_reflections[0] is not None:
                cr = image.corneal_reflections[0]
                positions.append((cr.x, cr.y))

        # Should behave like no noise (deterministic)
        if len(positions) > 1:
            first_pos = positions[0]
            for pos in positions:
                self.assertAlmostEqual(pos[0], first_pos[0], places=6)
                self.assertAlmostEqual(pos[1], first_pos[1], places=6)

    def test_apply_glint_noise_function(self):
        """Test the apply_glint_noise function directly."""
        original_pos = Point2D(100.0, 200.0)

        # Test no noise
        config_no_noise = GlintNoiseConfig(std=0.0)
        noisy_pos = apply_glint_noise(original_pos, config_no_noise)
        self.assertEqual(noisy_pos.x, original_pos.x)
        self.assertEqual(noisy_pos.y, original_pos.y)

        # Test with noise
        config_with_noise = GlintNoiseConfig(std=1.0, noise_type="gaussian")
        noisy_pos = apply_glint_noise(original_pos, config_with_noise)
        # Should be different (with very high probability)
        self.assertNotEqual(noisy_pos.x, original_pos.x)
        self.assertNotEqual(noisy_pos.y, original_pos.y)


if __name__ == "__main__":
    unittest.main()
