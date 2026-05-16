"""Unit tests for glint noise functionality."""

import pytest

from pyetsimul.camera_noise import GlintNoiseConfig, apply_glint_noise
from pyetsimul.types import Point2D


def test_no_noise() -> None:
    """Test that no noise is applied when config is None or noise_type is None."""
    original_pos = Point2D(10.0, 20.0)

    # Test with config=None
    pos_after_none_config = apply_glint_noise(original_pos, None)
    assert original_pos == pos_after_none_config

    # Test with noise_type=None
    config = GlintNoiseConfig(noise_type=None)
    pos_after_none_noise_type = apply_glint_noise(original_pos, config)
    assert original_pos == pos_after_none_noise_type


def test_gaussian_noise() -> None:
    """Test that Gaussian noise is applied."""
    original_pos = Point2D(10.0, 20.0)
    config = GlintNoiseConfig(noise_type="gaussian", std=1.0, seed=42)

    noisy_pos = apply_glint_noise(original_pos, config)

    assert original_pos.x != noisy_pos.x
    assert original_pos.y != noisy_pos.y


def test_uniform_noise() -> None:
    """Test that uniform noise is applied."""
    original_pos = Point2D(10.0, 20.0)
    config = GlintNoiseConfig(noise_type="uniform", std=1.0, seed=42)

    noisy_pos = apply_glint_noise(original_pos, config)

    assert original_pos.x != noisy_pos.x
    assert original_pos.y != noisy_pos.y


def test_constant_offset_noise() -> None:
    """Test that constant offset noise is applied correctly."""
    original_pos = Point2D(10.0, 20.0)
    config = GlintNoiseConfig(noise_type="constant_offset", offset_x=5.0, offset_y=-5.0)

    noisy_pos = apply_glint_noise(original_pos, config)

    assert abs(noisy_pos.x - 15.0) < 1e-7
    assert abs(noisy_pos.y - 15.0) < 1e-7


def test_config_validation() -> None:
    """Test that GlintNoiseConfig raises errors for invalid configurations."""
    with pytest.raises(ValueError, match=r"requires 'std' to be specified"):
        GlintNoiseConfig(noise_type="gaussian")  # Missing std

    with pytest.raises(ValueError, match=r"requires 'std' to be specified"):
        GlintNoiseConfig(noise_type="uniform")  # Missing std

    with pytest.raises(ValueError, match=r"requires 'offset_x' and 'offset_y'"):
        GlintNoiseConfig(noise_type="constant_offset", offset_x=1.0)  # Missing offset_y

    with pytest.raises(ValueError, match=r"requires 'offset_x' and 'offset_y'"):
        GlintNoiseConfig(noise_type="constant_offset", offset_y=1.0)  # Missing offset_x

    with pytest.raises(ValueError, match=r"Unknown noise type"):
        GlintNoiseConfig(noise_type="unknown_noise")  # Unknown noise type


def test_advanced_noise() -> None:
    """Test advanced noise mode with mean and covariance."""
    original_pos = Point2D(10.0, 20.0)
    config = GlintNoiseConfig(mean=[1.0, -1.0], covariance=[[2.0, 0.5], [0.5, 1.5]], seed=42)

    # Should automatically set noise_type to 'advanced'
    assert config.noise_type == "advanced"

    noisy_pos = apply_glint_noise(original_pos, config)

    assert original_pos.x != noisy_pos.x
    assert original_pos.y != noisy_pos.y


def test_advanced_mode_validation() -> None:
    """Test validation for advanced mode parameters."""
    # Missing covariance
    with pytest.raises(ValueError, match=r"requires both 'mean' and 'covariance'"):
        GlintNoiseConfig(mean=[0.0, 0.0])

    # Missing mean
    with pytest.raises(ValueError, match=r"requires both 'mean' and 'covariance'"):
        GlintNoiseConfig(covariance=[[1.0, 0.0], [0.0, 1.0]])

    # Wrong mean size
    with pytest.raises(ValueError, match=r"'mean' must be a 2-element list"):
        GlintNoiseConfig(mean=[0.0], covariance=[[1.0, 0.0], [0.0, 1.0]])

    # Wrong covariance size
    with pytest.raises(ValueError, match=r"'covariance' must be a 2x2 matrix"):
        GlintNoiseConfig(mean=[0.0, 0.0], covariance=[[1.0]])

    # Negative definite covariance
    with pytest.raises(ValueError, match=r"positive semidefinite"):
        GlintNoiseConfig(mean=[0.0, 0.0], covariance=[[-1.0, 0.0], [0.0, 1.0]])
