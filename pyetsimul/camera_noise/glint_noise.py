"""Glint noise simulation for eye tracking cameras."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..types import Point2D


@dataclass
class GlintNoiseConfig:
    """Configuration for glint detection noise simulation.

    Args:
        std: Standard deviation for glint position noise in pixels (0.0 = disabled)
        noise_type: Noise distribution type ('gaussian' or 'uniform')
        seed: Random seed for reproducible noise (None = random)
    """

    std: float = 0.0
    noise_type: str = "gaussian"
    seed: Optional[int] = None


def apply_glint_noise(glint_position: Point2D, config: GlintNoiseConfig) -> Point2D:
    """Add random noise to glint position.

    Args:
        glint_position: Original glint position in pixels
        config: Noise configuration parameters

    Returns:
        Point2D with added noise in pixel coordinates
    """
    if config.std <= 0.0:
        return glint_position

    # Set random seed if specified for reproducible noise
    if config.seed is not None:
        np.random.seed(config.seed)

    if config.noise_type == "gaussian":
        noise_x = np.random.normal(0, config.std)
        noise_y = np.random.normal(0, config.std)
    elif config.noise_type == "uniform":
        # For uniform distribution, use range of ±sqrt(3)*std to match variance
        range_val = config.std * np.sqrt(3)
        noise_x = np.random.uniform(-range_val, range_val)
        noise_y = np.random.uniform(-range_val, range_val)
    else:
        return glint_position

    return Point2D(x=glint_position.x + noise_x, y=glint_position.y + noise_y)
