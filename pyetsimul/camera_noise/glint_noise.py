"""Glint noise simulation for eye tracking cameras."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..types import Point2D


@dataclass
class GlintNoiseConfig:
    """Configuration for glint detection noise simulation.

    Args:
        noise_type: Noise distribution type ('gaussian', 'uniform', 'constant_offset')
        std: Standard deviation for glint position noise in pixels (required for 'gaussian'/'uniform')
        seed: Random seed for reproducible noise (optional)
        offset_x: Fixed offset in X direction (required for 'constant_offset')
        offset_y: Fixed offset in Y direction (required for 'constant_offset')
    """

    noise_type: Optional[str] = None
    std: Optional[float] = None
    seed: Optional[int] = None
    offset_x: Optional[float] = None
    offset_y: Optional[float] = None

    def __post_init__(self):
        """Validate noise configuration after initialization."""
        if self.noise_type is None:
            return

        if self.noise_type in ["gaussian", "uniform"]:
            if self.std is None:
                raise ValueError(f"'{self.noise_type}' noise type requires 'std' to be specified")
        elif self.noise_type == "constant_offset":
            if self.offset_x is None or self.offset_y is None:
                raise ValueError("'constant_offset' noise type requires 'offset_x' and 'offset_y' to be specified")
        else:
            raise ValueError(f"Unknown noise type: {self.noise_type}")


def apply_glint_noise(glint_position: Point2D, config: Optional[GlintNoiseConfig]) -> Point2D:
    """Add noise to glint position based on the provided configuration.

    Args:
        glint_position: Original glint position in pixels
        config: Noise configuration parameters. If None, no noise is applied.

    Returns:
        Point2D with added noise in pixel coordinates
    """
    if config is None or config.noise_type is None:
        return glint_position

    # Set random seed if specified for reproducible noise
    if config.seed is not None:
        np.random.seed(config.seed)

    if config.noise_type == "gaussian":
        noise_x = np.random.normal(0, config.std)
        noise_y = np.random.normal(0, config.std)
        return Point2D(x=glint_position.x + noise_x, y=glint_position.y + noise_y)

    elif config.noise_type == "uniform":
        range_val = config.std * np.sqrt(3)
        noise_x = np.random.uniform(-range_val, range_val)
        noise_y = np.random.uniform(-range_val, range_val)
        return Point2D(x=glint_position.x + noise_x, y=glint_position.y + noise_y)

    elif config.noise_type == "constant_offset":
        return Point2D(x=glint_position.x + config.offset_x, y=glint_position.y + config.offset_y)

    return glint_position
