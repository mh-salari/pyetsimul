"""Glint noise simulation for eye tracking cameras."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from ..types import Point2D


@dataclass
class GlintNoiseConfig:
    """Configuration for glint detection noise simulation.

    Two modes available:

    Simple mode:
        noise_type: 'gaussian', 'uniform', or 'constant_offset'
        std: Standard deviation in pixels
        offset_x/offset_y: For constant offset

    Advanced mode:
        mean: [mean_x, mean_y] - bias vector in pixels
        covariance: [[var_x, cov_xy], [cov_xy, var_y]] - 2x2 covariance matrix

    If mean and covariance are provided, advanced mode is used.
    """

    # Simple interface
    noise_type: Optional[str] = None
    std: Optional[float] = None
    offset_x: Optional[float] = None
    offset_y: Optional[float] = None

    # Advanced interface
    mean: Optional[list[float]] = None
    covariance: Optional[list[list[float]]] = None

    # Common
    seed: Optional[int] = None

    def __post_init__(self):
        """Validate noise configuration after initialization."""
        # Check if advanced mode is used
        if self.mean is not None or self.covariance is not None:
            if self.mean is None or self.covariance is None:
                raise ValueError("Advanced mode requires both 'mean' and 'covariance' to be specified")
            if len(self.mean) != 2:
                raise ValueError("'mean' must be a 2-element list [mean_x, mean_y]")
            if len(self.covariance) != 2 or len(self.covariance[0]) != 2 or len(self.covariance[1]) != 2:
                raise ValueError("'covariance' must be a 2x2 matrix [[var_x, cov_xy], [cov_xy, var_y]]")
            # Validate covariance matrix is positive semidefinite
            cov_matrix = np.array(self.covariance)
            eigenvalues = np.linalg.eigvals(cov_matrix)
            if np.any(eigenvalues < 0):
                raise ValueError("Covariance matrix must be positive semidefinite")

            # Store numpy arrays
            self._mean_array = np.array(self.mean)
            self._cov_matrix = cov_matrix

            # Set advanced mode
            self.noise_type = "advanced"
            return

        # Simple mode validation
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

    if config.noise_type == "advanced":
        noise = np.random.multivariate_normal(config._mean_array, config._cov_matrix)
        return Point2D(x=glint_position.x + noise[0], y=glint_position.y + noise[1])

    elif config.noise_type == "gaussian":
        noise_x = np.random.normal(0., config.std)
        noise_y = np.random.normal(0., config.std)
        return Point2D(x=glint_position.x + noise_x, y=glint_position.y + noise_y)

    elif config.noise_type == "uniform":
        range_val = config.std * np.sqrt(3)
        noise_x = np.random.uniform(-range_val, range_val)
        noise_y = np.random.uniform(-range_val, range_val)
        return Point2D(x=glint_position.x + noise_x, y=glint_position.y + noise_y)

    elif config.noise_type == "constant_offset":
        return Point2D(x=glint_position.x + config.offset_x, y=glint_position.y + config.offset_y)

    return glint_position
