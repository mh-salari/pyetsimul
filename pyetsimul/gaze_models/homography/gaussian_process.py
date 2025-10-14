"""Gaussian Process error correction for homography normalization.

This module provides a wrapper around scikit-learn's GaussianProcessRegressor
to implement the error correction model described in Hansen et al. (2010).
"""

import warnings

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel


class GaussianProcessErrorCorrection:
    """Gaussian Process error correction using scikit-learn.

    Models systematic errors by wrapping scikit-learn's GaussianProcessRegressor.
    Uses a squared exponential kernel (RBF) and a noise kernel (WhiteKernel).
    """

    def __init__(
        self,
        length_scale: float = 0.1,
        noise_level: float = 0.001,
        length_scale_bounds: tuple[float, float] = (0.01, 0.5),
        noise_level_bounds: tuple[float, float] = (1e-5, 0.01),
    ) -> None:
        """Initialize the GP error correction model.

        Args:
            length_scale: Initial length scale of the RBF kernel (meters).
                         Default 0.1 (100mm) is suitable for screen-scale coordinates.
            noise_level: Initial noise level (meters).
                        Default 0.001 (1mm) matches typical calibration error scales.
            length_scale_bounds: Min/max bounds for length scale optimization (meters).
                                Default (0.01, 0.5) = 10mm to 500mm.
            noise_level_bounds: Min/max bounds for noise level optimization (meters).
                               Default (1e-5, 0.01) = 0.01mm to 10mm.

        """
        kernel = RBF(length_scale=length_scale, length_scale_bounds=length_scale_bounds) + WhiteKernel(
            noise_level=noise_level, noise_level_bounds=noise_level_bounds
        )

        # Multi-output regression is handled by fitting one regressor per output dimension.
        self.gp_x = GaussianProcessRegressor(kernel=kernel, random_state=0)
        self.gp_y = GaussianProcessRegressor(kernel=kernel, random_state=0)

    def fit(self, X: "np.ndarray", y: "np.ndarray") -> None:
        """Fit the GP model to the calibration residuals."""
        # Suppress ConvergenceWarning, which is common when the optimizer
        # hits the bounds of the hyperparameter search space.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=ConvergenceWarning)
            self.gp_x.fit(X, y[:, 0])
            self.gp_y.fit(X, y[:, 1])

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        """Predict the error correction for new gaze points.

        Args:
            X: Mx2 array of query screen positions.

        Returns:
            Mx2 array of predicted error vectors.

        """
        pred_x = self.gp_x.predict(X)
        pred_y = self.gp_y.predict(X)
        return np.vstack([pred_x, pred_y]).T
