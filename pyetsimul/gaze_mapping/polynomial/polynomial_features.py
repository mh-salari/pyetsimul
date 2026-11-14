"""Structured polynomial feature representation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class PolynomialFeatures:
    """Structured polynomial feature representation.

    Handles both same features (shared x,y) and different features (independent x,y)
    polynomial features used in polynomial gaze model algorithms.
    """

    features: np.ndarray  # Feature array (1D or 2D)
    polynomial_name: str  # Name of polynomial that generated these features

    @property
    def uses_same_xy_features(self) -> bool:
        """Check if polynomial uses same features for both X and Y coordinates."""
        return self.features.ndim == 1 and self.features.dtype != object

    @property
    def uses_different_xy_features(self) -> bool:
        """Check if polynomial uses different features for X and Y coordinates."""
        return not self.uses_same_xy_features

    @property
    def feature_count(self) -> int:
        """Total number of features."""
        if self.uses_same_xy_features:
            return len(self.features)
        if self.features.dtype == object:
            return sum(len(coord_features) for coord_features in self.features)
        return self.features.shape[0] * self.features.shape[1]
