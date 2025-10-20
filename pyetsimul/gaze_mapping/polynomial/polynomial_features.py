"""Structured polynomial feature representation."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pyetsimul.geometry.plane_detection import PlaneInfo
    from pyetsimul.types.geometry import Point3D


@dataclass
class PolynomialFeatures:
    """Structured polynomial feature representation.

    Handles both same features (shared x,y) and different features (independent x,y)
    polynomial features used in polynomial gaze model algorithms. Encapsulates prediction logic.
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

    def predict(
        self, x_coefficients: np.ndarray, y_coefficients: np.ndarray, plane_info: "PlaneInfo | None" = None
    ) -> "Point3D":
        """Predict gaze coordinates using this polynomial's features.

        Args:
            x_coefficients: Calibration coefficients for first coordinate
            y_coefficients: Calibration coefficients for second coordinate
            plane_info: PlaneInfo object for coordinate reconstruction (optional, defaults to XZ plane)

        Returns:
            Point3D: Predicted gaze point in 3D coordinates

        """
        if self.uses_same_xy_features:
            coefficient_matrix = np.vstack([x_coefficients, y_coefficients])
            gaze_2d = coefficient_matrix @ self.features
        else:
            coord1_features, coord2_features = self._extract_coordinate_features()
            gaze_2d = np.array([x_coefficients @ coord1_features, y_coefficients @ coord2_features])

        # Reconstruct 3D point using plane information
        if plane_info is None:
            raise ValueError("plane_info is required for gaze prediction")
        return plane_info.reconstruct_3d_point(gaze_2d[0], gaze_2d[1])

    def _extract_coordinate_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Extract features for each coordinate, handling both array types."""
        if self.features.dtype == object:
            return self.features[0], self.features[1]
        return self.features[0, :], self.features[1, :]
