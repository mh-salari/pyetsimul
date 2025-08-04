"""
This module provides dataclasses for algorithm configurations and results
to replace dictionary-based state management.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import numpy as np
from .geometry import Point3D


@dataclass
class GazePrediction:
    """Result of gaze estimation algorithm."""

    gaze_point: Point3D  # Predicted gaze point in world coordinates
    confidence: float  # Confidence score [0, 1]
    algorithm_name: str  # Name of algorithm used
    processing_time: Optional[float] = None  # Processing time in seconds
    intermediate_results: Optional[Dict[str, Any]] = None  # Algorithm-specific data

    @property
    def is_reliable(self) -> bool:
        """Check if prediction meets minimum confidence threshold."""
        return self.confidence >= 0.5


@dataclass
class InterpolationConfig:
    """Configuration for polynomial interpolation algorithm."""

    polynomial_degree: int = 2  # Degree of polynomial fitting
    cross_terms: bool = True  # Include cross terms (xy, x²y, etc.)
    regularization: float = 1e-6  # L2 regularization strength
    calibration_points: int = 9  # Number of calibration points required

    @property
    def num_coefficients(self) -> int:
        """Calculate number of polynomial coefficients."""
        if self.cross_terms:
            return ((self.polynomial_degree + 1) * (self.polynomial_degree + 2)) // 2
        else:
            return self.polynomial_degree + 1


@dataclass
class AlgorithmState:
    """Base class for algorithm state management."""

    is_calibrated: bool = False
    calibration_error: Optional[float] = None  # RMS calibration error
    last_update: Optional[float] = None  # Timestamp of last state update

    def reset(self) -> None:
        """Reset algorithm to uncalibrated state."""
        self.is_calibrated = False
        self.calibration_error = None
        self.last_update = None


@dataclass
class PolynomialFeatures:
    """Structured polynomial feature representation.

    Handles both 1D feature vectors and 2D feature matrices
    used in interpolation algorithms. Encapsulates prediction logic.
    """

    features: np.ndarray  # Feature array (1D or 2D)
    polynomial_name: str  # Name of polynomial that generated these features

    @property
    def is_1d(self) -> bool:
        """Check if features are 1D (shared for x and y)."""
        return self.features.ndim == 1

    @property
    def is_2d(self) -> bool:
        """Check if features are 2D (separate for x and y)."""
        return self.features.ndim == 2

    @property
    def feature_count(self) -> int:
        """Number of features per coordinate."""
        if self.is_1d:
            return len(self.features)
        else:
            return self.features.shape[1]

    def predict(self, x_coefficients: np.ndarray, y_coefficients: np.ndarray, plane_info=None) -> "Point3D":
        """Predict gaze coordinates using this polynomial's features.

        Args:
            x_coefficients: Calibration coefficients for first coordinate
            y_coefficients: Calibration coefficients for second coordinate
            plane_info: PlaneInfo object for coordinate reconstruction (optional, defaults to XZ plane)

        Returns:
            Point3D: Predicted gaze point in 3D coordinates
        """
        if self.is_1d:
            # Shared features: reconstruct calibration matrix
            A = np.vstack([x_coefficients, y_coefficients])
            gaze_2d = A @ self.features
        else:
            # Separate features for each coordinate
            coord1_features = self.features[0, :]
            coord2_features = self.features[1, :]
            gaze_2d = np.array([x_coefficients @ coord1_features, y_coefficients @ coord2_features])

        # Reconstruct 3D point using plane information
        if plane_info is not None:
            return plane_info.reconstruct_3d_point(gaze_2d[0], gaze_2d[1])
        else:
            # Default to XZ plane for backward compatibility
            return Point3D(gaze_2d[0], 0.0, gaze_2d[1])


@dataclass
class HennesseyConfig:
    """Configuration for Hennessey gaze tracking algorithm."""

    cornea_radius: float = 7.98e-3  # Corneal radius in meters
    pupil_cornea_distance: float = 4.44e-3  # Distance from pupil to cornea center
    recalib_type: str = "hennessey"  # Recalibration method
    pupil_algorithm: str = "hennessey"  # Pupil detection algorithm
    parameter_error: float = 0.0  # Relative error in eye model parameters
    empirical_correction: float = 1.165  # Empirical pupil radius correction factor


@dataclass
class HennesseyState(AlgorithmState):
    """State for Hennessey algorithm."""

    cornea_center: Optional[Point3D] = None  # Estimated cornea center
    recalib_data: Optional[Dict[str, Any]] = None  # Recalibration parameters
    pupil_radius: Optional[float] = None  # Current pupil radius estimate


@dataclass
class InterpolationState(AlgorithmState):
    """State for interpolation algorithm."""

    x_coefficients: Optional[np.ndarray] = None  # Polynomial coefficients for x
    y_coefficients: Optional[np.ndarray] = None  # Polynomial coefficients for y
    input_normalization: Optional[Dict[str, float]] = None  # Input scaling parameters
