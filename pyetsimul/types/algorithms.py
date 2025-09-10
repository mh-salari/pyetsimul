"""
This module provides dataclasses for algorithm configurations and results
to replace dictionary-based state management.
"""

from dataclasses import dataclass
from typing import Optional, Any, Callable
import numpy as np
from .geometry import Point3D


@dataclass
class PolynomialDescriptor:
    """Descriptor for polynomial terms with automatic function generation.

    Describes polynomial terms mathematically using variables and their orders,
    enabling automatic function generation and feature type detection.

    Examples:
        Same features for X,Y:
            terms=["x", "y", "x*y", "x", "y", "1"]
            orders=[2, 2, [1,1], 1, 1, 0]

        Different features for X,Y:
            terms=[["x*y", "x", "1"], ["y", "1"]]
            orders=[[[1,1], 1, 0], [1, 0]]
    """

    name: str
    description: str
    terms: list[str] | list[list[str]]
    orders: list[int | list[int]] | list[list[int | list[int]]]

    def __post_init__(self):
        if len(self.terms) != len(self.orders):
            raise ValueError("terms and orders must have same length")
        self.orders = self._normalize_orders()

    def _normalize_orders(self):
        """Convert simplified orders format to standard [x_order, y_order] format."""

        def _normalize_orders_impl(orders, terms):
            normalized = []
            for order, term in zip(orders, terms):
                if isinstance(order, int):
                    if term == "x":
                        normalized.append([order, 0])
                    elif term == "y":
                        normalized.append([0, order])
                    elif term == "1":
                        if order != 0:
                            raise ValueError(f"If term is '1', order should be 0. Was: {order}")
                        normalized.append([0, 0])
                    else:
                        raise ValueError(
                            f"If term is not 'x' or 'y' (term was '{term}'), order must be specified as a list."
                        )
                else:
                    normalized.append(order)
            return normalized

        if self.uses_different_xy_features:
            return [_normalize_orders_impl(o, t) for o, t in zip(self.orders, self.terms)]
        else:
            return _normalize_orders_impl(self.orders, self.terms)

    @property
    def uses_different_xy_features(self) -> bool:
        """Check if polynomial uses different features for X and Y coordinates."""
        # Check if terms is list[list[str]] (different features) vs list[str] (same features)
        return isinstance(self.terms, list) and len(self.terms) > 0 and isinstance(self.terms[0], list)

    @property
    def uses_same_xy_features(self) -> bool:
        """Check if polynomial uses same features for both X and Y coordinates."""
        return not self.uses_different_xy_features

    @property
    def feature_count(self) -> int:
        """Total number of features."""
        if self.uses_different_xy_features:
            # For different features: return total number of features across all coordinates
            return sum(len(coord_terms) for coord_terms in self.terms)
        else:
            # For same features: return total number of shared features
            return len(self.terms)

    def get_term_descriptions(self) -> list[str] | list[list[str]]:
        """Get human-readable term descriptions for display."""
        if self.uses_different_xy_features:
            return [[self._format_term(order) for order in coord_orders] for coord_orders in self.orders]
        else:
            return [self._format_term(order) for order in self.orders]

    def _format_term(self, order: list[int]) -> str:
        """Format a single term with mathematical notation."""
        if len(order) != 2:
            raise ValueError(f"Invalid order format: {order}. Expected [x_order, y_order].")

        x_ord, y_ord = order

        # Handle constant term
        if x_ord == 0 and y_ord == 0:
            return "1"

        # Build mathematical notation with Unicode superscripts
        parts = []
        if x_ord > 0:
            parts.append(f"x{self._superscript(x_ord)}" if x_ord > 1 else "x")
        if y_ord > 0:
            parts.append(f"y{self._superscript(y_ord)}" if y_ord > 1 else "y")

        return "".join(parts) if parts else "1"

    def _superscript(self, n: int) -> str:
        """Convert number to Unicode superscript."""
        superscripts = {
            "0": "⁰",
            "1": "¹",
            "2": "²",
            "3": "³",
            "4": "⁴",
            "5": "⁵",
            "6": "⁶",
            "7": "⁷",
            "8": "⁸",
            "9": "⁹",
        }
        return "".join(superscripts[digit] for digit in str(n))

    def generate_function(self) -> Callable:
        """Generate polynomial function from descriptor."""
        if self.uses_different_xy_features:
            return self._generate_different_xy_function()
        else:
            return self._generate_same_xy_function()

    def _generate_same_xy_function(self) -> Callable:
        """Generate function for polynomials using same features for X and Y."""

        def polynomial_func(x: float, y: float) -> "PolynomialFeatures":
            features = np.array(
                [self._evaluate_term(term, order, x, y) for term, order in zip(self.terms, self.orders)]
            )
            return PolynomialFeatures(features=features, polynomial_name=self.name)

        return polynomial_func

    def _generate_different_xy_function(self) -> Callable:
        """Generate function for polynomials using different features for X and Y."""

        def polynomial_func(x: float, y: float) -> "PolynomialFeatures":
            coord_features = []
            for coord_terms, coord_orders in zip(self.terms, self.orders):
                coord_vals = np.array(
                    [self._evaluate_term(term, order, x, y) for term, order in zip(coord_terms, coord_orders)]
                )
                coord_features.append(coord_vals)
            # Handle case where coordinates have different numbers of features
            if all(len(coord_features[0]) == len(cf) for cf in coord_features):
                features = np.array(coord_features)
            else:
                # Use object array for mixed-length coordinates
                features = np.array(coord_features, dtype=object)
            return PolynomialFeatures(features=features, polynomial_name=self.name)

        return polynomial_func

    def _evaluate_term(self, term: str, order: list[int], x: float, y: float) -> float:
        """Evaluate a single polynomial term."""
        if term == "1":
            return 1.0

        # For standard x,y terms, use orders directly
        if len(order) == 2:  # [x_order, y_order]
            x_ord, y_ord = order
            return (x**x_ord) * (y**y_ord)

        # This should never happen with valid polynomial descriptors
        raise ValueError(f"Invalid term evaluation: term='{term}', order={order}. Expected [x_order, y_order] format.")


@dataclass
class GazePrediction:
    """Result of gaze estimation algorithm."""

    gaze_point: Point3D  # Predicted gaze point in world coordinates
    confidence: float  # Confidence score [0, 1]
    algorithm_name: str  # Name of algorithm used
    processing_time: Optional[float] = None  # Processing time in seconds
    intermediate_results: Optional[dict[str, Any]] = None  # Algorithm-specific data

    @property
    def is_reliable(self) -> bool:
        """Check if prediction meets minimum confidence threshold."""
        return self.confidence >= 0.5


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
        elif self.features.dtype == object:
            return sum(len(coord_features) for coord_features in self.features)
        else:
            return self.features.shape[0] * self.features.shape[1]

    def predict(self, x_coefficients: np.ndarray, y_coefficients: np.ndarray, plane_info=None) -> "Point3D":
        """Predict gaze coordinates using this polynomial's features.

        Args:
            x_coefficients: Calibration coefficients for first coordinate
            y_coefficients: Calibration coefficients for second coordinate
            plane_info: PlaneInfo object for coordinate reconstruction (optional, defaults to XZ plane)

        Returns:
            Point3D: Predicted gaze point in 3D coordinates
        """
        if self.uses_same_xy_features:
            A = np.vstack([x_coefficients, y_coefficients])
            gaze_2d = A @ self.features
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
        else:
            return self.features[0, :], self.features[1, :]


@dataclass
class PolynomialGazeModelState(AlgorithmState):
    """State for polynomial gaze model algorithm."""

    x_coefficients: Optional[np.ndarray] = None  # Polynomial coefficients for x
    y_coefficients: Optional[np.ndarray] = None  # Polynomial coefficients for y
    input_normalization: Optional[dict[str, float]] = None  # Input scaling parameters

    def serialize(self) -> dict:
        """Serialize polynomial gaze model state to dictionary."""
        return {
            "is_calibrated": self.is_calibrated,
            "calibration_error": self.calibration_error,
            "last_update": self.last_update,
            "x_coefficients": self.x_coefficients.tolist() if self.x_coefficients is not None else None,
            "y_coefficients": self.y_coefficients.tolist() if self.y_coefficients is not None else None,
            "input_normalization": self.input_normalization,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "PolynomialGazeModelState":
        """Deserialize from dictionary representation."""
        state = cls(
            is_calibrated=data["is_calibrated"],
            calibration_error=data["calibration_error"],
            last_update=data["last_update"],
            input_normalization=data["input_normalization"],
        )

        # Convert coefficient lists back to numpy arrays
        if data["x_coefficients"] is not None:
            state.x_coefficients = np.array(data["x_coefficients"])
        if data["y_coefficients"] is not None:
            state.y_coefficients = np.array(data["y_coefficients"])

        return state
