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
    enabling automatic function generation and separable/non-separable detection.

    Examples:
        Non-separable:
            terms=["x", "y", "x*y", "x", "y", "1"]
            orders=[2, 2, [1,1], 1, 1, 0]

        Separable:
            terms=[["x", "1"], ["y", "1"]]
            orders=[[[1,0], [0,0]], [[0,1], [0,0]]]
    """

    name: str
    description: str
    terms: list[str] | list[list[str]]
    orders: list[int | list[int]] | list[list[list[int]]]

    def __post_init__(self):
        self.orders = self._normalize_orders()
        if len(self.terms) != len(self.orders):
            raise ValueError("terms and orders must have same length")

    def _normalize_orders(self):
        """Convert simplified orders format to standard [x_order, y_order] format."""
        if self.is_separable:
            return self.orders

        normalized = []
        for i, order in enumerate(self.orders):
            if isinstance(order, int):
                term = self.terms[i]
                if term == "x":
                    normalized.append([order, 0])
                elif term == "y":
                    normalized.append([0, order])
                elif term == "1":
                    if order!=0:
                        raise ValueError(f"If term is '1', order should be 0. Was: {order}")
                    normalized.append([0, 0])
                else:
                    raise ValueError(f"If term is not 'x' or 'y' (term was '{term}'), order must be specified as a list.")
            else:
                normalized.append(order)
        return normalized

    @property
    def is_separable(self) -> bool:
        """Auto-determine if polynomial is separable from structure."""
        # Check if terms is list[list[str]] (separable) vs list[str] (non-separable)
        return isinstance(self.terms, list) and len(self.terms) > 0 and isinstance(self.terms[0], list)

    @property
    def feature_count(self) -> int:
        """Number of features per coordinate."""
        if self.is_separable:
            # For separable: return number of features in first coordinate (should match second)
            return len(self.terms[0])
        else:
            # For non-separable: return total number of shared features
            return len(self.terms)

    def get_term_descriptions(self) -> list[str]| list[list[str]]:
        """Get human-readable term descriptions for display."""
        if self.is_separable:
            return [
                [self._format_term(term, order) for term, order in zip(coord_terms, coord_orders)]
                for coord_terms, coord_orders in zip(self.terms, self.orders)
            ]
        else:
            return [self._format_term(term, order) for term, order in zip(self.terms, self.orders)]

    def _format_term(self, term: str, order: list[int]) -> str:
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
        if self.is_separable:
            return self._generate_separable_function()
        else:
            return self._generate_non_separable_function()

    def _generate_non_separable_function(self) -> Callable:
        """Generate function for non-separable polynomial."""

        def polynomial_func(x: float, y: float) -> "PolynomialFeatures":
            features = np.array(
                [self._evaluate_term(term, order, x, y) for term, order in zip(self.terms, self.orders)]
            )
            return PolynomialFeatures(features=features, polynomial_name=self.name)

        return polynomial_func

    def _generate_separable_function(self) -> Callable:
        """Generate function for separable polynomial."""

        def polynomial_func(x: float, y: float) -> "PolynomialFeatures":
            coord_features = []
            for coord_terms, coord_orders in zip(self.terms, self.orders):
                coord_vals = np.array(
                    [self._evaluate_term(term, order, x, y) for term, order in zip(coord_terms, coord_orders)]
                )
                coord_features.append(coord_vals)
            features = np.array(coord_features)
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

    Handles both non-separable (coupled x,y) and separable (independent x,y)
    polynomial features used in interpolation algorithms. Encapsulates prediction logic.
    """

    features: np.ndarray  # Feature array (1D or 2D)
    polynomial_name: str  # Name of polynomial that generated these features

    @property
    def is_non_separable(self) -> bool:
        """Check if polynomial is non-separable (same features shared for x and y coordinates)."""
        return self.features.ndim == 1

    @property
    def is_separable(self) -> bool:
        """Check if polynomial is separable (different features for x and y coordinates)."""
        return self.features.ndim == 2

    @property
    def feature_count(self) -> int:
        """Number of features per coordinate."""
        if self.is_non_separable:
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
        if self.is_non_separable:
            # Non-separable: same features used for both coordinates with different coefficients
            A = np.vstack([x_coefficients, y_coefficients])
            gaze_2d = A @ self.features
        else:
            # Separable: different features for each coordinate
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
class InterpolationState(AlgorithmState):
    """State for interpolation algorithm."""

    x_coefficients: Optional[np.ndarray] = None  # Polynomial coefficients for x
    y_coefficients: Optional[np.ndarray] = None  # Polynomial coefficients for y
    input_normalization: Optional[dict[str, float]] = None  # Input scaling parameters

    def serialize(self) -> dict:
        """Serialize interpolation state to dictionary."""
        return {
            "is_calibrated": self.is_calibrated,
            "calibration_error": self.calibration_error,
            "last_update": self.last_update,
            "x_coefficients": self.x_coefficients.tolist() if self.x_coefficients is not None else None,
            "y_coefficients": self.y_coefficients.tolist() if self.y_coefficients is not None else None,
            "input_normalization": self.input_normalization,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "InterpolationState":
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
