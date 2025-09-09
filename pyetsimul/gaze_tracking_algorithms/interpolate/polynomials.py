"""Polynomial feature functions for interpolation eye tracking.

Extensible registry system supporting user-defined polynomials.
"""

import numpy as np
from typing import Callable, Optional
from dataclasses import dataclass
from ...types.algorithms import PolynomialFeatures


def hennessey_2008(x: float, y: float) -> PolynomialFeatures:
    """Hennessey et al. (2008) polynomial: [x*y, x, y, 1]

    Mathematical model (1D - shared features):
    gaze_x = a₀*x*y + a₁*x + a₂*y + a₃
    gaze_y = b₀*x*y + b₁*x + b₂*y + b₃

    Args:
        x, y: PCR vector components

    Returns:
        Structured polynomial features
    """
    features = np.array([x * y, x, y, 1])
    return PolynomialFeatures(features=features, polynomial_name="hennessey_2008")


def hoorman_2008(x: float, y: float) -> PolynomialFeatures:
    """Hoorman et al. (2008) polynomial: [[x, 1], [y, 1]]

    Mathematical model (2D - separate features):
    gaze_x = a₀*x + a₁
    gaze_y = b₀*y + b₁

    Args:
        x, y: PCR vector components

    Returns:
        Structured polynomial features (2D)
    """
    features = np.array([[x, 1], [y, 1]])
    return PolynomialFeatures(features=features, polynomial_name="hoorman_2008")


def cerrolaza_2008(x: float, y: float) -> PolynomialFeatures:
    """Cerrolaza et al. (2008) polynomial: [x², y², x*y, x, y, 1]

    Mathematical model (1D - shared features):
    gaze_x = a₀*x² + a₁*y² + a₂*x*y + a₃*x + a₄*y + a₅
    gaze_y = b₀*x² + b₁*y² + b₂*x*y + b₃*x + b₄*y + b₅

    Args:
        x, y: PCR vector components

    Returns:
        Structured polynomial features
    """
    features = np.array([x**2, y**2, x * y, x, y, 1])
    return PolynomialFeatures(features=features, polynomial_name="cerrolaza_2008")


def second_order(x: float, y: float) -> PolynomialFeatures:
    """Second-order polynomial: [x²*y², x², y², x*y, x, y, 1]

    Mathematical model (1D - shared features):
    gaze_x = a₀*x²*y² + a₁*x² + a₂*y² + a₃*x*y + a₄*x + a₅*y + a₆
    gaze_y = b₀*x²*y² + b₁*x² + b₂*y² + b₃*x*y + b₄*x + b₅*y + b₆

    Args:
        x, y: PCR vector components

    Returns:
        Structured polynomial features
    """
    features = np.array([x**2 * y**2, x**2, y**2, x * y, x, y, 1])
    return PolynomialFeatures(features=features, polynomial_name="second_order")


def zhu_ji_2005(x: float, y: float) -> PolynomialFeatures:
    """Zhu and Ji (2005) polynomial: [[x*y, x, y, 1], [y², x, y, 1]]

    Mathematical model (2D - separate features):
    gaze_x = a₀*x*y + a₁*x + a₂*y + a₃
    gaze_y = b₀*y² + b₁*x + b₂*y + b₃

    Args:
        x, y: PCR vector components

    Returns:
        Structured polynomial features (2D)
    """
    features = np.array([[x * y, x, y, 1], [y**2, x, y, 1]])
    return PolynomialFeatures(features=features, polynomial_name="zhu_ji_2005")


def cerrolaza_villanueva_2008(x: float, y: float) -> PolynomialFeatures:
    """Cerrolaza and Villanueva (2008) polynomial: [[x², x, y, 1, 0], [x²*y, x², x*y, y, 1]]

    Mathematical model (2D - separate features):
    gaze_x = a₀*x² + a₁*x + a₂*y + a₃
    gaze_y = b₀*x²*y + b₁*x² + b₂*x*y + b₃*y + b₄

    Args:
        x, y: PCR vector components

    Returns:
        Structured polynomial features (2D)
    """
    features = np.array([[x**2, x, y, 1, 0], [x**2 * y, x**2, x * y, y, 1]])
    return PolynomialFeatures(features=features, polynomial_name="cerrolaza_villanueva_2008")


def blignaut_wium_2013(x: float, y: float) -> PolynomialFeatures:
    """Blignaut and Wium (2013) polynomial: [[1, x, x³, y², x*y, 0, 0], [1, x, x², y, y², x*y, x²*y]]

    Mathematical model (2D - separate features):
    gaze_x = a₀ + a₁*x + a₂*x³ + a₃*y² + a₄*x*y
    gaze_y = b₀ + b₁*x + b₂*x² + b₃*y + b₄*y² + b₅*x*y + b₆*x²*y

    Args:
        x, y: PCR vector components

    Returns:
        Structured polynomial features (2D)
    """
    features = np.array([[1, x, x**3, y**2, x * y, 0, 0], [1, x, x**2, y, y**2, x * y, x**2 * y]])
    return PolynomialFeatures(features=features, polynomial_name="blignaut_wium_2013")


@dataclass
class PolynomialInfo:
    """Information about a registered polynomial."""

    name: str
    function: Callable[[float, float], PolynomialFeatures]
    description: str
    model_type: str
    feature_count: int


class PolynomialRegistry:
    """Registry for polynomial functions with user registration support."""

    def __init__(self):
        """Initialize empty registry."""
        self._polynomials: dict[str, PolynomialInfo] = {}

    def register(
        self,
        name: str,
        function: Callable[[float, float], PolynomialFeatures],
        description: str,
        model_type: str,
        feature_count: int,
    ) -> None:
        """Register a polynomial function.

        Args:
            name: Unique polynomial name
            function: Polynomial function returning PolynomialFeatures
            description: Human-readable description
            model_type: "1D" or "2D" model type
            feature_count: Number of features per coordinate

        Raises:
            ValueError: If polynomial name already exists
        """
        if name in self._polynomials:
            raise ValueError(f"Polynomial '{name}' already registered")

        if model_type not in ["1D", "2D"]:
            raise ValueError(
                f"Model type must be '1D' or '2D', got '{model_type}'. "
                f"Use '1D' for shared features (e.g., Hennessey 2008: [x*y, x, y, 1]) "
                f"or '2D' for separate X/Y features (e.g., Hoorman 2008: [[x, 1], [y, 1]])"
            )

        self._validate_polynomial_function(function, name, feature_count)

        info = PolynomialInfo(
            name=name,
            function=function,
            description=description,
            model_type=model_type,
            feature_count=feature_count,
        )

        self._polynomials[name] = info

    def get_polynomial(self, name: str) -> Callable[[float, float], PolynomialFeatures]:
        """Get polynomial function by name.

        Args:
            name: Polynomial name

        Returns:
            Polynomial function

        Raises:
            ValueError: If polynomial name is not recognized
        """
        if name not in self._polynomials:
            available = ", ".join(self.list_polynomials())
            raise ValueError(f"Unknown polynomial '{name}'. Available: {available}")
        return self._polynomials[name].function

    def get_polynomial_info(self, name: str) -> Optional[PolynomialInfo]:
        """Get information about a registered polynomial.

        Args:
            name: Polynomial name

        Returns:
            PolynomialInfo if found, None otherwise
        """
        return self._polynomials.get(name)

    def list_polynomials(self) -> list[str]:
        """Get list of all registered polynomial names."""
        return list(self._polynomials.keys())

    def list_polynomials_with_info(self) -> list[PolynomialInfo]:
        """Get list of all polynomial information."""
        return list(self._polynomials.values())

    def filter_polynomials(self, model_type: Optional[str] = None) -> list[PolynomialInfo]:
        """Filter polynomials by criteria.

        Args:
            model_type: Filter by "1D" or "2D" model type

        Returns:
            List of polynomials meeting criteria
        """
        matches = []
        for info in self._polynomials.values():
            if model_type is not None and info.model_type != model_type:
                continue
            matches.append(info)
        return matches

    def _validate_polynomial_function(self, function: Callable[[float, float], PolynomialFeatures], name: str, expected_count: int) -> None:
        """Validate polynomial function signature and return type."""
        test_points = [(0.0, 0.0), (1.0, 1.0), (-1.0, -1.0), (0.5, -0.5)]

        for x, y in test_points:
            try:
                result = function(x, y)
                if not isinstance(result, PolynomialFeatures):
                    raise ValueError(f"Polynomial '{name}' must return PolynomialFeatures, got {type(result)}")

                # Validate that a polynomial's output is tagged with the expected name.
                if result.polynomial_name != name:
                    raise ValueError(
                        f"The output of polynomial '{name}' is tagged with the name '{result.polynomial_name}', which doesn't match the expected name ('{name}')"
                    )

                # Validate that a polynomial produces the expected feature count.
                actual_count = result.feature_count
                if actual_count != expected_count:
                    raise ValueError(
                        f"Polynomial '{name}' returns {result.feature_count} features, which doesn't match the expected number of features ({expected_count})"
                    )

                # Validate feature count consistency
                if hasattr(result, "features") and result.features is not None:
                    if result.is_2d:
                        expected_shape = (2, actual_count)
                    else:
                        expected_shape = (actual_count,)

                    if result.features.shape != expected_shape:
                        raise ValueError(
                            f"Polynomial '{name}' features shape {result.features.shape} doesn't match expected {expected_shape}"
                        )

            except Exception as e:
                raise ValueError(f"Polynomial '{name}' function validation failed at ({x}, {y}): {e}") from e

    def unregister(self, name: str) -> bool:
        """Remove a polynomial from the registry."""
        if name not in self._polynomials:
            return False

        del self._polynomials[name]
        return True


def _register_builtin_polynomials() -> None:
    """Register built-in polynomial functions."""
    builtin_polynomials = [
        ("hennessey_2008", hennessey_2008, "Hennessey et al. (2008) polynomial with cross-terms", "1D", 4),
        ("hoorman_2008", hoorman_2008, "Hoorman et al. (2008) linear polynomial", "2D", 2),
        ("cerrolaza_2008", cerrolaza_2008, "Cerrolaza et al. (2008) second-order polynomial", "1D", 6),
        ("second_order", second_order, "Second-order polynomial with all cross-terms", "1D", 7),
        ("zhu_ji_2005", zhu_ji_2005, "Zhu and Ji (2005) asymmetric polynomial", "2D", 4),
        (
            "cerrolaza_villanueva_2008",
            cerrolaza_villanueva_2008,
            "Cerrolaza and Villanueva (2008) asymmetric polynomial",
            "2D",
            5,
        ),
        ("blignaut_wium_2013", blignaut_wium_2013, "Blignaut and Wium (2013) high-order polynomial", "2D", 7),
    ]

    for name, func, desc, model_type, feature_count in builtin_polynomials:
        _polynomial_registry.register(name, func, desc, model_type, feature_count)


_polynomial_registry = PolynomialRegistry()
_register_builtin_polynomials()


def register_polynomial(
    name: str,
    function: Callable[[float, float], PolynomialFeatures],
    description: str,
    model_type: str,
    feature_count: int,
) -> None:
    """Register a user-defined polynomial in the global registry.

    Args:
        name: Unique polynomial name
        function: Polynomial function returning PolynomialFeatures
        description: Human-readable description
        model_type: "1D" or "2D" model type
        feature_count: Number of features per coordinate
    """
    _polynomial_registry.register(
        name=name,
        function=function,
        description=description,
        model_type=model_type,
        feature_count=feature_count,
    )


def get_polynomial(name: str = "cerrolaza_2008") -> Callable[[float, float], PolynomialFeatures]:
    """Get polynomial function by name from global registry.

    Args:
        name: Polynomial name (default: 'cerrolaza_2008')

    Returns:
        Polynomial function that returns structured features

    Raises:
        ValueError: If polynomial name is not recognized
    """
    return _polynomial_registry.get_polynomial(name)


def get_polynomial_info(name: str) -> Optional[PolynomialInfo]:
    """Get information about a polynomial from global registry."""
    return _polynomial_registry.get_polynomial_info(name)


def list_available_polynomials() -> list[str]:
    """List all available polynomials in the global registry."""
    return _polynomial_registry.list_polynomials()


def get_polynomial_registry() -> PolynomialRegistry:
    """Access the global polynomial registry for advanced operations."""
    return _polynomial_registry
