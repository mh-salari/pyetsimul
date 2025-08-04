"""Polynomial feature functions for interpolation eye tracking.

Based on various eye tracking calibration papers with different polynomial formulations.
"""

import numpy as np
from typing import Callable, Dict, List, Optional
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
    model_type: str  # "1D" or "2D"
    feature_count: int
    paper_reference: Optional[str] = None
    year: Optional[int] = None


class PolynomialRegistry:
    """Registry for polynomial functions."""

    def __init__(self):
        """Initialize empty registry."""
        self._polynomials: Dict[str, PolynomialInfo] = {}
        self._aliases: Dict[str, str] = {}  # Alias -> canonical name mapping

    def register(
        self,
        name: str,
        function: Callable[[float, float], PolynomialFeatures],
        description: str,
        model_type: str,
        feature_count: int,
        paper_reference: Optional[str] = None,
        year: Optional[int] = None,
        aliases: Optional[List[str]] = None,
    ) -> None:
        """Register a polynomial function.

        Args:
            name: Unique polynomial name
            function: Polynomial function
            description: Human-readable description
            model_type: "1D" or "2D" model type
            feature_count: Number of features per coordinate
            paper_reference: Academic paper reference
            year: Year of publication
            aliases: Alternative names for the polynomial
        """
        if name in self._polynomials:
            raise ValueError(f"Polynomial '{name}' already registered")

        info = PolynomialInfo(
            name=name,
            function=function,
            description=description,
            model_type=model_type,
            feature_count=feature_count,
            paper_reference=paper_reference,
            year=year,
        )

        self._polynomials[name] = info

        # Register aliases
        if aliases:
            for alias in aliases:
                if alias in self._aliases:
                    raise ValueError(f"Alias '{alias}' already registered")
                self._aliases[alias] = name

    def get_polynomial(self, name: str) -> Callable[[float, float], PolynomialFeatures]:
        """Get polynomial function by name.

        Args:
            name: Polynomial name or alias

        Returns:
            Polynomial function

        Raises:
            ValueError: If polynomial name is not recognized
        """
        info = self.get_polynomial_info(name)
        if info is None:
            available = ", ".join(self.list_polynomials())
            raise ValueError(f"Unknown polynomial '{name}'. Available: {available}")
        return info.function

    def get_polynomial_info(self, name: str) -> Optional[PolynomialInfo]:
        """Get information about a registered polynomial.

        Args:
            name: Polynomial name or alias

        Returns:
            PolynomialInfo if found, None otherwise
        """
        # Check if it's an alias first
        canonical_name = self._aliases.get(name, name)
        return self._polynomials.get(canonical_name)

    def list_polynomials(self) -> List[str]:
        """Get list of all registered polynomial names."""
        return list(self._polynomials.keys())

    def list_polynomials_with_info(self) -> List[PolynomialInfo]:
        """Get list of all polynomial information."""
        return list(self._polynomials.values())

    def filter_polynomials(self, model_type: Optional[str] = None) -> List[PolynomialInfo]:
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


# Global polynomial registry
_polynomial_registry = PolynomialRegistry()

# Register all polynomial functions
_polynomial_registry.register(
    "hennessey_2008",
    hennessey_2008,
    "Hennessey et al. (2008) polynomial with cross-terms",
    "1D",
    4,
    "Craig Hennessey et al. (2008)",
    2008,
)
_polynomial_registry.register(
    "hoorman_2008", hoorman_2008, "Hoorman et al. (2008) linear polynomial", "2D", 2, "Hoorman et al. (2008)", 2008
)
_polynomial_registry.register(
    "cerrolaza_2008",
    cerrolaza_2008,
    "Cerrolaza et al. (2008) second-order polynomial",
    "1D",
    6,
    "Cerrolaza et al. (2008)",
    2008,
)
_polynomial_registry.register(
    "second_order", second_order, "Second-order polynomial with all cross-terms", "1D", 7, None, None
)
_polynomial_registry.register(
    "zhu_ji_2005", zhu_ji_2005, "Zhu and Ji (2005) asymmetric polynomial", "2D", 4, "Zhu and Ji (2005)", 2005
)
_polynomial_registry.register(
    "cerrolaza_villanueva_2008",
    cerrolaza_villanueva_2008,
    "Cerrolaza and Villanueva (2008) asymmetric polynomial",
    "2D",
    5,
    "Cerrolaza and Villanueva (2008)",
    2008,
)
_polynomial_registry.register(
    "blignaut_wium_2013",
    blignaut_wium_2013,
    "Blignaut and Wium (2013) high-order polynomial",
    "2D",
    7,
    "Blignaut and Wium (2013)",
    2013,
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


def list_available_polynomials() -> List[str]:
    """List all available polynomials in the global registry."""
    return _polynomial_registry.list_polynomials()
