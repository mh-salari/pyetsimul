"""Polynomial feature functions for interpolation eye tracking.

Extensible registry system supporting user-defined polynomials.
"""

from typing import Callable, Optional
from dataclasses import dataclass
from ...types.algorithms import PolynomialFeatures, PolynomialDescriptor


# Hennessey et al. (2008) polynomial: [x*y, x, y, 1]
# Mathematical model (non-separable - shared features):
# gaze_x = a₀*x*y + a₁*x + a₂*y + a₃
# gaze_y = b₀*x*y + b₁*x + b₂*y + b₃
HENNESSEY_2008 = PolynomialDescriptor(
    name="hennessey_2008",
    description="Hennessey et al. (2008) polynomial with cross-terms",
    terms=["x*y", "x", "y", "1"],
    orders=[[1, 1], [1, 0], [0, 1], [0, 0]],
)


# Hoorman et al. (2008) polynomial: [[x, 1], [y, 1]]
# Mathematical model (separable - independent features):
# gaze_x = a₀*x + a₁
# gaze_y = b₀*y + b₁
HOORMAN_2008 = PolynomialDescriptor(
    name="hoorman_2008",
    description="Hoorman et al. (2008) linear polynomial",
    terms=[["x", "1"], ["y", "1"]],
    orders=[[[1, 0], [0, 0]], [[0, 1], [0, 0]]],
)


# Cerrolaza et al. (2008) polynomial: [x², y², x*y, x, y, 1]
# Mathematical model (non-separable - shared features):
# gaze_x = a₀*x² + a₁*y² + a₂*x*y + a₃*x + a₄*y + a₅
# gaze_y = b₀*x² + b₁*y² + b₂*x*y + b₃*x + b₄*y + b₅
CERROLAZA_2008 = PolynomialDescriptor(
    name="cerrolaza_2008",
    description="Cerrolaza et al. (2008) second-order polynomial",
    terms=["x*x", "y*y", "x*y", "x", "y", "1"],
    orders=[[2, 0], [0, 2], [1, 1], [1, 0], [0, 1], [0, 0]],
)


# Second-order polynomial: [x²*y², x², y², x*y, x, y, 1]
# Mathematical model (non-separable - shared features):
# gaze_x = a₀*x²*y² + a₁*x² + a₂*y² + a₃*x*y + a₄*x + a₅*y + a₆
# gaze_y = b₀*x²*y² + b₁*x² + b₂*y² + b₃*x*y + b₄*x + b₅*y + b₆
SECOND_ORDER = PolynomialDescriptor(
    name="second_order",
    description="Second-order polynomial with all cross-terms",
    terms=["x*x*y*y", "x*x", "y*y", "x*y", "x", "y", "1"],
    orders=[[2, 2], [2, 0], [0, 2], [1, 1], [1, 0], [0, 1], [0, 0]],
)


# Zhu and Ji (2005) polynomial: [[x*y, x, y, 1], [y², x, y, 1]]
# Mathematical model (separable - independent features):
# gaze_x = a₀*x*y + a₁*x + a₂*y + a₃
# gaze_y = b₀*y² + b₁*x + b₂*y + b₃
ZHU_JI_2005 = PolynomialDescriptor(
    name="zhu_ji_2005",
    description="Zhu and Ji (2005) asymmetric polynomial",
    terms=[["x*y", "x", "y", "1"], ["y*y", "x", "y", "1"]],
    orders=[[[1, 1], [1, 0], [0, 1], [0, 0]], [[0, 2], [1, 0], [0, 1], [0, 0]]],
)


# Cerrolaza and Villanueva (2008) polynomial: [[x², x, y, 1, 0], [x²*y, x², x*y, y, 1]]
# Mathematical model (separable - independent features):
# gaze_x = a₀*x² + a₁*x + a₂*y + a₃
# gaze_y = b₀*x²*y + b₁*x² + b₂*x*y + b₃*y + b₄
CERROLAZA_VILLANUEVA_2008 = PolynomialDescriptor(
    name="cerrolaza_villanueva_2008",
    description="Cerrolaza and Villanueva (2008) asymmetric polynomial",
    terms=[["x*x", "x", "y", "1", "1"], ["x*x*y", "x*x", "x*y", "y", "1"]],
    orders=[[[2, 0], [1, 0], [0, 1], [0, 0], [0, 0]], [[2, 1], [2, 0], [1, 1], [0, 1], [0, 0]]],
)


# Blignaut and Wium (2013) polynomial: [[1, x, x³, y², x*y, 0, 0], [1, x, x², y, y², x*y, x²*y]]
# Mathematical model (separable - independent features):
# gaze_x = a₀ + a₁*x + a₂*x³ + a₃*y² + a₄*x*y
# gaze_y = b₀ + b₁*x + b₂*x² + b₃*y + b₄*y² + b₅*x*y + b₆*x²*y
BLIGNAUT_WIUM_2013 = PolynomialDescriptor(
    name="blignaut_wium_2013",
    description="Blignaut and Wium (2013) high-order polynomial",
    terms=[["1", "x", "x*x*x", "y*y", "x*y", "1", "1"], ["1", "x", "x*x", "y", "y*y", "x*y", "x*x*y"]],
    orders=[
        [[0, 0], [1, 0], [3, 0], [0, 2], [1, 1], [0, 0], [0, 0]],
        [[0, 0], [1, 0], [2, 0], [0, 1], [0, 2], [1, 1], [2, 1]],
    ],
)


@dataclass
class PolynomialInfo:
    """Information about a registered polynomial."""

    descriptor: PolynomialDescriptor

    @property
    def name(self) -> str:
        """Get polynomial name from descriptor."""
        return self.descriptor.name

    @property
    def description(self) -> str:
        """Get polynomial description from descriptor."""
        return self.descriptor.description

    @property
    def model_type(self) -> str:
        """Get model type from descriptor (auto-determined)."""
        return "separable" if self.descriptor.is_separable else "non-separable"

    @property
    def feature_count(self) -> int:
        """Get feature count from descriptor."""
        return self.descriptor.feature_count

    @property
    def function(self) -> Callable[[float, float], PolynomialFeatures]:
        """Generate function from descriptor."""
        return self.descriptor.generate_function()


class PolynomialRegistry:
    """Registry for polynomial functions with user registration support."""

    def __init__(self):
        """Initialize empty registry."""
        self._polynomials: dict[str, PolynomialInfo] = {}

    def register(self, descriptor: PolynomialDescriptor) -> None:
        """Register a polynomial using descriptor.

        Args:
            descriptor: PolynomialDescriptor defining the polynomial

        Raises:
            ValueError: If polynomial name already exists
        """
        if descriptor.name in self._polynomials:
            raise ValueError(f"Polynomial '{descriptor.name}' already registered")

        info = PolynomialInfo(descriptor=descriptor)
        self._polynomials[descriptor.name] = info

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
            model_type: Filter by "non-separable" or "separable" model type

        Returns:
            List of polynomials meeting criteria
        """
        matches = []
        for info in self._polynomials.values():
            if model_type is not None and info.model_type != model_type:
                continue
            matches.append(info)
        return matches

    def unregister(self, name: str) -> bool:
        """Remove a polynomial from the registry."""
        if name not in self._polynomials:
            return False

        del self._polynomials[name]
        return True


polynomial_registry = PolynomialRegistry()

# Auto-register all built-in polynomial descriptors
builtin_descriptors = [
    HENNESSEY_2008,
    HOORMAN_2008,
    CERROLAZA_2008,
    SECOND_ORDER,
    ZHU_JI_2005,
    CERROLAZA_VILLANUEVA_2008,
    BLIGNAUT_WIUM_2013,
]

for descriptor in builtin_descriptors:
    polynomial_registry.register(descriptor)


def register_polynomial(descriptor: PolynomialDescriptor) -> None:
    """Register a user-defined polynomial."""
    polynomial_registry.register(descriptor)


def get_polynomial(name: str = "cerrolaza_2008") -> Callable[[float, float], PolynomialFeatures]:
    """Get polynomial function by name from global registry.

    Args:
        name: Polynomial name (default: 'cerrolaza_2008')

    Returns:
        Polynomial function that returns structured features

    Raises:
        ValueError: If polynomial name is not recognized
    """
    return polynomial_registry.get_polynomial(name)


def get_polynomial_info(name: str) -> Optional[PolynomialInfo]:
    """Get information about a polynomial from global registry."""
    return polynomial_registry.get_polynomial_info(name)


def list_available_polynomials() -> list[str]:
    """List all available polynomials in the global registry."""
    return polynomial_registry.list_polynomials()


def getpolynomial_registry() -> PolynomialRegistry:
    """Access the global polynomial registry for advanced operations."""
    return polynomial_registry
