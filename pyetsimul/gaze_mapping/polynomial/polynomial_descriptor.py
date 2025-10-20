"""Descriptor for polynomial terms with automatic function generation."""

from collections.abc import Callable
from dataclasses import dataclass
from itertools import starmap
from typing import Any

import numpy as np

from .polynomial_features import PolynomialFeatures


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

    def __post_init__(self) -> None:
        """Validate polynomial descriptor after initialization."""
        if len(self.terms) != len(self.orders):
            raise ValueError("terms and orders must have same length")
        self.orders = self._normalize_orders()

    def _normalize_orders(self) -> list[list[int]]:
        """Convert simplified orders format to standard [x_order, y_order] format."""

        def _normalize_orders_impl(orders: list[Any], terms: list[str]) -> list[list[int]]:
            normalized = []
            for order, term in zip(orders, terms, strict=False):
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
            return list(starmap(_normalize_orders_impl, zip(self.orders, self.terms, strict=False)))
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
        # For same features: return total number of shared features
        return len(self.terms)

    def get_term_descriptions(self) -> list[str] | list[list[str]]:
        """Get human-readable term descriptions for display."""
        if self.uses_different_xy_features:
            return [
                [PolynomialDescriptor._format_term(order) for order in coord_orders] for coord_orders in self.orders
            ]
        return [PolynomialDescriptor._format_term(order) for order in self.orders]

    @staticmethod
    def _format_term(order: list[int]) -> str:
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
            parts.append(f"x{PolynomialDescriptor._superscript(x_ord)}" if x_ord > 1 else "x")
        if y_ord > 0:
            parts.append(f"y{PolynomialDescriptor._superscript(y_ord)}" if y_ord > 1 else "y")

        return "".join(parts) if parts else "1"

    @staticmethod
    def _superscript(n: int) -> str:
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
        return self._generate_same_xy_function()

    def _generate_same_xy_function(self) -> Callable:
        """Generate function for polynomials using same features for X and Y."""

        def polynomial_func(x: float, y: float) -> "PolynomialFeatures":
            features = np.array([
                PolynomialDescriptor._evaluate_term(term, order, x, y)
                for term, order in zip(self.terms, self.orders, strict=False)
            ])
            return PolynomialFeatures(features=features, polynomial_name=self.name)

        return polynomial_func

    def _generate_different_xy_function(self) -> Callable:
        """Generate function for polynomials using different features for X and Y."""

        def polynomial_func(x: float, y: float) -> "PolynomialFeatures":
            coord_features = []
            for coord_terms, coord_orders in zip(self.terms, self.orders, strict=False):
                coord_vals = np.array([
                    PolynomialDescriptor._evaluate_term(term, order, x, y)
                    for term, order in zip(coord_terms, coord_orders, strict=False)
                ])
                coord_features.append(coord_vals)
            # Handle case where coordinates have different numbers of features
            if all(len(coord_features[0]) == len(cf) for cf in coord_features):
                features = np.array(coord_features)
            else:
                # Use object array for mixed-length coordinates
                features = np.array(coord_features, dtype=object)
            return PolynomialFeatures(features=features, polynomial_name=self.name)

        return polynomial_func

    @staticmethod
    def _evaluate_term(term: str, order: list[int], x: float, y: float) -> float:
        """Evaluate a single polynomial term."""
        if term == "1":
            return 1.0

        # For standard x,y terms, use orders directly
        if len(order) == 2:  # [x_order, y_order]
            x_ord, y_ord = order
            return (x**x_ord) * (y**y_ord)

        # This should never happen with valid polynomial descriptors
        raise ValueError(f"Invalid term evaluation: term='{term}', order={order}. Expected [x_order, y_order] format.")
