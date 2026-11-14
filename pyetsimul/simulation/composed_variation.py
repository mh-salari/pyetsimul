"""Composed parameter variations for complex experiment designs."""

import itertools
import math
from collections.abc import Iterable
from typing import Any

from .core import ParameterVariation


class ComposedVariation(ParameterVariation):
    """Combines multiple parameter variations into a single experiment."""

    def __init__(self, variations: list[ParameterVariation], param_name: str = "composed") -> None:
        """Initialize composed variation.

        Args:
            variations: List of parameter variations to combine
            param_name: Name for the composed parameter

        Raises:
            ValueError: If no variation is given, or two variations share a param_name

        """
        super().__init__(param_name)
        self.variations = variations

        if not variations:
            raise ValueError("Must provide at least one variation")

        # Each combination is a dict keyed by param_name, so two variations cannot share one.
        names = [v.param_name for v in variations]
        repeated = sorted({name for name in names if names.count(name) > 1})
        if repeated:
            raise ValueError(f"Variations in a composition need distinct param_name values; repeated: {repeated}")

    def __len__(self) -> int:
        """Return the total number of combinations (Cartesian product)."""
        return math.prod(len(v) for v in self.variations)

    def generate_values(self) -> Iterable[dict[str, Any]]:
        """Generate the Cartesian product of all variation values, one dict per combination."""
        names = [v.param_name for v in self.variations]
        for combination in itertools.product(*(v.generate_values() for v in self.variations)):
            yield dict(zip(names, combination, strict=True))

    def describe(self) -> str:
        """Return a human-readable description of the composed variations."""
        descriptions = [var.describe() for var in self.variations]
        return f"Combined: {' + '.join(descriptions)}"


class SequentialVariation(ParameterVariation):
    """Applies variations sequentially rather than in combination."""

    def __init__(self, variations: list[ParameterVariation], param_name: str = "sequential") -> None:
        """Initialize sequential variation.

        Args:
            variations: List of parameter variations to apply sequentially
            param_name: Name for the sequential parameter

        """
        super().__init__(param_name)
        self.variations = variations

        if not variations:
            raise ValueError("Must provide at least one variation")

    def describe(self) -> str:
        """Return a human-readable description of the sequential variations."""
        descriptions = [var.describe() for var in self.variations]
        return f"Sequential: {' → '.join(descriptions)}"

    def __len__(self) -> int:
        """Return the total number of values (sum of lengths)."""
        return sum(len(v) for v in self.variations)

    def generate_values(self) -> Iterable[dict[str, Any]]:
        """Generate sequential values from all variations."""
        for i, variation in enumerate(self.variations):
            for value in variation.generate_values():
                yield {"variation_index": i, "variation_name": variation.param_name, "value": value}
