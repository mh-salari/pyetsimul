"""Composed parameter variations for complex experiment designs."""

import math
from collections.abc import Generator, Iterable
from typing import Any

from .core import ParameterVariation


class ComposedVariation(ParameterVariation):
    """Combines multiple parameter variations into a single experiment."""

    def __init__(self, variations: list[ParameterVariation], param_name: str = "composed") -> None:
        """Initialize composed variation.

        Args:
            variations: List of parameter variations to combine
            param_name: Name for the composed parameter

        """
        super().__init__(param_name)
        self.variations = variations

        if not variations:
            raise ValueError("Must provide at least one variation")

    @property
    def description(self) -> str:
        """Get a description of the composed variation."""
        names = [v.__class__.__name__ for v in self.variations]
        return f"Composed({', '.join(names)})"

    def __len__(self) -> int:
        """Return the total number of combinations (Cartesian product)."""
        if not self.variations:
            return 0
        return math.prod(len(v) for v in self.variations)

    def generate_values(self) -> Iterable[dict[str, Any]]:
        """Generate Cartesian product of all variation values using a generator."""
        if not self.variations:
            return

        all_values = [(v.param_name, v.generate_values()) for v in self.variations]
        yield from self._generate_combinations(all_values, {}, 0)

    def describe(self) -> str:
        """Return a human-readable description of the composed variations."""
        descriptions = [var.describe() for var in self.variations]
        return f"Combined: {' + '.join(descriptions)}"

    def _generate_combinations(
        self, all_values: list[tuple[str, list[Any]]], current_combo: dict[str, Any], index: int
    ) -> Generator[dict[str, Any], None, None]:
        """Recursively generate all combinations."""
        if index == len(all_values):
            yield current_combo.copy()
            return

        param_name, values = all_values[index]
        for value in values:
            current_combo[param_name] = value
            yield from self._generate_combinations(all_values, current_combo, index + 1)
            del current_combo[param_name]


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
                combination = {"variation_index": i, "variation_name": variation.param_name, "value": value}
                yield combination
