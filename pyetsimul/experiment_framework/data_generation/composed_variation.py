"""Composed parameter variations for complex experiment designs."""

from typing import List, Any, Dict
from .core import ParameterVariation


class ComposedVariation(ParameterVariation):
    """Combines multiple parameter variations into a single experiment."""

    def __init__(self, variations: List[ParameterVariation], param_name: str = "composed"):
        """Initialize composed variation.

        Args:
            variations: List of parameter variations to combine
            param_name: Name for the composed parameter
        """
        super().__init__(param_name)
        self.variations = variations

        if not variations:
            raise ValueError("Must provide at least one variation")

    def generate_values(self) -> List[Dict[str, Any]]:
        """Generate cartesian product of all variation values."""
        # Get all individual variation values
        all_values = []
        for variation in self.variations:
            values = variation.generate_values()
            all_values.append((variation.param_name, values))

        # Generate cartesian product
        combinations = []
        self._generate_combinations(all_values, {}, 0, combinations)

        return combinations

    def _generate_combinations(self, all_values, current_combo, index, result):
        """Recursively generate all combinations."""
        if index == len(all_values):
            result.append(current_combo.copy())
            return

        param_name, values = all_values[index]
        for value in values:
            current_combo[param_name] = value
            self._generate_combinations(all_values, current_combo, index + 1, result)
            del current_combo[param_name]


class SequentialVariation(ParameterVariation):
    """Applies variations sequentially rather than in combination."""

    def __init__(self, variations: List[ParameterVariation], param_name: str = "sequential"):
        """Initialize sequential variation.

        Args:
            variations: List of parameter variations to apply sequentially
            param_name: Name for the sequential parameter
        """
        super().__init__(param_name)
        self.variations = variations

        if not variations:
            raise ValueError("Must provide at least one variation")

    def generate_values(self) -> List[Dict[str, Any]]:
        """Generate sequential values from all variations."""
        all_combinations = []

        for i, variation in enumerate(self.variations):
            values = variation.generate_values()
            for value in values:
                combination = {"variation_index": i, "variation_name": variation.param_name, "value": value}
                all_combinations.append(combination)

        return all_combinations
