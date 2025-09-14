"""Core parameter variation architecture for pure data generation."""

from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any

from ..core import Eye


class ParameterVariation(ABC):
    """Base class for all parameter variations - pure data generation only."""

    def __init__(self, param_name: str) -> None:
        """Initialize parameter variation.

        Args:
            param_name: Name of the parameter to vary

        """
        self.param_name = param_name

    @abstractmethod
    def generate_values(self) -> Iterable[Any]:
        """Generate all values for this parameter variation."""

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of variation values."""

    @abstractmethod
    def describe(self) -> str:
        """Return a human-readable description of this variation."""


class EyeParameterVariation(ParameterVariation):
    """Base class for variations that modify eye objects."""

    @abstractmethod
    def apply_to_eye(self, eye: Eye, value: Any) -> None:  # noqa: ANN401
        """Apply variation value to Eye object."""


class TargetVariation(ParameterVariation):
    """Base class for variations that provide gaze targets."""

    def get_targets(self) -> list[Any]:
        """Get all target values for this variation."""
        return list(self.generate_values())


class VariationStrategy(ABC):
    """Base strategy for using parameter variations."""

    @abstractmethod
    def execute(self, variation: ParameterVariation) -> Any:  # noqa: ANN401
        """Execute strategy over all variation values."""
