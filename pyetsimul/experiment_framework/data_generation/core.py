"""Core parameter variation architecture for pure data generation."""

from abc import ABC, abstractmethod
from typing import Any, Iterable, List
from ...core import Eye


class ParameterVariation(ABC):
    """Base class for all parameter variations - pure data generation only."""

    def __init__(self, param_name: str):
        self.param_name = param_name

    @abstractmethod
    def generate_values(self) -> Iterable[Any]:
        """Generate all values for this parameter variation."""
        pass

    @abstractmethod
    def __len__(self) -> int:
        """Return the total number of variation values."""
        pass


class EyeParameterVariation(ParameterVariation):
    """Base class for variations that modify eye objects."""

    @abstractmethod
    def apply_to_eye(self, eye: Eye, value: Any) -> None:
        """Apply variation value to Eye object."""
        pass


class TargetVariation(ParameterVariation):
    """Base class for variations that provide gaze targets."""

    def get_targets(self) -> List[Any]:
        """Get all target values for this variation."""
        return self.generate_values()


class VariationStrategy(ABC):
    """Base strategy for using parameter variations."""

    @abstractmethod
    def execute(self, eyes: List[Eye], variation: ParameterVariation) -> Any:
        """Execute strategy over all variation values."""
        pass
