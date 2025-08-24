"""Core parameter variation architecture."""

from abc import ABC, abstractmethod
from typing import Any, List
from ..core import Eye, EyeTracker


class ParameterVariation(ABC):
    """Base class for parameter variations."""

    def __init__(self, param_name: str):
        self.param_name = param_name

    @abstractmethod
    def generate_values(self) -> List[Any]:
        """Generate all values for this parameter variation."""
        pass

    @abstractmethod
    def apply_to_eye(self, eye: Eye, value: Any) -> None:
        """Apply variation value to Eye object."""
        pass


class VariationStrategy(ABC):
    """Base strategy for using parameter variations."""

    @abstractmethod
    def execute(self, eye: Eye, et: EyeTracker, variation: ParameterVariation) -> Any:
        """Execute strategy over all variation values."""
        pass
