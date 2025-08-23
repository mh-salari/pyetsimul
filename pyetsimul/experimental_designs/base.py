"""Base classes for experimental designs."""

from abc import ABC, abstractmethod
from typing import Any, Dict


class ExperimentalDesignBase(ABC):
    """Abstract base class for all experimental designs.

    Experimental designs specify the structure and parameters of experiments
    or evaluations. They can be used by both evaluation modules and data
    generation modules.
    """

    @abstractmethod
    def get_design_parameters(self) -> Dict[str, Any]:
        """Return the design parameters as a dictionary.

        Returns:
            Dictionary containing all design parameters
        """
        pass

    @abstractmethod
    def validate_design(self) -> bool:
        """Validate that the design parameters are consistent and valid.

        Returns:
            True if design is valid, raises ValueError if not
        """
        pass
