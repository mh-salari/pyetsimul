"""Generic parameter variation for any eye model parameter."""

import numpy as np
from typing import Iterable, Optional
from ..core import Eye
from .core import EyeParameterVariation


class GenericEyeVariation(EyeParameterVariation):
    """Generic parameter variation for any eye model parameter.

    Supports direct property assignment, method calls, and nested object parameters
    through string-based parameter paths.
    """

    def __init__(self, parameter_name: str, value_range: list[float], num_steps: int, description: Optional[str] = None):
        """Initialize generic parameter variation.

        Args:
            parameter_name: Parameter path (e.g., "fovea_alpha_deg", "cornea.anterior_radius")
            value_range: [min_value, max_value] range
            num_steps: Number of steps to generate
            description: Optional human-readable description
        """
        super().__init__(parameter_name)
        self.value_range = value_range
        self.num_steps = num_steps
        self._description = description

    def describe(self) -> str:
        """Return human-readable description of the parameter variation."""
        param_name = self.param_name.replace("_", " ")
        min_val, max_val = self.value_range
        return f"{param_name} {min_val:.3f}-{max_val:.3f} ({self.num_steps} steps)"

    def __len__(self) -> int:
        return self.num_steps

    def generate_values(self) -> Iterable[float]:
        """Generate parameter values using numpy linspace."""
        if self.num_steps == 1:
            yield (self.value_range[0] + self.value_range[1]) / 2
            return
        yield from np.linspace(self.value_range[0], self.value_range[1], self.num_steps)

    def apply_to_eye(self, eye: Eye, value: float) -> None:
        """Apply parameter value to eye using parameter path resolution."""
        self._set_parameter(eye, self.param_name, value)

    def _set_parameter(self, eye: Eye, parameter_path: str, value: float) -> None:
        """Set parameter value using path resolution."""
        # Handle method calls
        if parameter_path == "pupil_diameter":
            eye.set_pupil_diameter(value)
            return

        # Handle nested object parameters
        if "." in parameter_path:
            obj_name, attr_name = parameter_path.split(".", 1)
            obj = getattr(eye, obj_name)
            setattr(obj, attr_name, value)
            return

        # Handle direct property assignment
        setattr(eye, parameter_path, value)
