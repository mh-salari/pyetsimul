"""Generic parameter variation for any eye model parameter."""

from collections.abc import Iterable
from dataclasses import fields

import numpy as np

from ..core import Eye
from ..core.eye_model import EyeModel
from .core import EyeParameterVariation

_EYE_MODEL_FIELDS = frozenset(model_field.name for model_field in fields(EyeModel))


class GenericEyeVariation(EyeParameterVariation):
    """Generic parameter variation for any eye parameter, named by a path.

    ``"pupil_diameter"`` sets the pupil size. A path that begins with a field of the eye model
    (``"fovea_alpha_deg"``, ``"cornea.anterior_k"``) gives the eye a model with the changed value, so the
    eye is built again for it. Any other path names an attribute of the placed eye or of one of its
    objects (``"eyelid.openness"``).
    """

    def __init__(self, parameter_name: str, value_range: list[float], num_steps: int) -> None:
        """Initialize generic parameter variation.

        Args:
            parameter_name: Parameter path (e.g., "fovea_alpha_deg", "cornea.anterior_radius")
            value_range: [min_value, max_value] range
            num_steps: Number of steps to generate

        """
        super().__init__(parameter_name)
        self.value_range = value_range
        self.num_steps = num_steps

    def describe(self) -> str:
        """Return human-readable description of the parameter variation."""
        param_name = self.param_name.replace("_", " ")
        min_val, max_val = self.value_range
        return f"{param_name} {min_val:.3f}-{max_val:.3f} ({self.num_steps} steps)"

    def __len__(self) -> int:
        """Return the number of variation steps."""
        return self.num_steps

    def generate_values(self) -> Iterable[float]:
        """Generate parameter values using numpy linspace."""
        if self.num_steps == 1:
            yield (self.value_range[0] + self.value_range[1]) / 2
            return
        yield from np.linspace(self.value_range[0], self.value_range[1], self.num_steps)

    def apply_to_eye(self, eye: Eye, value: float) -> None:
        """Apply parameter value to eye using parameter path resolution."""
        GenericEyeVariation._set_parameter(eye, self.param_name, value)

    @staticmethod
    def _set_parameter(eye: Eye, parameter_path: str, value: float) -> None:
        """Set the parameter that the path names on the eye.

        Raises:
            AttributeError: If the path names no parameter of the eye model and no attribute of the eye.

        """
        if parameter_path == "pupil_diameter":
            eye.set_pupil_diameter(value)
            return

        name, _, attribute = parameter_path.partition(".")
        if name in _EYE_MODEL_FIELDS:
            eye.set_model(eye.model.with_parameter(parameter_path, value))
            return

        # State of the placed eye that is not part of its model.
        owner, leaf = eye, name
        if attribute and hasattr(eye, name):
            owner, leaf = getattr(eye, name), attribute
        if not hasattr(owner, leaf):
            raise AttributeError(
                f"Cannot vary {parameter_path!r}: {type(owner).__name__} has no attribute {leaf!r}, and "
                f"{name!r} is not a field of the eye model."
            )
        setattr(owner, leaf, value)
