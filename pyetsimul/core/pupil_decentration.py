"""Pupil decentration models and registry system.

This module provides a registry for different pupil decentration models,
allowing users to register custom decentration profiles.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..types import Position3D
from .default_configs import PupilDecentrationDefaults


class PupilDecentrationModel(ABC):
    """Abstract base class for pupil decentration models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Model name for registry."""

    @abstractmethod
    def calculate_offset(self, current_diameter: float, baseline_diameter: float, **kwargs: float) -> Position3D:
        """Calculate decentration offset based on pupil diameter change.

        Args:
            current_diameter: Current pupil diameter in mm
            baseline_diameter: Baseline diameter (zero decentration point) in mm
            **kwargs: Model-specific parameters

        Returns:
            Position3D offset for pupil decentration

        """


class PupilDecentrationRegistry:
    """Registry for pupil decentration models."""

    _models: dict[str, PupilDecentrationModel] = None

    @classmethod
    def _ensure_models_dict(cls) -> None:
        """Initialize models dictionary if needed."""
        if cls._models is None:
            cls._models = {}

    @classmethod
    def register(cls, model: PupilDecentrationModel) -> None:
        """Register a decentration model.

        Args:
            model: PupilDecentrationModel instance to register

        """
        cls._ensure_models_dict()
        cls._models[model.name] = model

    @classmethod
    def get_model(cls, name: str) -> PupilDecentrationModel:
        """Get registered model by name.

        Args:
            name: Model name

        Returns:
            PupilDecentrationModel instance

        Raises:
            ValueError: If model name not found

        """
        cls._ensure_models_dict()
        if name not in cls._models:
            raise ValueError(f"Unknown decentration model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model names.

        Returns:
            List of registered model names

        """
        cls._ensure_models_dict()
        return list(cls._models.keys())


class WildenmannModel(PupilDecentrationModel):
    """Wildenmann & Schaeffel (2013) linear decentration model.

    Implements linear pupil decentration based on:
    Wildenmann, U., & Schaeffel, F. (2013). Variations of pupil centration and their
    effects on video eye tracking. Ophthalmic and Physiological Optics, 33(6), 634-641.

    Key findings from the paper:
    - "pupil centration changed about linearly with pupil size"
    - "All these changes followed an about linear function"
    - Average shift: 0.05 mm per mm pupil constriction
    - Individual variation: 0.044-0.179 mm per mm change in pupil size
    - Strong linear correlation: R² > 0.98 in all cases
    - Baseline pupil sizes tested: 4.75 ± 0.52 mm (at 800 lux)
    - Pupil diameter range: 4.0-5.7 mm

    Built-in coefficient of 0.05 mm/mm with user override capability.
    Baseline diameter auto-detected from current eye or user-specified.
    """

    @property
    def name(self) -> str:
        """Get the decentration model name."""
        return "wildenmann_2013"

    def calculate_offset(  # noqa: PLR6301
        self, current_diameter: float, baseline_diameter: float, x_coeff: float, y_coeff: float, z_coeff: float = 0.0
    ) -> Position3D:
        """Calculate Wildenmann linear decentration offset.

        Args:
            current_diameter: Current pupil diameter in mm
            baseline_diameter: Baseline diameter (zero decentration) in mm
            x_coeff: X coefficient (mm/mm)
            y_coeff: Y coefficient (mm/mm)
            z_coeff: Axial (optical-axis) coefficient (mm/mm); 0 keeps the pupil plane fixed

        Returns:
            Position3D offset for pupil decentration

        """
        if current_diameter <= 0:
            raise ValueError(f"Current pupil diameter must be positive, got {current_diameter}")
        if baseline_diameter <= 0:
            raise ValueError(f"Baseline pupil diameter must be positive, got {baseline_diameter}")

        diameter_change = current_diameter - baseline_diameter
        x_offset = x_coeff * diameter_change
        y_offset = y_coeff * diameter_change
        z_offset = z_coeff * diameter_change

        return Position3D(x_offset, y_offset, z_offset)


@dataclass
class PupilDecentrationConfig:
    """Configuration for pupil decentration.

    Attributes:
        enabled: Whether decentration is enabled
        model_name: Name of registered decentration model to use
        baseline_diameter: Diameter at which decentration is zero (auto-set if None)
        x_coeff: X coefficient (mm/mm); resolve_for_eye() fills it from the eye-specific
            population mean when left None
        y_coeff: Y coefficient (mm/mm); resolved the same way
        use_individual_variation: If True, sample coefficients from distribution
        individual_seed: Seed for reproducible individual variation
        preserve_anatomical_direction: If True (default), variation only increases
            magnitude in the anatomical direction (no sign flip - all subjects shift
            in the same direction, just by different amounts). If False, uses symmetric
            normal distribution (mean ± std) matching paper's reported statistics exactly.
            Note: The paper reports mean ± std, so set to False to match paper statistics.

    """

    enabled: bool = False
    model_name: str = "wildenmann_2013"
    baseline_diameter: float | None = None  # Auto-set to current diameter if None

    # Common parameters for built-in models
    x_coeff: float | None = None
    y_coeff: float | None = None
    z_coeff: float = 0.0  # Axial (optical-axis) shift per mm of diameter change; 0 = in-plane decentration only

    # Individual variation from Wildenmann & Schaeffel (2013): 0.044-0.179 mm per mm
    use_individual_variation: bool = False
    individual_seed: int | None = None  # Seed for generating individual profile
    # Paper reports mean ± std; set to False to match paper statistics exactly
    preserve_anatomical_direction: bool = True  # If True, variation only increases magnitude (no sign flip)

    def resolve_for_eye(self, which_eye: str) -> None:
        """Fill x_coeff/y_coeff from the population model for ``which_eye`` when not given explicitly.

        Pupil decentration is eye-specific (the left and right horizontal coefficients have opposite sign),
        so the owning Eye supplies its side. A no-op when the coefficients were set directly or decentration
        is disabled. The Eye calls this once during construction.
        """
        explicit = self.x_coeff is not None or self.y_coeff is not None
        if explicit or not (self.enabled or self.use_individual_variation):
            return

        if which_eye == "left":
            x_mean, x_std = PupilDecentrationDefaults.LEFT_EYE_X_COEFF, PupilDecentrationDefaults.LEFT_EYE_X_STD
            y_mean, y_std = PupilDecentrationDefaults.LEFT_EYE_Y_COEFF, PupilDecentrationDefaults.LEFT_EYE_Y_STD
        else:
            x_mean, x_std = PupilDecentrationDefaults.RIGHT_EYE_X_COEFF, PupilDecentrationDefaults.RIGHT_EYE_X_STD
            y_mean, y_std = PupilDecentrationDefaults.RIGHT_EYE_Y_COEFF, PupilDecentrationDefaults.RIGHT_EYE_Y_STD

        if not self.use_individual_variation:
            self.x_coeff, self.y_coeff = x_mean, y_mean
            return

        rng = np.random.default_rng(self.individual_seed)
        if self.preserve_anatomical_direction:
            # Variation grows magnitude in the anatomical direction only; abs() keeps the sign fixed.
            self.x_coeff = x_mean + abs(rng.normal(0, x_std)) * (1 if x_mean >= 0 else -1)
            self.y_coeff = y_mean + abs(rng.normal(0, y_std)) * (1 if y_mean >= 0 else -1)
        else:  # symmetric mean ± N(0, std), matching the paper's reported statistics
            self.x_coeff = x_mean + rng.normal(0, x_std)
            self.y_coeff = y_mean + rng.normal(0, y_std)

    def get_model_params(self) -> dict:
        """Get parameters to pass to model's calculate_offset method."""
        return {"x_coeff": self.x_coeff, "y_coeff": self.y_coeff, "z_coeff": self.z_coeff}


def register_custom_decentration(name: str, calculation_func: Callable) -> None:
    """Register a custom decentration model.

    Args:
        name: Unique name for the model
        calculation_func: Function with signature:
            (current_diameter: float, baseline_diameter: float, **kwargs) -> Position3D

    Example:
        def my_custom_model(current_diameter, baseline_diameter, **kwargs):
            # Your custom decentration logic here
            return Position3D(x_offset, y_offset, 0.0)

        register_custom_decentration("my_lab_model", my_custom_model)

    """

    class CustomModel(PupilDecentrationModel):
        @property
        def name(self) -> str:
            return name

        def calculate_offset(self, current_diameter: float, baseline_diameter: float, **kwargs: float) -> Position3D:  # noqa: PLR6301
            return calculation_func(current_diameter, baseline_diameter, **kwargs)

    PupilDecentrationRegistry.register(CustomModel())


# Register built-in models
PupilDecentrationRegistry.register(WildenmannModel())
