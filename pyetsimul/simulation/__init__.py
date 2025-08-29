"""Pure data generation for eye tracking simulation."""

from .core import ParameterVariation, EyeParameterVariation, TargetVariation, VariationStrategy
from .eye_position import EyePositionVariation
from .target_position import TargetPositionVariation
from .grid_base import GridGenerator, RegularGrid, RandomGrid
from .composed_variation import ComposedVariation, SequentialVariation
from .generic import GenericEyeVariation
from .anatomy import (
    PupilSizeVariation,
    AngleKappaVariation,
    CorneaRadiusVariation,
    CorneaThicknessVariation,
)
from .data_generation import DataGenerationStrategy
from .config import (
    ExperimentConfig,
    create_experiment_config,
    validate_config,
)

__all__ = [
    "ParameterVariation",
    "EyeParameterVariation",
    "TargetVariation",
    "VariationStrategy",
    "EyePositionVariation",
    "TargetPositionVariation",
    "GridGenerator",
    "RegularGrid",
    "RandomGrid",
    "ComposedVariation",
    "SequentialVariation",
    "GenericEyeVariation",
    "PupilSizeVariation",
    "AngleKappaVariation",
    "CorneaRadiusVariation",
    "CorneaThicknessVariation",
    "DataGenerationStrategy",
    "ExperimentConfig",
    "create_experiment_config",
    "validate_config",
]
