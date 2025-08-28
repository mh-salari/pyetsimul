"""Pure data generation for eye tracking simulation."""

from .core import ParameterVariation, EyeParameterVariation, TargetVariation, VariationStrategy
from .spatial.eye_position import EyePositionVariation
from .spatial.target_position import TargetPositionVariation
from .spatial.grid_base import GridGenerator, RegularGrid, RandomGrid
from .composed_variation import ComposedVariation, SequentialVariation
from .physical.pupil_size import PupilSizeVariation
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
    "PupilSizeVariation",
    "DataGenerationStrategy",
    "ExperimentConfig",
    "create_experiment_config",
    "validate_config",
]
