"""Experiment framework for eye tracking simulation."""

# Import everything from data_generation submodule
from .data_generation import (
    ParameterVariation,
    EyeParameterVariation,
    TargetVariation,
    VariationStrategy,
    EyePositionVariation,
    TargetPositionVariation,
    GridGenerator,
    RegularGrid,
    RandomGrid,
    ComposedVariation,
    SequentialVariation,
    PupilSizeVariation,
    DataGenerationStrategy,
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
