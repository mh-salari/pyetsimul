"""Parameter variation framework for systematic eye tracking studies."""

from .core import ParameterVariation, VariationStrategy
from .spatial.eye_position import Eye3DPositionVariation
from .spatial.target_position import Target3DPositionVariation
from .strategies.data_generation import DataGenerationStrategy
from .config import (
    EyePositionConfig,
    TargetPositionConfig,
    create_eye_position_config,
    create_target_position_config,
    validate_config,
)
from .experiment_runner import run_experiments, run_single_config, run_all_configs

__all__ = [
    "ParameterVariation",
    "VariationStrategy",
    "Eye3DPositionVariation",
    "Target3DPositionVariation",
    "DataGenerationStrategy",
    "EyePositionConfig",
    "TargetPositionConfig",
    "create_eye_position_config",
    "create_target_position_config",
    "validate_config",
    "run_experiments",
    "run_single_config",
    "run_all_configs",
]
