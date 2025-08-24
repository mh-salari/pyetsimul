"""Parameter variation framework for systematic eye tracking studies."""

from .core import ParameterVariation, VariationStrategy
from .spatial.eye_position import EyePositionVariation
from .spatial.target_position import TargetPositionVariation
from .strategies.data_generation import DataGenerationStrategy
from .strategies.algorithm_comparison import AlgorithmComparisonStrategy
from .strategies.evaluation import EyePositionEvaluationStrategy
from .strategies.target_evaluation import TargetPositionEvaluationStrategy
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
    "EyePositionVariation",
    "TargetPositionVariation",
    "DataGenerationStrategy",
    "AlgorithmComparisonStrategy",
    "EyePositionEvaluationStrategy",
    "TargetPositionEvaluationStrategy",
    "EyePositionConfig",
    "TargetPositionConfig",
    "create_eye_position_config",
    "create_target_position_config",
    "validate_config",
    "run_experiments",
    "run_single_config",
    "run_all_configs",
]
