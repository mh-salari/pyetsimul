"""Generic configuration system for parameter variation experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union, Optional

from ..core import Eye, Camera, Light
from ..types import Position3D
from .core import ParameterVariation


@dataclass
class ExperimentConfig:
    """Generic configuration for any parameter variation experiment."""

    # Required experiment metadata
    experiment_name: str
    variation: ParameterVariation

    # Required hardware setup
    eyes: List[Eye]
    cameras: List[Camera]
    lights: List[Light]

    # Optional configuration
    gaze_target: Optional[Position3D] = None
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    def __post_init__(self):
        """Validate configuration."""
        if not self.experiment_name:
            raise ValueError("experiment_name cannot be empty")
        if not self.eyes:
            raise ValueError("Must specify at least one eye")
        if not self.cameras:
            raise ValueError("Must specify at least one camera")
        if not self.lights:
            raise ValueError("Must specify at least one light")
        if self.variation is None:
            raise ValueError("Must specify a parameter variation")

    @property
    def experiment_type(self) -> str:
        """Get experiment type from the variation."""
        return f"{self.variation.param_name}_variation"

    def get_variation(self) -> ParameterVariation:
        """Get the parameter variation object."""
        return self.variation

    def get_gaze_target(self) -> Optional[Position3D]:
        """Get the gaze target if specified."""
        return self.gaze_target


# Helper factory functions for common setups
def create_experiment_config(
    experiment_name: str,
    variation: ParameterVariation,
    eyes: List[Eye],
    cameras: List[Camera],
    lights: List[Light],
    gaze_target: Optional[Position3D] = None,
    output_dir: Union[str, Path] = "outputs",
) -> ExperimentConfig:
    """Create experiment configuration with validation."""
    return ExperimentConfig(
        experiment_name=experiment_name,
        variation=variation,
        eyes=eyes,
        cameras=cameras,
        lights=lights,
        gaze_target=gaze_target,
        output_dir=Path(output_dir),
    )


def validate_config(config: ExperimentConfig) -> None:
    """Validate experiment configuration."""
    # Validation is automatic via __post_init__, just ensure config exists
    if not config:
        raise ValueError("Config cannot be None")
