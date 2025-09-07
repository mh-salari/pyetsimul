"""Generic configuration system for parameter variation experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..core import Eye, Camera, Light
from ..types import Position3D


@dataclass
class ExperimentConfig:
    """Generic configuration for any parameter variation experiment."""

    # Required experiment metadata
    experiment_name: str

    # Required hardware setup
    eyes: list[Eye]
    cameras: list[Camera]
    lights: list[Light]

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

    def get_gaze_target(self) -> Optional[Position3D]:
        """Get the gaze target if specified."""
        return self.gaze_target


# Helper factory functions for common setups
def create_experiment_config(
    experiment_name: str,
    eyes: list[Eye],
    cameras: list[Camera],
    lights: list[Light],
    gaze_target: Optional[Position3D] = None,
    output_dir: str | Path = "outputs",
) -> ExperimentConfig:
    """Create experiment configuration with validation."""
    return ExperimentConfig(
        experiment_name=experiment_name,
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
