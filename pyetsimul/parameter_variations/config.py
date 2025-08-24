"""Configuration templates for parameter variation experiments."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Union
from ..core import Eye, Camera, Light
from ..types import Position3D
from .spatial.eye_position import Eye3DPositionVariation
from .spatial.target_position import Target3DPositionVariation


@dataclass
class EyePositionConfig:
    """Configuration template for eye position variation experiments."""

    # Required experiment metadata
    experiment_name: str
    experiment_type: str = "eye_position_variation"
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Required hardware setup
    eyes: List[Eye] = field(default_factory=list)
    cameras: List[Camera] = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)

    # Required parameter variation
    eye_variation: Eye3DPositionVariation = None
    gaze_target: Position3D = None

    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.eyes:
            raise ValueError("Must specify at least one eye")
        if not self.cameras:
            raise ValueError("Must specify at least one camera")
        if not self.lights:
            raise ValueError("Must specify at least one light")
        if self.eye_variation is None:
            raise ValueError("Must specify eye_variation")
        if self.gaze_target is None:
            raise ValueError("Must specify gaze_target")


@dataclass
class TargetPositionConfig:
    """Configuration template for target position variation experiments."""

    # Required experiment metadata
    experiment_name: str
    experiment_type: str = "target_position_variation"
    output_dir: Path = field(default_factory=lambda: Path("outputs"))

    # Required hardware setup
    eyes: List[Eye] = field(default_factory=list)
    cameras: List[Camera] = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)

    # Required parameter variation
    target_variation: Target3DPositionVariation = None

    def __post_init__(self):
        """Validate required fields after initialization."""
        if not self.eyes:
            raise ValueError("Must specify at least one eye")
        if not self.cameras:
            raise ValueError("Must specify at least one camera")
        if not self.lights:
            raise ValueError("Must specify at least one light")
        if self.target_variation is None:
            raise ValueError("Must specify target_variation")


# Factory functions for easy config creation
def create_eye_position_config(
    experiment_name: str,
    eye_center: Position3D,
    gaze_target: Position3D,
    dx: List[float],
    dy: List[float],
    dz: List[float],
    grid_size: List[int],
    eyes: List[Eye],
    cameras: List[Camera],
    lights: List[Light],
    output_dir: Union[str, Path] = "outputs",
) -> EyePositionConfig:
    """Create an eye position variation config with validation."""

    eye_variation = Eye3DPositionVariation(center=eye_center, dx=dx, dy=dy, dz=dz, grid_size=grid_size)

    return EyePositionConfig(
        experiment_name=experiment_name,
        eyes=eyes,
        cameras=cameras,
        lights=lights,
        eye_variation=eye_variation,
        gaze_target=gaze_target,
        output_dir=Path(output_dir),
    )


def create_target_position_config(
    experiment_name: str,
    grid_center: Position3D,
    dx: List[float],
    dy: List[float],
    dz: List[float],
    grid_size: List[int],
    eyes: List[Eye],
    cameras: List[Camera],
    lights: List[Light],
    output_dir: Union[str, Path] = "outputs",
) -> TargetPositionConfig:
    """Create a target position variation config with validation."""

    target_variation = Target3DPositionVariation(grid_center=grid_center, dx=dx, dy=dy, dz=dz, grid_size=grid_size)

    return TargetPositionConfig(
        experiment_name=experiment_name,
        eyes=eyes,
        cameras=cameras,
        lights=lights,
        target_variation=target_variation,
        output_dir=Path(output_dir),
    )


def validate_config(config: Union[EyePositionConfig, TargetPositionConfig]) -> None:
    """Validate a config object and provide helpful error messages."""
    try:
        # The dataclass __post_init__ will validate required fields
        pass
    except ValueError as e:
        raise ValueError(f"Config validation failed: {e}")

    # Additional validation
    if isinstance(config, EyePositionConfig):
        if len(config.eye_variation.grid_size) != 3:
            raise ValueError("eye_variation.grid_size must have exactly 3 elements [nx, ny, nz]")
        if len(config.eye_variation.dx) != 2:
            raise ValueError("eye_variation.dx must have exactly 2 elements [min, max]")
        if len(config.eye_variation.dy) != 2:
            raise ValueError("eye_variation.dy must have exactly 2 elements [min, max]")
        if len(config.eye_variation.dz) != 2:
            raise ValueError("eye_variation.dz must have exactly 2 elements [min, max]")

    elif isinstance(config, TargetPositionConfig):
        if len(config.target_variation.grid_size) != 3:
            raise ValueError("target_variation.grid_size must have exactly 3 elements [nx, ny, nz]")
        if len(config.target_variation.dx) != 2:
            raise ValueError("target_variation.dx must have exactly 2 elements [min, max]")
        if len(config.target_variation.dy) != 2:
            raise ValueError("target_variation.dy must have exactly 2 elements [min, max]")
        if len(config.target_variation.dz) != 2:
            raise ValueError("target_variation.dz must have exactly 2 elements [min, max]")
