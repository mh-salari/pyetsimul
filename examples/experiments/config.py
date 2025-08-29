"""Experiment configurations for data generation."""

from pathlib import Path
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.experiment_framework.data_generation import (
    ExperimentConfig,
    EyePositionVariation,
    PupilSizeVariation,
    TargetPositionVariation,
)


default_gaze_target = Position3D(0.0, 0.0, 200e-3)
output_dir = Path(__file__).parent / "outputs"


def create_hardware_setup():
    """Create standard eye tracking hardware setup."""
    # Eye configuration
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = Position3D(0.0, 550e-3, 350e-3)

    # Camera configuration
    camera = Camera(err=0.0, err_type="gaussian")
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    camera.point_at(eye.position)

    # Light configuration
    light = Light(position=Position3D(200e-3, 0, 350e-3))

    return eye, camera, light


def create_calibration_points():
    """Create standard calibration grid."""
    return [
        Position3D(-200e-3, 0.0, 50e-3),
        Position3D(0, 0.0, 50e-3),
        Position3D(200e-3, 0.0, 50e-3),
        Position3D(-200e-3, 0.0, 200e-3),
        Position3D(0, 0.0, 200e-3),
        Position3D(200e-3, 0.0, 200e-3),
        Position3D(-200e-3, 0.0, 350e-3),
        Position3D(0, 0.0, 350e-3),
        Position3D(200e-3, 0.0, 350e-3),
    ]


# Configuration 1: Eye position variation
def create_eye_position_config():
    """Configuration for eye position variation experiment."""
    eye, camera, light = create_hardware_setup()

    variation = EyePositionVariation(
        center=eye.position,  # Central eye position
        dx=[-50e-3, 50e-3],  # X varies: ±50mm
        dy=[-50e-3, 50e-3],  # Y varies: ±50mm
        dz=[0.0, 0.0],  # Z fixed: no variation
        grid_size=[16, 16, 1],  # 16x16x1 = 2D grid in XY plane
    )

    return ExperimentConfig(
        experiment_name="eye_position",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )


# Configuration 2: Gaze movement variation
def create_gaze_movement_config():
    """Configuration for gaze movement experiment."""
    eye, camera, light = create_hardware_setup()

    variation = TargetPositionVariation(
        grid_center=Position3D(0, 0, 200e-3),
        dx=[-200e-3, 200e-3],  # X varies: ±200mm
        dy=[0.0, 0.0],  # Y fixed: no variation
        dz=[-150e-3, 150e-3],  # Z varies: ±150mm
        grid_size=[16, 1, 16],  # 16x1x16 = 2D grid in XZ plane
    )

    return ExperimentConfig(
        experiment_name="gaze_movement",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )


# Configuration 3: Pupil size variation
def create_pupil_size_config():
    """Configuration for pupil size variation experiment."""
    eye, camera, light = create_hardware_setup()

    variation = PupilSizeVariation(
        diameter_range=[3.0e-3, 7.5e-3],  # 3mm to 7.5mm
        num_steps=10,  # 10 different sizes
    )

    return ExperimentConfig(
        experiment_name="pupil_size",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )
