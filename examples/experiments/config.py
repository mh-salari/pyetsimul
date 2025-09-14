"""Experiment configurations and variations for data generation.

This module defines:
- A single shared hardware configuration (eye, camera, light)
- Reusable variation definitions
- Import variations directly into your experiments
"""

from pathlib import Path

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.simulation import (
    AngleKappaVariation,
    ComposedVariation,
    CorneaRadiusVariation,
    CorneaThicknessVariation,
    ExperimentConfig,
    EyePositionVariation,
    PupilSizeVariation,
    TargetPositionVariation,
)
from pyetsimul.types import Position3D, RotationMatrix


def create_experiment_config(name: str) -> ExperimentConfig:
    """Create experiment config with standard hardware setup."""
    # Standard eye configuration
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = Position3D(0.0, 550e-3, 350e-3)

    # Standard camera configuration
    camera = Camera(err=0.0, err_type="gaussian")
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    camera.point_at(eye.position)

    # Standard light configuration
    light = Light(position=Position3D(200e-3, 0, 350e-3))

    return ExperimentConfig(
        experiment_name=name,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=Position3D(0.0, 0.0, 200e-3),
        output_dir=Path(__file__).parent / "outputs",
    )


# 3x3 calibration grid on XZ plane
calibration_points = [
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


# Single parameter variations
eye_position_variation = EyePositionVariation(
    center=Position3D(0.0, 550e-3, 350e-3),
    dx=[-50e-3, 50e-3],
    dy=[-50e-3, 50e-3],
    dz=[0.0, 0.0],
    grid_size=[16, 16, 1],
)

target_position_variation = TargetPositionVariation(
    grid_center=Position3D(0, 0, 200e-3),
    dx=[-200e-3, 200e-3],
    dy=[0.0, 0.0],
    dz=[-150e-3, 150e-3],
    grid_size=[16, 1, 16],
)

pupil_size_variation = PupilSizeVariation(
    diameter_range=[3.0e-3, 7.5e-3],
    num_steps=10,
)

angle_kappa_variation = AngleKappaVariation(
    alpha_range_deg=[4.0, 8.0],
    beta_range_deg=[1.0, 3.0],
    num_steps=10,
)

corneal_radius_variation = CorneaRadiusVariation([7.5e-3, 8.5e-3], 20)

corneal_thickness_variation = CorneaThicknessVariation([0.4e-3, 0.7e-3], 15)

# Multi-parameter variations
angle_kappa_pupil_variation = ComposedVariation(
    [
        AngleKappaVariation([3.0, 8.0], [0.0, 3.0], 8),
        PupilSizeVariation([2.0e-3, 8.0e-3], 8),
    ],
    "angle_kappa_pupil",
)

eye_position_angle_kappa_variation = ComposedVariation(
    [
        EyePositionVariation(
            center=Position3D(0.0, 550e-3, 350e-3),
            dx=[-30e-3, 30e-3],
            dy=[-30e-3, 30e-3],
            dz=[0.0, 0.0],
            grid_size=[6, 6, 1],
        ),
        AngleKappaVariation([4.0, 7.0], [1.0, 3.0], 6),
    ],
    "eye_position_angle_kappa",
)

corneal_shape_variation = ComposedVariation(
    [
        CorneaRadiusVariation([7.6e-3, 8.4e-3], 12),
        CorneaThicknessVariation([0.45e-3, 0.65e-3], 12),
    ],
    "corneal_shape",
)
