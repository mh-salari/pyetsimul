"""Experiment configurations and variations for data generation.

This module defines:
- A single shared hardware configuration (eye, camera, light)
- Reusable variation definitions
- Import variations directly into your experiments
"""

from pathlib import Path

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.pupil_decentration import PupilDecentrationConfig
from pyetsimul.simulation import (
    AngleKappaVariation,
    ComposedVariation,
    CorneaRadiusVariation,
    CorneaThicknessVariation,
    ExperimentConfig,
    EyePositionVariation,
    PupilDecentrationVariation,
    PupilSizeVariation,
    PupilSizeWithDecentrationVariation,
    TargetPositionVariation,
)
from pyetsimul.types import Position3D, RotationMatrix


def create_experiment_config(name: str) -> ExperimentConfig:
    """Create experiment config with standard hardware setup."""
    # Standard eye configuration
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = Position3D(0.0, 550, 350)

    # Standard camera configuration
    camera = Camera(err=0.0, err_type="gaussian")
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    camera.point_at(eye.position)

    # Standard light configuration
    light = Light(position=Position3D(200, 0, 350))

    return ExperimentConfig(
        experiment_name=name,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=Position3D(0.0, 0.0, 200),
        output_dir=Path(__file__).parent / "outputs",
    )


def create_homography_experiment_config(name: str) -> ExperimentConfig:
    """Create experiment config for homography normalization (requires 4+ lights)."""
    # Standard eye configuration (same as regular config)
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = Position3D(0.0, 550, 350)

    # Standard camera configuration (same as regular config)
    camera = Camera(err=0.0, err_type="gaussian")
    camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    camera.point_at(eye.position)

    # Four lights at corners of calibration grid
    lights = [
        Light(position=Position3D(-200, 0.0, 50)),  # Bottom-left corner
        Light(position=Position3D(200, 0.0, 50)),  # Bottom-right corner
        Light(position=Position3D(200, 0.0, 350)),  # Top-right corner
        Light(position=Position3D(-200, 0.0, 350)),  # Top-left corner
    ]

    return ExperimentConfig(
        experiment_name=name,
        eyes=[eye],
        cameras=[camera],
        lights=lights,  # Four lights instead of one
        gaze_target=Position3D(0.0, 0.0, 200),
        output_dir=Path(__file__).parent / "outputs",
    )


# 3x3 calibration grid on XZ plane
calibration_points = [
    Position3D(-200, 0.0, 50),
    Position3D(0, 0.0, 50),
    Position3D(200, 0.0, 50),
    Position3D(-200, 0.0, 200),
    Position3D(0, 0.0, 200),
    Position3D(200, 0.0, 200),
    Position3D(-200, 0.0, 350),
    Position3D(0, 0.0, 350),
    Position3D(200, 0.0, 350),
]


# Single parameter variations
eye_position_variation = EyePositionVariation(
    center=Position3D(0.0, 550, 350),
    dx=[-50, 50],
    dy=[-50, 50],
    dz=[0.0, 0.0],
    grid_size=[16, 16, 1],
)

target_position_variation = TargetPositionVariation(
    grid_center=Position3D(0, 0, 200),
    dx=[-200, 200],
    dy=[0.0, 0.0],
    dz=[-150, 150],
    grid_size=[16, 1, 16],
)

pupil_size_variation = PupilSizeVariation(
    diameter_range=[3.0, 7.5],
    num_steps=10,
)

angle_kappa_variation = AngleKappaVariation(
    alpha_range_deg=[4.0, 8.0],
    beta_range_deg=[1.0, 3.0],
    num_steps=10,
)

corneal_radius_variation = CorneaRadiusVariation([7.5, 8.5], 20)

corneal_thickness_variation = CorneaThicknessVariation([0.4, 0.7], 15)

pupil_size_with_decentration_variation = PupilSizeWithDecentrationVariation(
    diameter_range=[3.0, 7.5],
    decentration_config=PupilDecentrationConfig(
        enabled=True,
        model_name="wildenmann_2013",
        baseline_diameter=4.75,  # Wildenmann baseline
    ),
    num_steps=10,
)

pupil_decentration_variation = PupilDecentrationVariation(
    dx_range=[-0.1, 0.1],  # ±0.1mm horizontal - realistic dynamic range per Wildenmann 2013
    dy_range=[-0.15, 0.15],  # ±0.15mm vertical - ~0.05mm shift per mm pupil size change
    num_steps=10,
)

# Multi-parameter variations
angle_kappa_pupil_variation = ComposedVariation(
    [
        AngleKappaVariation([3.0, 8.0], [0.0, 3.0], 8),
        PupilSizeVariation([2.0, 8.0], 8),
    ],
    "angle_kappa_pupil",
)

eye_position_angle_kappa_variation = ComposedVariation(
    [
        EyePositionVariation(
            center=Position3D(0.0, 550, 350),
            dx=[-30, 30],
            dy=[-30, 30],
            dz=[0.0, 0.0],
            grid_size=[6, 6, 1],
        ),
        AngleKappaVariation([4.0, 7.0], [1.0, 3.0], 6),
    ],
    "eye_position_angle_kappa",
)

corneal_shape_variation = ComposedVariation(
    [
        CorneaRadiusVariation([7.6, 8.4], 12),
        CorneaThicknessVariation([0.45, 0.65], 12),
    ],
    "corneal_shape",
)
