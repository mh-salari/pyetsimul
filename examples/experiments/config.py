"""Experiment configurations for data generation."""

from pathlib import Path
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.simulation import (
    ExperimentConfig,
    EyePositionVariation,
    PupilSizeVariation,
    TargetPositionVariation,
    AngleKappaVariation,
    CorneaRadiusVariation,
    CorneaThicknessVariation,
    ComposedVariation,
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


# Configuration 4: Angle kappa variation
def create_angle_kappa_config():
    """Configuration for angle kappa variation experiment."""
    eye, camera, light = create_hardware_setup()

    variation = AngleKappaVariation(
        alpha_range_deg=[4.0, 8.0],  # Horizontal angle kappa: 4-8°
        beta_range_deg=[1.0, 3.0],  # Vertical angle kappa: 1-3°
        num_steps=10,  # 10×10 = 100 combinations
    )

    return ExperimentConfig(
        experiment_name="angle_kappa",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )


# Configuration 5: Corneal radius variation
def create_corneal_radius_config():
    """Configuration for corneal radius variation experiment."""
    eye, camera, light = create_hardware_setup()

    variation = CorneaRadiusVariation([7.5e-3, 8.5e-3], 20)

    return ExperimentConfig(
        experiment_name="corneal_radius",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )


# Configuration 6: Corneal thickness variation
def create_corneal_thickness_config():
    """Configuration for corneal thickness variation experiment."""
    eye, camera, light = create_hardware_setup()

    variation = CorneaThicknessVariation([0.4e-3, 0.7e-3], 15)

    return ExperimentConfig(
        experiment_name="corneal_thickness",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )


# Configuration 7: Individual differences (Angle kappa + Pupil size)
def create_individual_differences_config():
    """Configuration combining angle kappa and pupil size variations."""
    eye, camera, light = create_hardware_setup()

    individual_variations = [
        AngleKappaVariation([3.0, 8.0], [0.0, 3.0], 8),  # 8×8 = 64 combinations
        PupilSizeVariation([2.0e-3, 8.0e-3], 8),  # Extended range: 2-8mm
    ]
    variation = ComposedVariation(individual_variations, "individual_differences")

    return ExperimentConfig(
        experiment_name="individual_differences",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )


# Configuration 8: Observer movement with individual differences
def create_observer_individual_config():
    """Configuration combining eye position and angle kappa variations."""
    eye, camera, light = create_hardware_setup()

    observer_variations = [
        EyePositionVariation(
            center=eye.position,
            dx=[-30e-3, 30e-3],  # ±30mm horizontal movement
            dy=[-30e-3, 30e-3],  # ±30mm vertical movement
            dz=[0.0, 0.0],  # Fixed depth
            grid_size=[6, 6, 1],  # 6×6×1 = 36 positions
        ),
        AngleKappaVariation([4.0, 7.0], [1.0, 3.0], 6),  # 6×6 = 36 combinations
    ]
    variation = ComposedVariation(observer_variations, "observer_individual")

    return ExperimentConfig(
        experiment_name="observer_individual",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )


# Configuration 9: Corneal shape variation (Radius + Thickness)
def create_corneal_shape_config():
    """Configuration combining corneal radius and thickness variations."""
    eye, camera, light = create_hardware_setup()

    corneal_variations = [
        CorneaRadiusVariation([7.6e-3, 8.4e-3], 12),  # Physiological range
        CorneaThicknessVariation([0.45e-3, 0.65e-3], 12),  # Normal population range
    ]
    variation = ComposedVariation(corneal_variations, "corneal_shape")

    return ExperimentConfig(
        experiment_name="corneal_shape",
        variation=variation,
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=default_gaze_target,
        output_dir=output_dir,
    )
