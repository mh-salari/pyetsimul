"""Polynomial interpolation eye tracking example.

Demonstrates polynomial interpolation-based gaze tracking with comprehensive accuracy evaluation.
"""

from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.parameter_variations import (
    EyePositionVariation,
    TargetPositionVariation,
    EyePositionEvaluationStrategy,
    TargetPositionEvaluationStrategy,
)
from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points
from pyetsimul.core import Light, Camera, Eye
from pyetsimul.types import Position3D, RotationMatrix
from tabulate import tabulate


def main():
    """Run polynomial interpolation eye tracking demonstration with comprehensive evaluation."""
    print("Python Interpolate Test (System Integration)\n")

    # Eye position
    eye_position = Position3D(0, 550e-3, 350e-3)

    # Use validate_handedness=False for legacy MATLAB coordinate system compatibility

    # Create eye configuration with structured types
    eye = Eye()
    # Use validate_handedness=False for legacy MATLAB coordinate system compatibility
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = eye_position

    # Create camera configuration with structured types
    cam = Camera(err=0.0, err_type="gaussian")
    # Use validate_handedness=False for legacy MATLAB coordinate system compatibility
    cam.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    # Point camera at the eye position
    cam.point_at(eye.position)

    # Create light configuration with structured type
    light = Light(position=Position3D(200e-3, 0, 350e-3))

    # Create calibration grid using structured Position3D objects
    # Original format was [x, z] pairs, now converted to Position3D(x, y=0.0, z)
    calib_points = [
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

    # Setup tracker with default method
    method = "cerrolaza_2008"
    et = InterpolationTracker.create([cam], [light], calib_points, method)
    # Set legacy optical-then-kappa behavior for evaluation with the original MATLAB behavior
    et.use_legacy_look_at = True

    # Display configuration summary
    print("Configuration Summary:")
    headers = ["Component", "Parameter", "Value", "Unit"]
    data = [
        [
            "Eye",
            "Position (x, y, z)",
            f"({eye_position.x * 1000:.1f}, {eye_position.y * 1000:.1f}, {eye_position.z * 1000:.1f})",
            "mm",
        ],
        [
            "Camera",
            "Position (x, y, z)",
            f"({cam.position.x * 1000:.1f}, {cam.position.y * 1000:.1f}, {cam.position.z * 1000:.1f})",
            "mm",
        ],
        [
            "Light",
            "Position (x, y, z)",
            f"({light.position.x * 1000:.1f}, {light.position.y * 1000:.1f}, {light.position.z * 1000:.1f})",
            "mm",
        ],
        ["Algorithm", "Method", method, "-"],
        ["Calibration", "Points", f"{len(calib_points)}", "points"],
    ]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    print()

    # Calibrate the eye tracker once
    print("Calibrating eye tracker...")
    et.run_calibration(eye)

    print("1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=eye)
    calib_results.print_summary("Calibration Test Summary")

    print("2. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)

    # Create target position variation using new architecture
    target_position_variation = TargetPositionVariation(
        grid_center=Position3D(0, 0, 200e-3),
        dx=[-200e-3, 200e-3],  # X varies: ±200mm
        dy=[0.0, 0.0],  # Y fixed: no variation
        dz=[-150e-3, 150e-3],  # Z varies: ±150mm
        grid_size=[16, 1, 16],  # 16x1x16 = 2D grid in XZ plane
    )

    # Use new target position evaluation strategy
    target_strategy = TargetPositionEvaluationStrategy(observer_position=eye_position)
    screen_results = target_strategy.execute(eye, et, target_position_variation)

    screen_results.print_summary("Screen Test Summary")

    print("\n3. Testing over observer (fixed gaze, sweep observer positions):")
    print("-" * 60)
    # Create eye position variation using new architecture
    eye_position_variation = EyePositionVariation(
        center=eye_position,  # Central eye position
        dx=[-50e-3, 50e-3],  # X varies: ±50mm
        dy=[-50e-3, 50e-3],  # Y varies: ±50mm
        dz=[0.0, 0.0],  # Z fixed: no variation
        grid_size=[16, 16, 1],  # 16x16x1 = 2D grid in XY plane
    )

    # Use new eye position evaluation strategy
    evaluation_strategy = EyePositionEvaluationStrategy(gaze_target=Position3D(0, 0, 200e-3))
    observer_results = evaluation_strategy.execute(eye, et, eye_position_variation)

    observer_results.print_summary("Observer Test Summary")


if __name__ == "__main__":
    main()
