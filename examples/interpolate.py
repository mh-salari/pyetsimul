"""Polynomial interpolation eye tracking example.

Demonstrates polynomial interpolation-based gaze tracking with comprehensive accuracy evaluation.
"""

from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.evaluation import (
    accuracy_over_observer_positions,
    accuracy_at_calibration_points,
)
from pyetsimul.evaluation.gaze_movement import accuracy_over_gaze_points
from pyetsimul.experimental_designs import GazeMovement
from pyetsimul.evaluation.analysis_utils import print_error_summary
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
    print_error_summary(calib_results, "Calibration Test Summary")

    print("2. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)

    # Create 2D gaze movement pattern: X and Z vary (XZ plane), Y fixed
    # This matches the calibration plane which is XZ
    gaze_movement = GazeMovement(
        grid_center=Position3D(0, 0, 200e-3),
        dx=[-200e-3, 200e-3],  # X varies: ±200mm
        dy=[0.0, 0.0],  # Y fixed: no variation
        dz=[-150e-3, 150e-3],  # Z varies: ±150mm
        grid_size=[16, 1, 16],  # 16x1x16 = 2D grid in XZ plane
    )
    screen_results = accuracy_over_gaze_points(et, eye=eye, gaze_movement=gaze_movement)

    print_error_summary(screen_results, "Screen Test Summary")

    print("\n3. Testing over observer (fixed gaze, sweep observer positions):")
    print("-" * 60)
    # Use a fixed gaze target for observer position testing
    gaze_target = Position3D(0, 0, 200e-3)  # Center of the screen
    observer_results = accuracy_over_observer_positions(et, eye=eye, gaze_target=gaze_target)

    print_error_summary(observer_results, "Observer Test Summary")


if __name__ == "__main__":
    main()
