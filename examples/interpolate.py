"""Polynomial interpolation eye tracking example.

Demonstrates polynomial interpolation-based gaze tracking with comprehensive accuracy evaluation.
"""

from et_simul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from et_simul.evaluation import (
    accuracy_over_gaze_points,
    accuracy_over_observer_positions,
    accuracy_at_calibration_points,
)
from et_simul.evaluation.analysis_utils import print_error_summary
from et_simul.core import Light, Camera, Eye
from et_simul.types import Position3D, RotationMatrix
import numpy as np
import argparse


def main():
    """Run polynomial interpolation eye tracking demonstration with comprehensive evaluation."""
    parser = argparse.ArgumentParser(description="Interpolation Eye Tracker Test")
    parser.add_argument(
        "--eye-position",
        type=float,
        nargs=3,
        default=[0, 550e-3, 350e-3],
        metavar=("X", "Y", "Z"),
        help="Eye position in meters (default: 0 250e-3 350e-3)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cerrolaza_2008",
        choices=[
            "cerrolaza_2008",
            "hennessey_2008",
            "second_order",
            "zhu_ji_2005",
            "blignaut_wium_2013",
        ],
        help="Interpolation method (default: cerrolaza_2008)",
    )

    args = parser.parse_args()

    print("=== Python Interpolate Test (System Integration) ===\n")

    # Create eye position using structured type
    eye_position = Position3D(args.eye_position[0], args.eye_position[1], args.eye_position[2])
    print(
        f"Using eye position: X={eye_position.x * 1000:.1f}mm, Y={eye_position.y * 1000:.1f}mm, Z={eye_position.z * 1000:.1f}mm"
    )
    print(f"Using interpolation method: {args.method}\n")

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

    # Setup tracker with selected method
    et = InterpolationTracker.create([cam], [light], calib_points, args.method)

    # Calibrate the eye tracker once
    print("Calibrating eye tracker...")
    et.run_calibration(eye)

    print("1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=eye)
    print_error_summary(calib_results, "Calibration Test Summary")

    print("2. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)

    grid_center = Position3D(0, 0, 200e-3)
    screen_results = accuracy_over_gaze_points(et, eye=eye, grid_center=grid_center)

    print_error_summary(screen_results, "Screen Test Summary")

    print("\n3. Testing over observer (fixed gaze, sweep observer positions):")
    print("-" * 60)
    # Use a fixed gaze target for observer position testing
    gaze_target = Position3D(0, 0, 200e-3)  # Center of the screen
    observer_results = accuracy_over_observer_positions(et, eye=eye, gaze_target=gaze_target)

    print_error_summary(observer_results, "Observer Test Summary")


if __name__ == "__main__":
    main()
