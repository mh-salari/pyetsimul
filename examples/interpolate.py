from et_simul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from et_simul.performance_analysis import (
    accuracy_over_gaze_points,
    accuracy_over_observer_positions,
    accuracy_at_calibration_points,
)
from et_simul.performance_analysis.analysis_utils import print_error_summary
from et_simul.core import Light, Camera, Eye
import numpy as np
import argparse


def main():
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

    eye_position = np.array(args.eye_position)
    print(
        f"Using eye position: X={eye_position[0] * 1000:.1f}mm, Y={eye_position[1] * 1000:.1f}mm, Z={eye_position[2] * 1000:.1f}mm"
    )
    print(f"Using interpolation method: {args.method}\n")

    # Create eye configuration
    eye = Eye()
    eye.set_rest_orientation(np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    eye.position = eye_position

    # Create camera configuration
    cam = Camera(err=0.0, err_type="gaussian")
    cam.orientation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    cam.rest_trans = cam.trans.copy()
    # Point camera at the eye position
    cam.point_at(eye.position.tolist() + [1])

    # Create light configuration
    light = Light(position=np.array([200e-3, 0, 350e-3]))

    # Create calibration grid
    calib_points = [
        [-200e-3, 50e-3],
        [0, 50e-3],
        [200e-3, 50e-3],
        [-200e-3, 200e-3],
        [0, 200e-3],
        [200e-3, 200e-3],
        [-200e-3, 350e-3],
        [0, 350e-3],
        [200e-3, 350e-3],
    ]

    # Setup tracker with selected method
    et = InterpolationTracker.setup([cam], [light], calib_points, args.method)

    print("1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=eye)
    print_error_summary(calib_results, "Calibration Test Summary")

    print("2. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)
    screen_results = accuracy_over_gaze_points(et, eye=eye)

    print_error_summary(screen_results, "Screen Test Summary")

    print("\n3. Testing over observer (fixed gaze, sweep observer positions):")
    print("-" * 60)
    observer_results = accuracy_over_observer_positions(et, eye=eye)

    print_error_summary(observer_results, "Observer Test Summary")


if __name__ == "__main__":
    main()
