from et_simul.gaze_tracking_algorithms.hennessey import HennesseyTracker
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
    parser = argparse.ArgumentParser(description="Hennessey Eye Tracker Test")
    parser.add_argument(
        "--eye-position", 
        type=float, 
        nargs=3, 
        default=[0, 550e-3, 350e-3],
        metavar=("X", "Y", "Z"),
        help="Eye position in meters (default: 0 550e-3 350e-3)"
    )
    
    args = parser.parse_args()
    
    print("=== Python Hennessey Test (System Integration) ===\n")
    
    eye_position = np.array(args.eye_position)
    print(f"Using eye position: X={eye_position[0]*1000:.1f}mm, Y={eye_position[1]*1000:.1f}mm, Z={eye_position[2]*1000:.1f}mm\n")

    # Create eye configuration
    eye = Eye()
    eye.rest_orientation = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    eye.position = eye_position

    # Create camera configuration
    cam = Camera(err=0.0, err_type="uniform")
    cam.orientation = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    cam.rest_trans = cam.trans.copy()
    # Point camera at the eye position
    cam.point_at(eye.position.tolist() + [1])

    # Create light configuration (dual lights on vertical edge of monitor)
    light1 = Light(position=np.array([200e-3, 0, 50e-3, 1]))
    light2 = Light(position=np.array([200e-3, 0, 300e-3, 1]))

    # Nine-point calibration pattern
    calib_points = [
        [0, 200e-3],  # Point 1
        [-150e-3, 50e-3],  # Point 2
        [-150e-3, 350e-3],  # Point 3
        [150e-3, 50e-3],  # Point 4
        [150e-3, 350e-3],  # Point 5
        [-100e-3, 150e-3],  # Point 6
        [-100e-3, 250e-3],  # Point 7
        [100e-3, 150e-3],  # Point 8
        [100e-3, 250e-3],  # Point 9
    ]

    # Setup tracker with external components
    et = HennesseyTracker.setup([cam], [light1, light2], np.array(calib_points).T)

    print("1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=eye)
    print_error_summary(calib_results, "Calibration Test Summary")

    print("2. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)
    screen_results = accuracy_over_gaze_points(et, eye=eye)
    print_error_summary(screen_results, "Screen Test Summary")

    print(f"\n3. Testing over observer (fixed gaze, sweep observer positions):")
    print("-" * 60)
    observer_results = accuracy_over_observer_positions(et, eye=eye)
    print_error_summary(observer_results, "Observer Test Summary")


if __name__ == "__main__":
    main()
