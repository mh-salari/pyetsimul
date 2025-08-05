from et_simul.gaze_tracking_algorithms.hennessey import HennesseyTracker
from et_simul.types import HennesseyConfig, Position3D, RotationMatrix
from et_simul.evaluation import (
    accuracy_over_gaze_points,
    accuracy_over_observer_positions,
    accuracy_at_calibration_points,
)
from et_simul.evaluation.analysis_utils import print_error_summary
from et_simul.core import Light, Camera, Eye
from tabulate import tabulate


def main():
    print("Python Hennessey Test (System Integration)\n")

    # Eye position
    eye_position = Position3D(0, 550e-3, 350e-3)

    # Create eye configuration
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = eye_position

    # Create camera configuration
    cam = Camera(err=0.0, err_type="uniform")
    cam.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    # Point camera at the eye position
    cam.point_at(eye.position)

    # Create light configuration (dual lights on vertical edge of monitor)
    light1 = Light(position=Position3D(200e-3, 0, 50e-3))

    light2 = Light(position=Position3D(200e-3, 0, 300e-3))

    # Nine-point calibration pattern - convert to Position3D
    calib_points = [
        Position3D(0, 0.0, 200e-3),  # Point 1
        Position3D(-150e-3, 0.0, 50e-3),  # Point 2
        Position3D(-150e-3, 0.0, 350e-3),  # Point 3
        Position3D(150e-3, 0.0, 50e-3),  # Point 4
        Position3D(150e-3, 0.0, 350e-3),  # Point 5
        Position3D(-100e-3, 0.0, 150e-3),  # Point 6
        Position3D(-100e-3, 0.0, 250e-3),  # Point 7
        Position3D(100e-3, 0.0, 150e-3),  # Point 8
        Position3D(100e-3, 0.0, 250e-3),  # Point 9
    ]

    # Create configuration for Hennessey algorithm
    config = HennesseyConfig()

    # Setup tracker with external components using new factory method
    et = HennesseyTracker.create([cam], [light1, light2], calib_points, config)

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
            "Light 1",
            "Position (x, y, z)",
            f"({light1.position.x * 1000:.1f}, {light1.position.y * 1000:.1f}, {light1.position.z * 1000:.1f})",
            "mm",
        ],
        [
            "Light 2",
            "Position (x, y, z)",
            f"({light2.position.x * 1000:.1f}, {light2.position.y * 1000:.1f}, {light2.position.z * 1000:.1f})",
            "mm",
        ],
        ["Algorithm", "Method", "Hennessey geometric", "-"],
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
