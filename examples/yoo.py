"""Yoo and Chung eye tracking example.

Demonstrates cross-ratio based gaze tracking with configurable setup.
"""

from et_simul.gaze_tracking_algorithms.yoo import YooTracker
from et_simul.evaluation import accuracy_at_calibration_points, accuracy_over_gaze_points
from et_simul.core import Eye, Camera, Light
from et_simul.types import Position3D, RotationMatrix
from et_simul.types.algorithms import YooConfig
from tabulate import tabulate


def main():
    """Test Yoo and Chung eye tracking with configurable setup."""
    print("Python Yoo and Chung Test\n")

    # Eye position
    eye_position = Position3D(0, 550e-3, 350e-3)

    # Create eye
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = eye_position

    # Create camera configuration
    cam = Camera(err=0.0, err_type="gaussian")
    cam.trans[:3, :3] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]  # Camera orientation
    cam.point_at(eye_position)  # Point camera at eye

    # Create 5 lights: 4 around monitor corners, 1 co-located with camera
    light_positions = [
        Position3D(200e-3, 0, 350e-3),  # Top right
        Position3D(-200e-3, 0, 350e-3),  # Top left
        Position3D(-200e-3, 0, 50e-3),  # Bottom left
        Position3D(200e-3, 0, 50e-3),  # Bottom right
        # Position3D(0, 0, 0),             # Co-located with camera
    ]

    lights = [Light(position=pos) for pos in light_positions]

    # Calibration points at the four monitor corners
    calib_points = [
        Position3D(200e-3, 0, 350e-3),  # Top right
        Position3D(-200e-3, 0, 350e-3),  # Top left
        Position3D(-200e-3, 0, 50e-3),  # Bottom left
        Position3D(200e-3, 0, 50e-3),  # Bottom right
    ]

    # Create algorithm configuration
    config = YooConfig(screen_width=400e-3, screen_height=300e-3, screen_center_x=0.0, screen_center_y=200e-3)

    # Setup tracker with external components
    et = YooTracker.create([cam], lights, calib_points, config)

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
        ["Lights", "Count", f"{len(lights)}", "lights"],
        ["Screen", "Size (W x H)", f"{config.screen_width * 1000:.0f} x {config.screen_height * 1000:.0f}", "mm"],
        ["Algorithm", "Method", "yoo_chung", "-"],
        ["Calibration", "Points", f"{len(calib_points)}", "points"],
    ]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    print()

    # Calibrate
    print("Calibrating...")
    et.run_calibration(eye)
    print(f"Alpha values: {et.algorithm_state.alpha_values}")
    print()

    # Test calibration accuracy
    print("1. Testing calibration fit:")
    calib_results = accuracy_at_calibration_points(et, eye=eye)

    print(f"Mean error: {calib_results['mtr']['mean'] * 1000:.1f} mm")
    print(f"Max error: {calib_results['mtr']['max'] * 1000:.1f} mm")
    print()

    print("2. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)

    grid_center = Position3D(0, 0, 200e-3)
    screen_results = accuracy_over_gaze_points(et, eye=eye, grid_center=grid_center)

    print(f"Mean error: {screen_results['mtr']['mean'] * 1000:.1f} mm")
    print(f"Max error: {screen_results['mtr']['max'] * 1000:.1f} mm")


if __name__ == "__main__":
    main()
