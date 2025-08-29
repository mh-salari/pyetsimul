"""Refactored polynomial interpolation eye tracking example with DRY principles.

Demonstrates complete separation of data generation and evaluation:
1. Data Generation: Uses existing DataGenerationStrategy to generate measurement data in-memory
2. Evaluation: Uses new evaluate_gaze_accuracy to feed pre-generated data to eye tracker
"""

from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.experiment_framework.data_generation import DataGenerationStrategy
from pyetsimul.experiment_framework.data_generation.spatial import TargetPositionVariation, EyePositionVariation
from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.visualization.gaze_accuracy_plots import GazeAccuracyPlotter
from pyetsimul.core import Light, Camera, Eye
from pyetsimul.types import Position3D, RotationMatrix


def main():
    """Run polynomial interpolation eye tracking with separated data generation and evaluation."""
    print("Python Interpolate Test - Refactored (Data Generation + Evaluation Separation)\n")

    # Eye position
    eye_position = Position3D(0, 550e-3, 350e-3)

    # Create eye configuration with structured types
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
    eye.position = eye_position

    # Create camera configuration with structured types
    cam = Camera(err=0.0, err_type="gaussian")
    cam.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
    cam.point_at(eye.position)

    # Create light configuration
    light = Light(position=Position3D(200e-3, 0, 350e-3))

    # Create calibration grid
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

    # Setup tracker
    method = "cerrolaza_2008"
    et = InterpolationTracker.create([cam], [light], calib_points, method)
    et.use_legacy_look_at = True

    # Display configuration summary
    et.pprint(eye)

    # Calibrate the eye tracker once
    print("Calibrating eye tracker...")
    et.run_calibration(eye)

    print("1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=eye)
    calib_results.pprint("Calibration Test Summary")

    print("\n2. Testing over screen (fixed observer, sweep gaze positions):")
    print("-" * 60)

    # Step 1: Generate screen test data using existing DataGenerationStrategy
    screen_target_variation = TargetPositionVariation(
        grid_center=Position3D(0, 0, 200e-3),
        dx=[-200e-3, 200e-3],  # X varies: ±200mm
        dy=[0.0, 0.0],  # Y fixed: no variation
        dz=[-150e-3, 150e-3],  # Z varies: ±150mm
        grid_size=[16, 1, 16],  # 16x1x16 = 2D grid in XZ plane
    )

    screen_data_gen = DataGenerationStrategy(
        cameras=[cam],
        lights=[light],
        gaze_target=None,  # Targets come from variation
        experiment_name="screen_test",
        save_to_file=False,  # Keep in memory only
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    # Generate data in-memory
    screen_dataset = screen_data_gen.execute([eye], screen_target_variation)

    # Step 2: Evaluate using generic evaluation function
    screen_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=screen_dataset, description="Evaluating screen test data"
    )

    screen_results.pprint("Screen Test Summary")

    # Plot screen test results using dedicated plotter
    plotter = GazeAccuracyPlotter()
    plotter.plot(screen_results, et, "Screen Test - Gaze Accuracy")

    print("\n3. Testing over observer (fixed gaze, sweep observer positions):")
    print("-" * 60)

    # Step 1: Generate observer test data
    observer_eye_variation = EyePositionVariation(
        center=eye_position,  # Central eye position
        dx=[-50e-3, 50e-3],  # X varies: ±50mm
        dy=[-50e-3, 50e-3],  # Y varies: ±50mm
        dz=[0.0, 0.0],  # Z fixed: no variation
        grid_size=[16, 16, 1],  # 16x16x1 = 2D grid in XY plane
    )

    observer_data_gen = DataGenerationStrategy(
        cameras=[cam],
        lights=[light],
        gaze_target=Position3D(0, 0, 200e-3),  # Fixed gaze target
        experiment_name="observer_test",
        save_to_file=False,  # Keep in memory only
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    # Generate data in-memory
    observer_dataset = observer_data_gen.execute([eye], observer_eye_variation)

    # Step 2: Evaluate using generic evaluation function
    observer_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=observer_dataset, description="Evaluating observer test data"
    )

    observer_results.pprint("Observer Test Summary")

    # Plot observer test results using dedicated plotter
    plotter.plot(observer_results, et, "Observer Test - Eye Movement Analysis")


if __name__ == "__main__":
    main()
