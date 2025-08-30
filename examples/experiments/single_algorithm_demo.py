"""Polynomial interpolation eye tracking example using shared configuration."""

import matplotlib.pyplot as plt

from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.simulation import DataGenerationStrategy
from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.visualization.gaze_accuracy_plots import GazeAccuracyPlotter
from config import (
    create_experiment_config,
    calibration_points,
    eye_position_variation,
    target_position_variation,
    pupil_size_variation,
    angle_kappa_variation,
    corneal_radius_variation,
)


def main():
    """Run polynomial interpolation eye tracking using shared configuration."""
    base_config = create_experiment_config("base")

    print("Python Interpolate Test - Using Shared Configuration\n")

    # Setup tracker using shared config
    method = "cerrolaza_2008"
    et = InterpolationTracker.create(base_config.cameras, base_config.lights, calibration_points, method)
    et.use_legacy_look_at = True

    # Display configuration summary
    et.pprint(base_config.eyes[0])

    # Calibrate the eye tracker
    print("Calibrating eye tracker...")
    et.run_calibration(base_config.eyes[0])

    print("1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=base_config.eyes[0])
    calib_results.pprint("Calibration Test Summary")

    print("\n2. Testing over screen (target position variation):")
    print("-" * 60)

    screen_data_gen = DataGenerationStrategy(
        eyes=base_config.eyes,
        cameras=base_config.cameras,
        lights=base_config.lights,
        gaze_target=base_config.gaze_target,
        experiment_name="screen_test",
        save_to_file=False,
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    screen_dataset = screen_data_gen.execute(target_position_variation)
    screen_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=screen_dataset, description="Evaluating screen test data"
    )
    screen_results.pprint("Screen Test Summary")

    # Plot screen test results
    plotter = GazeAccuracyPlotter()
    plotter.plot(screen_results, et, "Screen Test - Gaze Accuracy")

    plt.show(block=False)
    plt.close("all")

    print("\n3. Testing over observer (eye position movement):")
    print("-" * 60)

    observer_data_gen = DataGenerationStrategy(
        eyes=base_config.eyes,
        cameras=base_config.cameras,
        lights=base_config.lights,
        gaze_target=base_config.gaze_target,
        experiment_name="observer_test",
        save_to_file=False,
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    observer_dataset = observer_data_gen.execute(eye_position_variation)
    observer_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=observer_dataset, description="Evaluating observer test data"
    )
    observer_results.pprint("Observer Test Summary")

    # Plot observer test results
    plotter.plot(observer_results, et, "Observer Test - Eye Movement Analysis")

    plt.show(block=False)
    plt.close("all")

    print("\n4. Testing multiple anatomical variations:")
    print("-" * 60)

    # Single strategy with shared hardware setup
    shared_strategy = DataGenerationStrategy(
        eyes=base_config.eyes,
        cameras=base_config.cameras,
        lights=base_config.lights,
        gaze_target=base_config.gaze_target,
        experiment_name="reuse_demo",
        save_to_file=False,
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    # Test multiple variations with the same strategy
    variations = {
        "Pupil Size": pupil_size_variation,
        "Angle Kappa": angle_kappa_variation,
        "Corneal Radius": corneal_radius_variation,
    }

    for test_name, variation in variations.items():
        print(f"\n    {test_name} Test:")
        print("    " + "-" * 30)

        test_dataset = shared_strategy.execute(variation)
        test_results = evaluate_gaze_accuracy(
            eye_tracker=et, dataset=test_dataset, description=f"Evaluating {test_name.lower()} variation data"
        )
        test_results.pprint(f"{test_name} Test Summary")


if __name__ == "__main__":
    main()
