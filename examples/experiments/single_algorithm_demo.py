"""Polynomial interpolation eye tracking example using experiment configuration.

Demonstrates using the experiment framework configuration system
"""

import matplotlib.pyplot as plt

from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.simulation import DataGenerationStrategy
from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.visualization.gaze_accuracy_plots import GazeAccuracyPlotter
from config import (
    create_calibration_points,
    create_eye_position_config,
    create_gaze_movement_config,
    create_pupil_size_config,
    create_angle_kappa_config,
    create_corneal_radius_config,
    create_individual_differences_config,
)


def main():
    """Run polynomial interpolation eye tracking using experiment configuration."""
    print("Python Interpolate Test - Using Experiment Configuration\n")

    # Use eye position config as base (includes hardware setup and pupil size)
    config = create_eye_position_config()

    # Setup tracker using config
    calib_points = create_calibration_points()
    method = "cerrolaza_2008"
    et = InterpolationTracker.create(config.cameras, config.lights, calib_points, method)
    et.use_legacy_look_at = True

    # Display configuration summary
    et.pprint(config.eyes[0])

    # Calibrate the eye tracker
    print("Calibrating eye tracker...")
    et.run_calibration(config.eyes[0])

    print("1. Testing calibration accuracy:")
    print("-" * 60)
    calib_results = accuracy_at_calibration_points(et, eye=config.eyes[0])
    calib_results.pprint("Calibration Test Summary")

    print("\n2. Testing over screen (using gaze movement config):")
    print("-" * 60)

    # Use gaze movement config for screen test
    screen_config = create_gaze_movement_config()
    screen_data_gen = DataGenerationStrategy(
        cameras=screen_config.cameras,
        lights=screen_config.lights,
        gaze_target=screen_config.gaze_target,
        experiment_name="screen_test",
        save_to_file=False,
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    screen_dataset = screen_data_gen.execute(screen_config.eyes, screen_config.variation)
    screen_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=screen_dataset, description="Evaluating screen test data"
    )
    screen_results.pprint("Screen Test Summary")

    # Plot screen test results
    plotter = GazeAccuracyPlotter()
    plotter.plot(screen_results, et, "Screen Test - Gaze Accuracy")

    plt.show(block=False)
    input("Press Enter to continue to observer test...")
    plt.close("all")

    print("\n3. Testing over observer (eye position movement):")
    print("-" * 60)

    # Use original config for observer test
    observer_data_gen = DataGenerationStrategy(
        cameras=config.cameras,
        lights=config.lights,
        gaze_target=config.gaze_target,
        experiment_name="observer_test",
        save_to_file=False,
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    observer_dataset = observer_data_gen.execute(config.eyes, config.variation)
    observer_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=observer_dataset, description="Evaluating observer test data"
    )
    observer_results.pprint("Observer Test Summary")

    # Plot observer test results
    plotter.plot(observer_results, et, "Observer Test - Eye Movement Analysis")

    plt.show(block=False)
    input("Press Enter to continue to anatomical tests...")
    plt.close("all")

    print("\n4. Testing multiple anatomical variations:")
    print("-" * 60)

    # Test configurations that demonstrate different aspects
    test_configs = {
        "Pupil Size": create_pupil_size_config(),
        "Angle Kappa": create_angle_kappa_config(),
        "Corneal Radius": create_corneal_radius_config(),
        "Individual Differences": create_individual_differences_config(),
    }

    for test_name, test_config in test_configs.items():
        print(f"\n  {test_name} Test:")
        print("  " + "-" * 40)

        test_data_gen = DataGenerationStrategy(
            cameras=test_config.cameras,
            lights=test_config.lights,
            gaze_target=test_config.gaze_target,
            experiment_name=f"{test_name.lower().replace(' ', '_')}_test",
            save_to_file=False,
            use_legacy_look_at=et.use_legacy_look_at,
            use_refraction=et.use_refraction,
        )

        test_dataset = test_data_gen.execute(test_config.eyes, test_config.variation)
        test_results = evaluate_gaze_accuracy(
            eye_tracker=et, dataset=test_dataset, description=f"Evaluating {test_name.lower()} variation data"
        )
        test_results.pprint(f"{test_name} Test Summary")

        # Note: Anatomical variations (pupil size, angle kappa, etc.) cannot be plotted spatially
        # as they don't have dx/dy/dz grid attributes - only spatial variations can be plotted


if __name__ == "__main__":
    main()
