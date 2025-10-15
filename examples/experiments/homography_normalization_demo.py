"""Homography gaze model eye tracking example using shared configuration."""

import matplotlib.pyplot as plt
from config import (
    angle_kappa_variation,
    calibration_points,
    corneal_radius_variation,
    create_homography_experiment_config,  # Use homography config
    eye_position_variation,
    pupil_size_variation,
    target_position_variation,
)

from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.gaze_models import HomographyNormalizationGazeModel
from pyetsimul.simulation import DataGenerationStrategy
from pyetsimul.visualization.gaze_accuracy_plots import GazeAccuracyPlotter


def main() -> None:
    """Run homography gaze model eye tracking using shared configuration."""
    base_config = create_homography_experiment_config("base")

    print("Python Homography Normalization Gaze Model Test\n")

    # Setup tracker
    et = HomographyNormalizationGazeModel.create(
        base_config.cameras,
        base_config.lights,
        calibration_points,
        use_gp_correction=True,  # Enable GP correction
        ransac_threshold=5.0,
    )
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

    # Single reusable strategy for all tests
    data_gen = DataGenerationStrategy(
        eyes=base_config.eyes,
        cameras=base_config.cameras,
        lights=base_config.lights,
        gaze_target=base_config.gaze_target,
        experiment_name="base",
        save_to_file=False,
        use_legacy_look_at=et.use_legacy_look_at,
        use_refraction=et.use_refraction,
    )

    print("\n2. Testing over screen (target position variation):")
    print("-" * 60)
    screen_dataset = data_gen.execute(target_position_variation)
    screen_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=screen_dataset, description="Evaluating screen test data"
    )
    screen_results.pprint("Screen Test Summary")
    plotter = GazeAccuracyPlotter()
    plotter.plot(screen_results, et, "Screen Test - Gaze Accuracy")

    print("\n3. Testing over observer (eye position movement):")
    print("-" * 60)
    observer_dataset = data_gen.execute(eye_position_variation)
    observer_results = evaluate_gaze_accuracy(
        eye_tracker=et, dataset=observer_dataset, description="Evaluating observer test data"
    )
    observer_results.pprint("Observer Test Summary")
    plotter.plot(observer_results, et, "Observer Test - Eye Movement Analysis")

    print("\n4. Testing multiple anatomical variations:")
    print("-" * 60)
    variations = {
        "Pupil Size": pupil_size_variation,
        "Angle Kappa": angle_kappa_variation,
        "Corneal Radius": corneal_radius_variation,
    }

    for test_name, variation in variations.items():
        print(f"\n    {test_name} Test:")
        print("    " + "-" * 30)
        data_gen.set_experiment_name(f"{test_name.lower().replace(' ', '_')}_test")
        test_dataset = data_gen.execute(variation)
        test_results = evaluate_gaze_accuracy(
            eye_tracker=et, dataset=test_dataset, description=f"Evaluating {test_name.lower()} variation data"
        )
        test_results.pprint(f"{test_name} Test Summary")

    print("\nAll tests complete. Displaying all plots...")
    plt.show()


if __name__ == "__main__":
    main()
