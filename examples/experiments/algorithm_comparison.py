"""Algorithm comparison experiment using shared configuration."""

from typing import Any

from config import (
    angle_kappa_variation,
    calibration_points,
    corneal_radius_variation,
    corneal_thickness_variation,
    create_experiment_config,
    eye_position_variation,
    pupil_size_variation,
    target_position_variation,
)

from pyetsimul.evaluation.algorithm_comparison import AlgorithmRanking, compare_algorithms
from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points
from pyetsimul.gaze_models.polynomial import PolynomialGazeModel
from pyetsimul.gaze_models.polynomial.polynomials import list_available_polynomials
from pyetsimul.simulation import DataGenerationStrategy
from pyetsimul.simulation.config import ExperimentConfig
from pyetsimul.simulation.core import ParameterVariation
from pyetsimul.simulation.data_loading import load_experiment_data


def setup_algorithms() -> dict[str, PolynomialGazeModel]:
    """Setup and calibrate all available algorithms."""
    base_config = create_experiment_config("base")
    available_methods = list_available_polynomials()
    algorithms = {}

    print(f"\nCalibrating {len(available_methods)} algorithms:")

    for method in available_methods:
        algorithm = PolynomialGazeModel.create(
            cameras=base_config.cameras, lights=base_config.lights, calib_points=calibration_points, polynomial=method
        )
        algorithm.use_legacy_look_at = True
        algorithm.run_calibration(base_config.eyes[0])

        display_name = algorithm.polynomial_name.replace("_", " ").title()
        print(f"  - {display_name}...")

        calib_results = accuracy_at_calibration_points(algorithm, eye=base_config.eyes[0], interactive_plot=False)
        mean_error = calib_results.errors["deg"]["mean"]
        print(f"    Calibration accuracy: {mean_error:.3f}°")

        algorithms[display_name] = algorithm

    return algorithms


def generate_dataset(
    config: ExperimentConfig, variation: ParameterVariation, use_cache: bool = True
) -> dict[str, Any]:
    """Generate dataset and optionally cache it."""
    if use_cache:
        try:
            cached_dataset = load_experiment_data(config.experiment_name, config.output_dir)
            total_measurements = cached_dataset["total_measurements"]
            print(f"Loaded cached dataset with {total_measurements} measurements")
            return cached_dataset
        except FileNotFoundError:
            print(f"No cached dataset found for experiment '{config.experiment_name}'")

    print("Generating new dataset...")

    data_gen = DataGenerationStrategy(
        eyes=config.eyes,
        cameras=config.cameras,
        lights=config.lights,
        gaze_target=config.gaze_target,
        output_dir=config.output_dir,
        experiment_name=config.experiment_name,
        save_to_file=use_cache,
        use_legacy_look_at=True,
    )

    dataset = data_gen.execute(variation)
    print(f"Generated {dataset['total_measurements']} measurements")

    if use_cache:
        print("Dataset cached for future use")

    return dataset


def run_experiment(
    config: ExperimentConfig, variation: ParameterVariation, algorithms: dict[str, Any], use_cache: bool = True
) -> AlgorithmRanking:
    """Run algorithm comparison on a specific variation."""
    print("\n" + "=" * 60)
    print(f"TEST: {config.experiment_name.upper()}")
    print("=" * 60)

    dataset = generate_dataset(config, variation, use_cache)
    comparison = compare_algorithms(algorithms, dataset, config.experiment_name)
    description = variation.describe()
    comparison.pprint(f"{description}")

    return comparison


def main() -> None:
    """Compare algorithms using experiment configurations."""
    use_cache = True

    print("Algorithm Comparison - Using Experiment Framework Configurations")
    print("=" * 80)
    print(f"Dataset Caching: {use_cache}")
    print("=" * 80)

    # Define experiments to run
    experiments = {
        "Eye Position Variation": eye_position_variation,
        "Target Position Variation": target_position_variation,
        "Pupil Size Variation": pupil_size_variation,
        "Angle Kappa Variation": angle_kappa_variation,
        "Corneal Radius Variation": corneal_radius_variation,
        "Corneal Thickness Variation": corneal_thickness_variation,
    }

    print("Loaded experiment configurations:")
    for name, variation in experiments.items():
        config = create_experiment_config(name)
        description = variation.describe()
        print(f"  - {name}: {description}")

    algorithms = setup_algorithms()

    results = {}
    for experiment_name, variation in experiments.items():
        config = create_experiment_config(experiment_name)
        results[experiment_name] = run_experiment(config, variation, algorithms, use_cache)


if __name__ == "__main__":
    main()
