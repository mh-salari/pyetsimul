"""Algorithm comparison experiment using configuration system.

Compares multiple eye tracking algorithms on identical datasets using the experiment
framework configurations. Tests algorithm performance across different variations.
Supports caching/loading of experiment data to avoid regeneration.
"""

import json
from pyetsimul.gaze_tracking_algorithms.interpolate import InterpolationTracker
from pyetsimul.gaze_tracking_algorithms.interpolate.polynomials import list_available_polynomials
from pyetsimul.evaluation.algorithm_comparison import compare_algorithms
from pyetsimul.evaluation.calibration_analysis import accuracy_at_calibration_points
from pyetsimul.experiment_framework.data_generation import DataGenerationStrategy
from pyetsimul.experiment_framework.data_generation import ComposedVariation, ExperimentConfig
from config import (
    create_eye_position_config,
    create_calibration_points,
    create_gaze_movement_config,
    create_pupil_size_config,
)


def setup_algorithms(config):
    """Setup and calibrate all available algorithms using experiment config."""
    calib_points = create_calibration_points()
    available_methods = list_available_polynomials()

    algorithms = {}
    print(f"\nCalibrating {len(available_methods)} algorithms:")

    for method in available_methods:
        # Create algorithm using config hardware
        algorithm = InterpolationTracker.create(
            cameras=config.cameras, lights=config.lights, calib_points=calib_points, polynomial=method
        )
        algorithm.use_legacy_look_at = True

        # Calibrate using first eye from config
        algorithm.run_calibration(config.eyes[0])

        # Use polynomial name as display name
        display_name = algorithm.polynomial_name.replace("_", " ").title()
        print(f"  - {display_name}...")

        # Verify calibration
        calib_results = accuracy_at_calibration_points(algorithm, eye=config.eyes[0], interactive_plot=False)
        mean_error = calib_results.errors["deg"]["mean"]
        print(f"    Calibration accuracy: {mean_error:.3f}°")

        algorithms[display_name] = algorithm

    return algorithms


def load_cached_dataset(config, test_name):
    """Load cached dataset if it exists."""
    # Ensure output directory exists
    config.output_dir.mkdir(parents=True, exist_ok=True)

    cache_filename = f"{config.experiment_name}_data.json"
    cache_path = config.output_dir / cache_filename

    print(f"Looking for cached dataset: {cache_path}")

    if cache_path.exists():
        print(f"Found cached dataset: {cache_path}")
        try:
            with open(cache_path, "r") as f:
                cached_data = json.load(f)

            # Create dataset structure expected by compare_algorithms
            # The JSON contains the raw experiment data, so we need to wrap it
            total_measurements = cached_data.get("experiment_metadata", {}).get("total_parameter_values", 0)
            dataset = {
                "total_measurements": total_measurements,
                "data": cached_data,
                "parameter_name": cached_data.get("experiment_metadata", {}).get("parameter_variation", ""),
                "saved_files": [str(cache_path)],
            }
            print(f"Loaded cached dataset with {total_measurements} measurements")
            return dataset
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to load cached data ({e}), will regenerate")
            return None
    else:
        print(f"No cached dataset found at: {cache_path}")
        return None


def generate_and_cache_dataset(config, test_name, use_cache=True):
    """Generate dataset and optionally cache it."""
    # Try to load cached data first if use_cache is True
    if use_cache:
        cached_dataset = load_cached_dataset(config, test_name)
        if cached_dataset is not None:
            return cached_dataset

    # Generate new dataset
    print("Generating new dataset...")
    data_gen = DataGenerationStrategy(
        cameras=config.cameras,
        lights=config.lights,
        gaze_target=config.gaze_target,
        output_dir=config.output_dir,  # Use config's output directory
        experiment_name=config.experiment_name,  # Use config's experiment name
        save_to_file=use_cache,  # Save to file if caching is enabled
        use_legacy_look_at=True,
    )

    dataset = data_gen.execute(config.eyes, config.variation)
    print(f"Generated {dataset['total_measurements']} measurements")

    if use_cache:
        print("Dataset cached for future use")

    return dataset


def run_experiment(config, algorithms, test_name, use_cache=True):
    """Run algorithm comparison on a specific experiment configuration."""
    print("\n" + "=" * 60)
    print(f"TEST: {test_name.upper()}")
    print("=" * 60)

    # Load or generate dataset
    dataset = generate_and_cache_dataset(config, test_name, use_cache)

    # Compare algorithms
    comparison = compare_algorithms(algorithms, dataset, test_name)
    comparison.pprint(f"{test_name} - Algorithm Ranking")

    return comparison


def create_composed_config():
    """Create composed variation using existing config variations."""
    base_config = create_eye_position_config()

    # Get variations from existing configs
    eye_pos_var = create_eye_position_config().variation
    gaze_mov_var = create_gaze_movement_config().variation
    pupil_size_var = create_pupil_size_config().variation

    # Compose them together (125 * 25 * 10 = 31,250 measurements!)
    composed = ComposedVariation(
        variations=[eye_pos_var, gaze_mov_var, pupil_size_var], param_name="eye_pos_gaze_pupil_composed"
    )

    return ExperimentConfig(
        experiment_name="composed_variation",
        variation=composed,
        eyes=base_config.eyes,
        cameras=base_config.cameras,
        lights=base_config.lights,
        gaze_target=base_config.gaze_target,
        output_dir=base_config.output_dir,
    )


def main():
    """Compare algorithms using experiment configurations."""
    use_cache = True

    print("Algorithm Comparison - Using Experiment Framework Configurations")
    print("=" * 80)
    print(f"Dataset Caching: {use_cache}")
    print("=" * 80)

    # Load experiment configurations
    configs = {
        "Eye Position Variation": create_eye_position_config(),
        "Gaze Movement Variation": create_gaze_movement_config(),
        "Pupil Size Variation": create_pupil_size_config(),
        "Composed Variation": create_composed_config(),
    }

    print("Loaded experiment configurations:")
    for name, config in configs.items():
        var_info = f"{config.variation.__class__.__name__}"
        if hasattr(config.variation, "grid_size"):
            var_info += f" ({config.variation.grid_size})"
        elif hasattr(config.variation, "num_steps"):
            var_info += f" ({config.variation.num_steps} steps)"
        print(f"  - {name}: {var_info}")

    # Setup algorithms using first config (they all use same hardware)
    base_config = next(iter(configs.values()))
    algorithms = setup_algorithms(base_config)

    # Run comparisons on each configuration
    results = {}
    for test_name, config in configs.items():
        results[test_name] = run_experiment(config, algorithms, test_name, use_cache)


if __name__ == "__main__":
    main()
