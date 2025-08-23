"""Eye movement experiment data generation.

Generates synthetic eye tracking data for eye movement experiments using
Python configuration files. Tests robustness to observer position changes.
"""

import argparse
import importlib.util
from pathlib import Path
from pyetsimul.data_generation import EyeMovementExperiment
from pyetsimul.visualization import plot_interactive_setup


def load_config(config_path: str):
    """Load configuration from Python file."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def main(config_path: str, show_setup: bool = False):
    """Run eye movement experiment data generation example."""
    if not config_path:
        raise ValueError("Config path must be provided")

    # Load configuration
    config = load_config(config_path)
    print(f"Loaded configuration from: {config_path}")

    # Create experiment
    experiment = EyeMovementExperiment(
        movement_pattern=config.movement_pattern,
        eyes=config.eyes,
        cameras=config.cameras,
        lights=config.lights,
        experiment_name=config.experiment_name,
    )

    # Show setup visualization if requested
    if show_setup:
        # Use the first eye and camera for visualization
        eye_base = config.eyes[0]
        camera = config.cameras[0]
        target_point = config.movement_pattern.gaze_target

        plot_interactive_setup(eye_base, config.lights, camera, target_point)
        print("Setup visualization displayed.")

    # Generate the data
    print("\nRunning experiment...")
    experiment.run_experiment()

    # Save results
    print("\nSaving experiment data...")
    saved_files = experiment.save_experiment_data(output_dir=str(config.output_dir), save_json=True, save_csv=True)

    # Display summary
    summary = experiment.get_experiment_summary()
    print("\nExperiment Summary:")
    print(f"- Experiment: {summary['experiment_name']}")
    print(f"- Total measurements: {summary['total_measurements']}")
    print(f"- Eyes: {summary['eye_count']}, Cameras: {summary['camera_count']}, Lights: {summary['light_count']}")

    design = summary["design_parameters"]
    print(f"- Eye movement pattern: {design['total_positions']} positions")
    print(f"- Varying dimensions: {design['varying_dimensions']['count']}D")

    print("\nData saved to:")
    for format_type, filepath in saved_files.items():
        print(f"- {format_type.upper()}: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run eye movement experiment data generation")
    parser.add_argument("config_path", help="Path to Python configuration file")
    parser.add_argument("--show-setup", action="store_true", help="Display setup visualization")

    args = parser.parse_args()
    main(args.config_path, args.show_setup)
