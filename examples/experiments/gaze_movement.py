"""Example: Generate eye tracking data using GazeMovement experimental design.

This example demonstrates how to use the GazeMovementExperiment to generate
synthetic eye tracking data using Python configuration files.
"""

import argparse
import importlib.util
from pathlib import Path
from pyetsimul.data_generation import GazeMovementExperiment
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
    """Run gaze movement experiment data generation example."""
    if not config_path:
        raise ValueError("Config path must be provided")

    # Load experiment configuration
    config = load_config(config_path)

    # Create and run experiment
    print("Setting up gaze movement experiment...")
    experiment = GazeMovementExperiment(
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
        target_point = config.movement_pattern.grid_center

        plot_interactive_setup(eye_base, config.lights, camera, target_point)
        print("Setup visualization displayed.")

    # Generate the data
    print("\nRunning experiment...")
    experiment.run_experiment()

    # Save results
    print("\nSaving experiment data...")
    saved_files = experiment.save_experiment_data(output_dir=str(config.output_dir), save_json=True, save_csv=True)

    # Print summary
    summary = experiment.get_experiment_summary()
    print("\nExperiment Summary:")
    print(f"- Name: {summary['experiment_name']}")
    print(f"- Total measurements: {summary['total_measurements']}")
    print(f"- Target grid size: {summary['design_parameters']['grid_size']}")
    print(f"- Eye count: {summary['eye_count']}")
    print(f"- Camera count: {summary['camera_count']}")
    print(f"- Light count: {summary['light_count']}")

    print("\nFiles saved:")
    for format_type, filepath in saved_files.items():
        print(f"- {format_type.upper()}: {filepath}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run gaze movement experiment data generation")
    parser.add_argument("config_path", help="Path to Python configuration file")
    parser.add_argument("--show-setup", action="store_true", help="Display setup visualization")

    args = parser.parse_args()
    main(args.config_path, args.show_setup)
