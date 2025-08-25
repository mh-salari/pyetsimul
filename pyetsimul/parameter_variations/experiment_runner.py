"""Experiment runner for parameter variations."""

import importlib.util
from pathlib import Path
from typing import Optional

from .strategies import DataGenerationStrategy
from ..visualization import plot_setup_and_camera_view


def load_config(config_path: Path):
    """Load configuration from Python file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)

    return config


def run_single_config(config_path: Path, show_setup: bool = False):
    """Generate eye tracking data for a single config file."""
    config_module = load_config(config_path)
    config = config_module.config  # Get the actual config object
    print(f"\n{'=' * 60}")
    print(f"Running: {config.experiment_name}")
    print(f"Config: {config_path.name}")
    print(f"Type: {config.experiment_type}")
    print(f"{'=' * 60}")

    if show_setup:
        try:
            if config.experiment_type == "eye_position_variation":
                target_points = [config.gaze_target] * len(config.eyes)
            else:
                target_points = [config.target_variation.grid_center] * len(config.eyes)

            plot_setup_and_camera_view(config.eyes, target_points, config.cameras, config.lights)
            print("Setup visualization displayed.")
        except Exception as e:
            print(f"Setup visualization failed: {e}")

    # Generate single unified dataset: camera → eye → variations
    if config.experiment_type == "eye_position_variation":
        variation = config.eye_variation
        gaze_target = config.gaze_target
    elif config.experiment_type == "target_position_variation":
        variation = config.target_variation
        gaze_target = None
    else:
        raise ValueError(f"Unknown experiment_type: {config.experiment_type}")

    data_gen_strategy = DataGenerationStrategy(
        cameras=config.cameras,
        lights=config.lights,
        gaze_target=gaze_target,
        output_format="both",
        output_dir=str(config.output_dir),
        experiment_name=config.experiment_name,
    )

    print(f"Generating {config.experiment_type} data...")
    result = data_gen_strategy.execute(config.eyes, variation)

    print(f"\nResults for {config.experiment_name}:")
    print(f"- Parameter varied: {result['parameter_name']}")
    print(f"- Total measurements: {result['total_measurements']}")
    print(f"- Eyes: {len(config.eyes)}, Cameras: {len(config.cameras)}, Lights: {len(config.lights)}")

    print("Files saved:")
    for filepath in result["saved_files"]:
        print(f"- {filepath}")

    return result


def run_all_configs(configs_dir: Path, show_setup: bool = False):
    """Run all config files in the configs directory."""
    config_files = list(configs_dir.glob("*.py"))
    if not config_files:
        raise ValueError(f"No config files found in {configs_dir}")

    print(f"Found {len(config_files)} config files:")
    for config_file in config_files:
        print(f"- {config_file.name}")

    results = []
    for config_file in config_files:
        try:
            result = run_single_config(config_file, show_setup)
            results.append(result)
        except Exception as e:
            print(f"ERROR running {config_file.name}: {e}")
            continue

    print(f"\n{'=' * 60}")
    print("All configs completed!")
    print(f"{'=' * 60}")

    return results


def run_experiments(config_path: Optional[str] = None, configs_dir: Optional[str] = None, show_setup: bool = False):
    """Run parameter variation experiments from config file(s)."""
    if config_path:
        return run_single_config(Path(config_path), show_setup)
    elif configs_dir:
        return run_all_configs(Path(configs_dir), show_setup)
    else:
        raise ValueError("Must provide either config_path or configs_dir")
