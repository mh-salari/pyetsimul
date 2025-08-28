"""Experiment runner for parameter variations."""

import importlib.util
from pathlib import Path
from typing import Optional

from .strategies import DataGenerationStrategy
from ..visualization import plot_setup_and_camera_view


def load_config(config_path: Path):
    """Load configuration from Python file and return all config objects."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config from {config_path}")

    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)

    # Find all config objects by checking for required attributes
    configs = {}
    for attr_name in dir(config_module):
        if not attr_name.startswith("_"):  # Skip private attributes
            attr_value = getattr(config_module, attr_name)
            # Check if it's a config object by looking for required attributes
            if hasattr(attr_value, "experiment_name") and hasattr(attr_value, "experiment_type"):
                configs[attr_name] = attr_value

    if not configs:
        raise ValueError(
            f"No config objects found in {config_path}. Looking for objects with 'experiment_name' and 'experiment_type' attributes."
        )

    return configs


def run_single_config(config_path: Path, show_setup: bool = False):
    """Generate eye tracking data for a single config file."""
    configs = load_config(config_path)

    results = []
    for config_name, config in configs.items():
        print(f"\n{'=' * 60}")
        print(f"Running: {config.experiment_name}")
        print(f"Config: {config_path.name} -> {config_name}")
        print(f"Type: {config.experiment_type}")
        print(f"{'=' * 60}")

        if show_setup:
            try:
                target_points = config.get_visualization_targets()
                plot_setup_and_camera_view(config.eyes, target_points, config.cameras, config.lights)
                print("Setup visualization displayed.")
            except Exception as e:
                print(f"Setup visualization failed: {e}")

        # Generate single unified dataset: camera → eye → variations
        variation = config.get_variation()
        gaze_target = config.get_gaze_target()

        data_gen_strategy = DataGenerationStrategy(
            cameras=config.cameras,
            lights=config.lights,
            gaze_target=gaze_target,
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

        results.append(result)

    return results if len(results) > 1 else results[0] if results else None


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
            # Handle both single result and list of results
            if isinstance(result, list):
                results.extend(result)
            elif result is not None:
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
