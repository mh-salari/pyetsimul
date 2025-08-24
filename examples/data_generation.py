"""Simple example using parameter variation experiment runner."""

import argparse
from pathlib import Path
from pyetsimul.parameter_variations import run_experiments


def main():
    """Run parameter variation experiments."""
    parser = argparse.ArgumentParser(description="Generate eye tracking data using parameter variations")
    parser.add_argument("--config", help="Path to single Python configuration file")
    parser.add_argument("--configs-dir", help="Directory containing config files to run")
    parser.add_argument("--show-setup", action="store_true", help="Display setup visualization")

    args = parser.parse_args()

    if not args.config and not args.configs_dir:
        # Default behavior: run all configs in experiments/configs with visualization
        default_configs_dir = Path(__file__).parent / "experiments" / "configs"
        print(f"No arguments provided, running all configs in: {default_configs_dir}")
        run_experiments(configs_dir=str(default_configs_dir), show_setup=True)
    else:
        run_experiments(args.config, args.configs_dir, args.show_setup)


if __name__ == "__main__":
    main()
