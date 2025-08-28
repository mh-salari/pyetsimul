"""Demonstration of composing multiple parameter variations."""

from pathlib import Path
from pyetsimul.experiment_framework.data_generation import (
    ComposedVariation,
    SequentialVariation,
    DataGenerationStrategy,
)
from config import create_eye_position_config, create_pupil_size_config, create_gaze_movement_config


def main():
    """Demonstrate parameter variation composition."""
    Path("outputs").mkdir(exist_ok=True)

    # Create individual variations
    eye_pos_var = create_eye_position_config().variation
    pupil_size_var = create_pupil_size_config().variation
    gaze_mov_var = create_gaze_movement_config().variation

    # We need a base config for camera, lights etc.
    base_config = create_eye_position_config()

    # Demonstrate ComposedVariation (cartesian product)
    print("Generating data for ComposedVariation ...")
    composed = ComposedVariation(
        variations=[eye_pos_var, pupil_size_var, gaze_mov_var], param_name="eye_pos_pupil_gaze_composed"
    )

    composed_strategy = DataGenerationStrategy(
        cameras=base_config.cameras,
        lights=base_config.lights,
        gaze_target=base_config.gaze_target,  # This will be used if the composed variation doesn't have a target variation
        output_dir=str(base_config.output_dir),
        experiment_name="composed_variation_demo",
    )

    result = composed_strategy.execute(base_config.eyes, composed)
    print(f"Saved composed variation data to: {result['saved_files']}")

    # Demonstrate SequentialVariation (one after another)
    sequential = SequentialVariation(
        variations=[eye_pos_var, pupil_size_var, gaze_mov_var], param_name="eye_pos_pupil_gaze_sequential"
    )

    sequential_strategy = DataGenerationStrategy(
        cameras=base_config.cameras,
        lights=base_config.lights,
        gaze_target=base_config.gaze_target,  # This will be used if the composed variation doesn't have a target variation
        output_dir=str(base_config.output_dir),
        experiment_name="sequential_variation_demo",
    )

    print("\nGenerating data for SequentialVariation ...")
    result = sequential_strategy.execute(base_config.eyes, sequential)
    print(f"Saved sequential variation data to: {result['saved_files']}")


if __name__ == "__main__":
    main()
