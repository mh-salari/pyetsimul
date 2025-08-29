"""Generate eye tracking data using configurations."""

from pyetsimul.simulation import DataGenerationStrategy
from config import (
    create_eye_position_config,
    create_gaze_movement_config,
    create_pupil_size_config,
    create_angle_kappa_config,
    create_corneal_radius_config,
    create_corneal_thickness_config,
    create_individual_differences_config,
    create_observer_individual_config,
    create_corneal_shape_config,
)


def main():
    configs = [
        create_eye_position_config(),
        create_gaze_movement_config(),
        create_pupil_size_config(),
        create_angle_kappa_config(),
        create_corneal_radius_config(),
        create_corneal_thickness_config(),
        create_individual_differences_config(),
        create_observer_individual_config(),
        create_corneal_shape_config(),
    ]

    print(f"Generating data for {len(configs)} configurations...\n")

    for i, config in enumerate(configs, 1):
        print(f"[{i}/{len(configs)}] Generating {config.experiment_name}")
        print(f"  Variation type: {config.variation.__class__.__name__}")
        print(f"  Total samples: {len(config.variation)}")

        strategy = DataGenerationStrategy(
            cameras=config.cameras,
            lights=config.lights,
            gaze_target=config.gaze_target,
            output_dir=str(config.output_dir),
            experiment_name=config.experiment_name,
        )
        strategy.execute(config.eyes, config.variation)
        print(f"  ✓ Completed: {config.experiment_name}\n")


if __name__ == "__main__":
    main()
