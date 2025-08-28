"""Generate eye tracking data using configurations."""

from pyetsimul.experiment_framework.data_generation import DataGenerationStrategy
from config import create_eye_position_config, create_gaze_movement_config, create_pupil_size_config


def main():
    configs = [
        create_eye_position_config(),
        create_gaze_movement_config(),
        create_pupil_size_config(),
    ]

    for config in configs:
        strategy = DataGenerationStrategy(
            cameras=config.cameras,
            lights=config.lights,
            gaze_target=config.gaze_target,
            output_dir=str(config.output_dir),
            experiment_name=config.experiment_name,
        )
        strategy.execute(config.eyes, config.variation)


if __name__ == "__main__":
    main()
