"""Generate eye tracking data using shared configuration."""

from config import (
    angle_kappa_pupil_variation,
    angle_kappa_variation,
    corneal_radius_variation,
    corneal_shape_variation,
    corneal_thickness_variation,
    create_experiment_config,
    eye_position_angle_kappa_variation,
    eye_position_variation,
    pupil_size_variation,
    target_position_variation,
)

from pyetsimul.simulation import DataGenerationStrategy


def main() -> None:
    """Generate datasets for all variations."""
    base_config = create_experiment_config("base")

    variations = {
        "eye_position": eye_position_variation,
        "target_position": target_position_variation,
        "pupil_size": pupil_size_variation,
        "angle_kappa": angle_kappa_variation,
        "corneal_radius": corneal_radius_variation,
        "corneal_thickness": corneal_thickness_variation,
        "angle_kappa_pupil": angle_kappa_pupil_variation,
        "eye_position_angle_kappa": eye_position_angle_kappa_variation,
        "corneal_shape": corneal_shape_variation,
    }

    print(f"Generating data for {len(variations)} variations...\n")

    for i, (name, variation) in enumerate(variations.items(), 1):
        print(f"[{i}/{len(variations)}] Generating {name}")
        print(f"  Variation type: {variation.__class__.__name__}")
        print(f"  Total samples: {len(variation)}")

        strategy = DataGenerationStrategy(
            eyes=base_config.eyes,
            cameras=base_config.cameras,
            lights=base_config.lights,
            gaze_target=base_config.gaze_target,
            output_dir=str(base_config.output_dir),
            experiment_name=name,
            use_legacy_look_at=True,
        )
        strategy.execute(variation)
        print(f"  Completed: {name}\n")


if __name__ == "__main__":
    main()
