"""Demonstrates simple data generation using pyetsimul.

The process is as follows:
1.  Setup Scene:
    - Create Eye object(s) with specific position(s).
    - Create Camera object(s) and point them at the eye(s).
    - Create Light source(s).

2.  Configure Data Generation:
    - Initialize a DataGenerationStrategy with the scene components (eyes, cameras, lights).
    - This strategy manages running simulations and saving data.

3.  Execute Variations:
    - Define a `TargetPositionVariation` to simulate eye(s) looking at different points on a grid.
    - Execute the simulation for this variation. The output is a JSON file with simulation data.
    - Define an `EyePositionVariation` to simulate head/eye movement.
    - Execute the simulation for this second variation.
"""

from pathlib import Path

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.simulation import (
    DataGenerationStrategy,
    EyePositionVariation,
    TargetPositionVariation,
)
from pyetsimul.types import Position3D, RotationMatrix


def main() -> None:
    """Executes the data generation simulation."""
    # 1. Setup scene components: eyes, cameras, and lights.
    # Create an Eye object. The system supports multiple eyes, but for this example, we use one.
    eye = Eye()
    eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]]))
    eye.position = Position3D(0.0, 550, 350)

    # Create a Camera object. The system supports multiple cameras.
    camera = Camera()
    camera.point_at(eye.position)  # Pointing this camera at the single eye for simplicity.

    # Create a Light object. The system supports multiple lights.
    light = Light(position=Position3D(200, 0, 350))

    # 2. Configure the data generation strategy.
    # The strategy takes lists of eyes, cameras, and lights.
    data_gen = DataGenerationStrategy(
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=Position3D(0.0, 0.0, 200),
        experiment_name="",
        output_dir=Path(__file__).parent / "outputs",
    )

    # 3. Define and execute variations.

    # First variation: Change the gaze target position (e.g., simulating looking at a screen).
    target_position_variation = TargetPositionVariation(
        grid_center=Position3D(0, 0, 200),
        dx=[-200, 200],
        dy=[0.0, 0.0],
        dz=[-150, 150],
        grid_size=[16, 1, 16],
    )

    data_gen.set_experiment_name("screen_test")
    _ = data_gen.execute(target_position_variation)

    # Second variation: Change the eye's position (e.g., simulating head movement).
    eye_position_variation = EyePositionVariation(
        center=Position3D(0.0, 550, 350),
        dx=[-50, 50],
        dy=[-50, 50],
        dz=[0.0, 0.0],
        grid_size=[16, 16, 1],
    )

    data_gen.set_experiment_name("observer_test")
    _ = data_gen.execute(eye_position_variation)


if __name__ == "__main__":
    main()
