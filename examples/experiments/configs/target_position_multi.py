"""Multi-eye target position variation experiment configuration."""

from pathlib import Path
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.parameter_variations import create_target_position_config
from pyetsimul.core import Eye, Camera, Light

# First eye configuration
first_eye = Eye()
first_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
first_eye.position = Position3D(0, 0.550, 0.350)

# Second eye configuration
second_eye = Eye()
second_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
second_eye.position = Position3D(0.064, 0.550, 0.350)
second_eye.set_pupil_radii(5e-3, 5e-3)  # 5mm pupil radii (x, y)

# First camera configuration
first_camera = Camera(err=0.0, err_type="gaussian")
first_camera.position = Position3D(0, 0, 0)
first_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
first_camera.point_at(Position3D(0, 0.550, 0.350))

# Second camera configuration
second_camera = Camera(err=0.0, err_type="gaussian")
second_camera.position = Position3D(0.05, 0, 0)
second_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
second_camera.point_at(Position3D(0.064, 0.550, 0.350))

# Light configuration
lights = [
    Light(position=Position3D(-0.100, 0.100, 0.350)),
    Light(position=Position3D(0.100, 0.100, 0.350)),
    Light(position=Position3D(-0.100, -0.100, 0.350)),
    Light(position=Position3D(0.100, -0.100, 0.350)),
]

# Experiment configuration with parameter validation
config = create_target_position_config(
    experiment_name="target_position_multi",
    grid_center=Position3D(0, 0, 0.200),
    dx=[-0.200, 0.200],
    dy=[0.0, 0.0],
    dz=[-0.150, 0.150],
    grid_size=[16, 1, 16],
    eyes=[first_eye, second_eye],
    cameras=[first_camera, second_camera],
    lights=lights,
    output_dir=Path(__file__).parent.parent / "outputs",
)
