"""Multi-eye position variation experiment configuration."""

from pathlib import Path
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.parameter_variations import create_eye_position_config

# Left eye configuration
left_eye = Eye()
left_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
left_eye.position = Position3D(-0.032, 0.550, 0.350)

# Right eye configuration
right_eye = Eye()
right_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
right_eye.position = Position3D(0.032, 0.550, 0.350)

# Left camera configuration
left_camera = Camera(err=0.0, err_type="gaussian")
left_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
left_camera.position = Position3D(-0.025, 0, 0)
left_camera.point_at(left_eye.position)

# Right camera configuration
right_camera = Camera(err=0.0, err_type="gaussian")
right_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
right_camera.position = Position3D(0.025, 0, 0)
right_camera.point_at(right_eye.position)

# Light configuration
lights = [
    Light(position=Position3D(0.150, -0.050, 0.350)),
    Light(position=Position3D(-0.150, -0.050, 0.350)),
    Light(position=Position3D(0.150, 0.050, 0.350)),
    Light(position=Position3D(-0.150, 0.050, 0.350)),
]

# Experiment configuration with parameter validation
config = create_eye_position_config(
    experiment_name="eye_position_multi",
    eye_center=Position3D(0, 0.550, 0.350),
    gaze_target=Position3D(0, 0, 0.200),
    dx=[-0.050, 0.050],
    dy=[-0.050, 0.050],
    dz=[0.0, 0.0],
    grid_size=[12, 12, 1],
    eyes=[left_eye, right_eye],
    cameras=[left_camera, right_camera],
    lights=lights,
    output_dir=Path(__file__).parent.parent / "outputs",
)
