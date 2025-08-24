"""Single eye position variation experiment configuration."""

from pathlib import Path
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.parameter_variations import create_eye_position_config

# Eye configuration
eye = Eye()
eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
eye.position = Position3D(0, 0.550, 0.350)

# Camera configuration
camera = Camera(err=0.0, err_type="gaussian")
camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
camera.point_at(eye.position)

# Light configuration
light = Light(position=Position3D(0.200, 0, 0.350))

# Experiment configuration with parameter validation
config = create_eye_position_config(
    experiment_name="eye_position_single",
    eye_center=Position3D(0, 0.550, 0.350),
    gaze_target=Position3D(0, 0, 0.200),
    dx=[-0.050, 0.050],
    dy=[-0.050, 0.050],
    dz=[0.0, 0.0],
    grid_size=[16, 16, 1],
    eyes=[eye],
    cameras=[camera],
    lights=[light],
    output_dir=Path(__file__).parent.parent / "outputs",
)
