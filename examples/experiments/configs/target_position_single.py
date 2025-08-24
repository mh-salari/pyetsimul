"""Single target position variation experiment configuration."""

from pathlib import Path
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.parameter_variations import create_target_position_config
from pyetsimul.core import Eye, Camera, Light

# Eye configuration
eye = Eye()
eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
eye.position = Position3D(0, 0.550, 0.350)

# Camera configuration
camera = Camera(err=0.0, err_type="gaussian")
camera.position = Position3D(0, 0, 0)
camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
camera.point_at(Position3D(0, 0.550, 0.350))

# Light configuration
light = Light(position=Position3D(0.200, 0, 0.350))

# Experiment configuration with parameter validation
config = create_target_position_config(
    experiment_name="target_position_single",
    grid_center=Position3D(0, 0, 0.200),
    dx=[-0.200, 0.200],
    dy=[0.0, 0.0],
    dz=[-0.150, 0.150],
    grid_size=[16, 1, 16],
    eyes=[eye],
    cameras=[camera],
    lights=[light],
    output_dir=Path(__file__).parent.parent / "outputs",
)
