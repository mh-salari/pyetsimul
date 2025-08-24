"""Multi gaze movement experiment configuration with binocular setup.

Configuration for gaze movement analysis using two eyes, two cameras,
and four light sources for binocular tracking.
"""

from pathlib import Path
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.experimental_designs import GazeMovement
from pyetsimul.core import Eye, Camera, Light

# Experiment metadata
experiment_name = "gaze_movement_multi"
output_dir = Path(__file__).parent.parent / "outputs"

# Eye configuration
first_eye = Eye()
first_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
first_eye.position = Position3D(0, 0.550, 0.350)

second_eye = Eye()
second_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
second_eye.position = Position3D(0.064, 0.550, 0.350)

eyes = [first_eye, second_eye]

# Camera configuration
first_camera = Camera(err=0.0, err_type="gaussian")
first_camera.position = Position3D(0, 0, 0)
first_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
first_camera.point_at(Position3D(0, 0.550, 0.350))

second_camera = Camera(err=0.0, err_type="gaussian")
second_camera.position = Position3D(0.05, 0, 0)
second_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
second_camera.point_at(Position3D(0.064, 0.550, 0.350))

cameras = [first_camera, second_camera]

# Light configuration
lights = [
    Light(position=Position3D(-0.100, 0.100, 0.350)),
    Light(position=Position3D(0.100, 0.100, 0.350)),
    Light(position=Position3D(-0.100, -0.100, 0.350)),
    Light(position=Position3D(0.100, -0.100, 0.350)),
]

# Gaze movement pattern
movement_pattern = GazeMovement(
    grid_center=Position3D(0, 0, 0.200),
    dx=[-0.200, 0.200],
    dy=[0.0, 0.0],
    dz=[-0.150, 0.150],
    grid_size=[16, 1, 16],
)
