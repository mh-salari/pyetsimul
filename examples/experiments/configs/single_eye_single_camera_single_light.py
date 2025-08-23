"""Gaze movement experiment configuration.

This configuration defines a single eye with one camera and one light source
for basic gaze movement tracking experiments.
"""

from pathlib import Path
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.experimental_designs import GazeMovement
from pyetsimul.core import Eye, Camera, Light

# Experiment metadata
experiment_name = "single_eye_single_camera_single_light"
output_dir = Path(__file__).parent.parent / "outputs" / experiment_name

# Eye configuration
eye = Eye()
eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
eye.position = Position3D(0, 0.550, 0.350)
eyes = [eye]

# Camera configuration
camera = Camera(err=0.0, err_type="gaussian")
camera.position = Position3D(0, 0, 0).to_point3d()
camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
camera.point_at(Position3D(0, 0.550, 0.350))
cameras = [camera]

# Light configuration
lights = [Light(position=Position3D(0.200, 0, 0.350))]

# Gaze movement pattern
movement_pattern = GazeMovement(
    grid_center=Position3D(0, 0, 0.200),
    dx=[-0.200, 0.200],
    dy=[0.0, 0.0],
    dz=[-0.150, 0.150],
    grid_size=[16, 1, 16],
)
