"""Multi eye movement experiment configuration with binocular setup.

Configuration for eye movement analysis using two eyes, two cameras,
and four light sources for binocular tracking.
"""

from pathlib import Path
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.experimental_designs import EyeMovement

# Experiment metadata
experiment_name = "eye_movement_multi"
output_dir = Path(__file__).parent.parent / "outputs"

# Eye configuration
left_eye = Eye()
left_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
left_eye.position = Position3D(-0.032, 0.550, 0.350)  # 32mm left of center

right_eye = Eye()
right_eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
right_eye.position = Position3D(0.032, 0.550, 0.350)  # 32mm right of center

eyes = [left_eye, right_eye]

# Camera configuration
left_camera = Camera(err=0.0, err_type="gaussian")
left_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
left_camera.position = Position3D(-0.025, 0, 0)  # 25mm left of center
left_camera.point_at(left_eye.position)

right_camera = Camera(err=0.0, err_type="gaussian")
right_camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
right_camera.position = Position3D(0.025, 0, 0)  # 25mm right of center
right_camera.point_at(right_eye.position)

cameras = [left_camera, right_camera]

# Light configuration
lights = [
    Light(position=Position3D(0.150, -0.050, 0.350)),  # Front right
    Light(position=Position3D(-0.150, -0.050, 0.350)),  # Front left
    Light(position=Position3D(0.150, 0.050, 0.350)),  # Back right
    Light(position=Position3D(-0.150, 0.050, 0.350)),  # Back left
]

# Eye movement pattern
movement_pattern = EyeMovement(
    eye_center=Position3D(0, 0.550, 0.350),  # Central position between eyes
    gaze_target=Position3D(0, 0, 0.200),  # Fixed gaze target at screen center
    dx=[-0.050, 0.050],  # X varies: ±50mm
    dy=[-0.050, 0.050],  # Y varies: ±50mm
    dz=[0.0, 0.0],  # Z fixed: no variation
    grid_size=[12, 12, 1],  # 12x12x1 = 2D grid in XY plane (smaller for binocular)
)
