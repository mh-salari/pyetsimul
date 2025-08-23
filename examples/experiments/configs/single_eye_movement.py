"""Single eye movement experiment configuration."""

from pathlib import Path
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.experimental_designs import EyeMovement

# Experiment metadata
experiment_name = "single_eye_movement"
output_dir = Path(__file__).parent.parent / "outputs"

# Single eye configuration - matches first eye from interpolate example
eye = Eye()
eye.set_rest_orientation(RotationMatrix([[1, 0, 0], [0, 0, 1], [0, 1, 0]], validate_handedness=False))
eye.position = Position3D(0, 0.550, 0.350)  # Initial position (will be varied)
eyes = [eye]

# Single camera configuration - matches interpolate example
camera = Camera(err=0.0, err_type="gaussian")
camera.orientation = RotationMatrix([[1, 0, 0], [0, 0, -1], [0, 1, 0]], validate_handedness=False)
camera.point_at(eye.position)
cameras = [camera]

# Single light configuration - matches interpolate example
light = Light(position=Position3D(0.200, 0, 0.350))
lights = [light]

# Eye movement pattern - eye moves in XY plane around central position
# This matches the default pattern from observer_position_analysis.py
movement_pattern = EyeMovement(
    eye_center=Position3D(0, 0.550, 0.350),  # Central eye position
    gaze_target=Position3D(0, 0, 0.200),  # Fixed gaze target at screen center
    dx=[-0.050, 0.050],  # X varies: ±50mm
    dy=[-0.050, 0.050],  # Y varies: ±50mm
    dz=[0.0, 0.0],  # Z fixed: no variation
    grid_size=[16, 16, 1],  # 16x16x1 = 2D grid in XY plane
)
