"""EyeLink 1000 Plus physical setup dimensions and calibration configuration.

All distances in meters. The coordinate system is centered on the screen:
    x  — horizontal (positive = right)
    y  — depth from screen (positive = away from screen toward the eye)
    z  — vertical (positive = up)

Layout (side view, not to scale):
    Screen (y=0)  <-475mm->  Camera  <-505mm->  Eye (y=980mm)
                              (below screen center)
"""

from pyetsimul.types import Position3D

# Screen dimensions
SCREEN_WIDTH = 376e-3
SCREEN_HEIGHT = 301e-3
SCREEN_HALF_W = SCREEN_WIDTH / 2
SCREEN_HALF_H = SCREEN_HEIGHT / 2

# EyeLink HV9 calibration area (88% x 83% of screen)
CAL_AREA_X = 0.88
CAL_AREA_Y = 0.83
CAL_HALF_W = SCREEN_WIDTH * CAL_AREA_X / 2
CAL_HALF_H = SCREEN_HEIGHT * CAL_AREA_Y / 2

# Camera and eye distances from screen plane
EYE_TO_SCREEN = 980e-3
CAMERA_TO_SCREEN = 475e-3

# Real-world heights from ground
SCREEN_BOTTOM_FROM_GROUND = 0.12  # screen bottom edge
CAMERA_FROM_GROUND = 0.15  # camera center
LIGHT_FROM_GROUND = 0.15  # IR light source
EYE_FROM_GROUND = 0.42  # eye center at chin rest height

# Horizontal offsets from screen center
# The IR light is mounted on the camera arm, so it always moves with the camera.
# LIGHT_X = CAMERA_X + LIGHT_CAMERA_OFFSET (26.5 cm to the right of camera)
CAMERA_X = -0.18  # 18 cm to the left
LIGHT_X = CAMERA_X + 0.265  # IR light: 26.5 cm to the right of camera
EYE_X_RIGHT = 0.03  # right eye: 3 cm to the right
EYE_X_LEFT = -0.03  # left eye: 3 cm to the left

# Vertical offsets derived from ground heights (screen center is the z=0 reference)
SCREEN_CENTER_FROM_GROUND = SCREEN_BOTTOM_FROM_GROUND + SCREEN_HALF_H
CAMERA_Z = CAMERA_FROM_GROUND - SCREEN_CENTER_FROM_GROUND
LIGHT_Z = LIGHT_FROM_GROUND - SCREEN_CENTER_FROM_GROUND
EYE_Z = EYE_FROM_GROUND - SCREEN_CENTER_FROM_GROUND

# HV9 calibration grid: 9 targets on a cross + corners pattern
HV9_CALIBRATION_POINTS: list[Position3D] = [
    Position3D(0.0, 0.0, 0.0),  # Center
    Position3D(0.0, 0.0, CAL_HALF_H),  # Top center
    Position3D(0.0, 0.0, -CAL_HALF_H),  # Bottom center
    Position3D(-CAL_HALF_W, 0.0, 0.0),  # Left center
    Position3D(CAL_HALF_W, 0.0, 0.0),  # Right center
    Position3D(-CAL_HALF_W, 0.0, CAL_HALF_H),  # Top-left
    Position3D(CAL_HALF_W, 0.0, CAL_HALF_H),  # Top-right
    Position3D(-CAL_HALF_W, 0.0, -CAL_HALF_H),  # Bottom-left
    Position3D(CAL_HALF_W, 0.0, -CAL_HALF_H),  # Bottom-right
]
