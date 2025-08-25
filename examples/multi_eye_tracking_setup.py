#!/usr/bin/env python3
"""
Realistic eye tracking setup example.
- Both eyes looking at the same point
- Two cameras 5cm apart horizontally (left-right)
- Eyes positioned above cameras (realistic screen-based setup)
"""

import matplotlib.pyplot as plt
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.visualization import plot_setup_and_camera_view

# Create two eyes with realistic spacing (6-7cm interpupillary distance)
eye1 = Eye(eyelid_enabled=True, pupil_boundary_points=50)
eye1.pupil.set_radii(50e-3, 50e-3)
eye1.eyelid.openness = 1
rest_orientation1 = RotationMatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
eye1.set_rest_orientation(rest_orientation1)
eye1.position = Position3D(-32e-3, 150e-3, 50e-3)

eye2 = Eye(eyelid_enabled=True, pupil_boundary_points=50)
eye2.pupil.set_radii(50e-3, 50e-3)
eye2.eyelid.openness = 1
rest_orientation2 = RotationMatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
eye2.set_rest_orientation(rest_orientation2)
eye2.position = Position3D(32e-3, 150e-3, 50e-3)

# Single target point that both eyes look at (screen center)
target_point = Position3D(0, 0, 25e-3)  # Screen at origin

# Create lights positioned around the cameras
lights = [
    Light(position=Position3D(-10e-3, 0, 0)),  # Left side light at screen level
    Light(position=Position3D(10e-3, 0, 0)),  # Right side light at screen level
]

# Calculate average position between the two eyes
eye_center = Position3D(
    (eye1.position.x + eye2.position.x) / 2,
    (eye1.position.y + eye2.position.y) / 2,
    (eye1.position.z + eye2.position.z) / 2 - 20e-3,
)

# Calculate average position between the two eyes
eye_center2 = Position3D(
    (eye1.position.x + eye2.position.x) / 2 + 25e-3,
    (eye1.position.y + eye2.position.y) / 2,
    (eye1.position.z + eye2.position.z) / 2 - 20e-3,
)


# Define world coordinate frame (right-handed system)
world_frame = RotationMatrix(
    [
        [1, 0, 0],  # World +X axis
        [0, 1, 0],  # World +Y axis
        [0, 0, 1],  # World +Z axis
    ]
)

focal_length = 1500
# Create two cameras 5cm apart horizontally, positioned at screen level
# Use higher focal length to zoom in and make pupils appear larger
camera1 = Camera()
camera1.camera_matrix.focal_length = focal_length
camera1.position = Position3D(0, 0, 0)  # Left camera at screen

camera1.point_at(eye_center, world_frame)  # Point with world frame alignment

camera2 = Camera()
camera2.camera_matrix.focal_length = focal_length
camera2.position = Position3D(25e-3, 0, 0)  # Second camera position
camera2.point_at(eye_center2, world_frame)  # Point with world frame alignment


fig = plot_setup_and_camera_view(
    eyes=[eye1, eye2],
    look_at_targets=[target_point, target_point],
    cameras=[camera1, camera2],
    lights=lights,
)


plt.show()
