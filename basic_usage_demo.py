#!/usr/bin/env python3
"""
Basic usage demonstration for PyEtSimul paper.
Creates setup using established patterns from examples/ directory.
"""

import matplotlib.pyplot as plt
from pathlib import Path
from pyetsimul.core import Eye, Camera, Light
from pyetsimul.types import Position3D, RotationMatrix
from pyetsimul.visualization.coordinate_utils import prepare_eye_data_for_plots
from pyetsimul.visualization.setup_plots import plot_setup
from pyetsimul.visualization.camera_view import plot_camera_view_of_eye

# Create eye-tracking setup following examples/setup_visualization.py pattern
eye = Eye(eyelid_enabled=True, pupil_boundary_points=50)
eye.eyelid.openness = 0.65

rest_orientation = RotationMatrix([[-1, 0, 0], [0, 0, 1], [0, 1, 0]])
eye.set_rest_orientation(rest_orientation)
eye.position = Position3D(0, 150e-3, 50e-3)

# Create single light source
light = Light(position=Position3D(50e-3, 0, 0))

# Setup camera pointing at eye
camera = Camera()
camera.point_at(eye.position)

# Start target point
target_point = Position3D(0, 0, 50e-3)

# Make eye look at target
eye.look_at(target_point)

# Take image of eye
image = camera.take_image(eye, [light])

# Access projected features
pupil_center = image.pupil_center
corneal_reflections = image.corneal_reflections

print("Basic usage example executed successfully!")
print(f"Pupil center: {pupil_center}")
print(f"Number of corneal reflections: {len(corneal_reflections)}")
if corneal_reflections[0] is not None:
    print(f"First CR position: {corneal_reflections[0]}")

# Create side-by-side visualization manually
fig = plt.figure(figsize=(16, 8), constrained_layout=True)
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax2 = fig.add_subplot(1, 2, 2)

# Prepare data for plotting
prepared_data = prepare_eye_data_for_plots(eye, target_point, [light], camera)

# Left panel: 3D setup view
plot_setup(ax1, prepared_data["eye_data"], target_point, [light], camera, prepared_data["cr_3d_list"])

# Right panel: Camera view
plot_camera_view_of_eye(image, camera, prepared_data["cr_3d_list"], ax=ax2)

plt.tight_layout()

# Create figures directory if it doesn't exist
Path("figures").mkdir(exist_ok=True)
plt.savefig("figures/basic_usage_demo.pdf", transparent=True, bbox_inches="tight")
plt.show()
