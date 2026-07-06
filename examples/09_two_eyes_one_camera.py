"""Two eyes, one camera: a binocular pair imaged by a single camera.

A binocular eye tracker watches two eyes, not one. This places a left and a right eye at a ~64 mm interpupillary
distance, both converging on one screen point, and a single camera that frames both. which_eye mirrors the
eye-specific geometry (angle kappa, pupil decentration) between the two sides, so the eyes are not identical
copies. plot_setup_and_camera_view draws the 3D scene and the camera view; with one camera the two eyes are drawn
in different colours, so the left pupil reads apart from the right, each with its centre and the two glints from
the two lights.
"""

import matplotlib.pyplot as plt

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_setup_and_camera_view

base = get_eye_model("PyEtSimul")
# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
target = Position3D(0, 0, 25)  # one screen point both eyes converge on

# A left and a right eye ~64 mm apart, 150 mm in front of the camera and 50 mm up. which_eye mirrors the
# eye-specific geometry (angle kappa, pupil decentration) between the two sides.
eye_left = Eye(model=base, which_eye="left")
eye_left.position = Position3D(-32, 150, 50)  # left of centre
eye_left.set_rest_orientation_at_target(target, up=Position3D(0, 0, 1))  # rest it facing the screen

eye_right = Eye(model=base, which_eye="right")
eye_right.position = Position3D(32, 150, 50)  # right of centre
eye_right.set_rest_orientation_at_target(target, up=Position3D(0, 0, 1))  # rest it facing the screen

# Two lights, so every eye shows two glints.
lights = [Light(position=Position3D(-15, 0, 0)), Light(position=Position3D(15, 0, 0))]

# One camera, centred, framing both eyes; the longer focal length zooms in so the pupils are large enough to see.
camera = Camera()
camera.camera_matrix.focal_length = 1500  # px
camera.position = Position3D(0, 0, 0)
# Aim at the eye midpoint, rolled so the line between the eyes stays horizontal in the image. This camera is
# centred so the roll is zero, but point_at_binocular is the right way to aim a camera at two eyes.
camera.point_at_binocular(eye_left.position, eye_right.position)

# The plot applies each eye's look_at target, then draws the 3D scene and the camera view (eyes in distinct colours).
fig = plot_setup_and_camera_view(
    eyes=[eye_left, eye_right],
    look_at_targets=[target, target],  # both eyes converge on the same screen point
    cameras=camera,
    lights=lights,
)
plt.show()
