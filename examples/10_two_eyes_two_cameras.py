"""Two eyes, two cameras: the binocular setup and telling which pupil is which.

With two eyes and two cameras the camera view holds four pupils (two cameras x two eyes), and without a code it
is impossible to say which is which. The left panel is the 3D setup (both eyes, both cameras, the lights); the
right panel is the camera view, where each pupil gets two cues: its BORDER COLOUR says which camera saw it
(matching that camera's colour in the 3D setup) and its LINE STYLE says which eye. A left and a right eye sit at
a ~64 mm interpupillary distance, both converging on one screen point; two cameras at screen level each frame the
pair. which_eye mirrors the eye-specific geometry (angle kappa, pupil decentration) between the two sides. The
legend names both cues, and the two glints per eye are coloured by their light.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_camera_view_of_eye, plot_setup, prepare_eye_data_for_plots
from pyetsimul.visualization.plot_config import create_plot_config

EYE_STYLES = ["-", "--"]  # pupil line style per eye: solid = left, dashed = right

base = get_eye_model("PyEtSimul")
# Positions are in millimetres; +x right, +y depth (away from the cameras), +z up.
target = Position3D(0, 0, 25)  # one screen point both eyes converge on

# A left and a right eye ~64 mm apart, 150 mm in front of the cameras and 50 mm up. which_eye mirrors the
# eye-specific geometry (angle kappa, pupil decentration) between the two sides.
eye_left = Eye(model=base, which_eye="left")
eye_left.position = Position3D(-32, 150, 50)  # left of centre
eye_left.set_rest_orientation_at_target(target, up=Position3D(0, 0, 1))  # rest it facing the screen
eye_left.look_at(target)  # gaze at the screen point

eye_right = Eye(model=base, which_eye="right")
eye_right.position = Position3D(32, 150, 50)  # right of centre
eye_right.set_rest_orientation_at_target(target, up=Position3D(0, 0, 1))  # rest it facing the screen
eye_right.look_at(target)  # gaze at the screen point
eyes = [eye_left, eye_right]

# Two lights, so every eye shows two glints (the camera view colours them by light).
lights = [Light(position=Position3D(-15, 0, 0)), Light(position=Position3D(15, 0, 0))]

# A stereo camera pair at screen level, 60 mm apart. Each sits off to one side, so point_at_binocular aims it at
# the eye midpoint AND rolls it so the line between the eyes stays horizontal in the image (a plain point_at
# would leave each off-axis camera rolled, tilting the two eyes to different heights).
cameras = []
for x in (-30, 30):
    camera = Camera()
    camera.camera_matrix.focal_length = 1500  # px; zoom in on the eye pair
    camera.position = Position3D(x, 0, 0)
    camera.point_at_binocular(eye_left.position, eye_right.position)  # aim at the midpoint, rolled to level the eyes
    cameras.append(camera)

# Left panel: the 3D setup. plot_setup returns the colours it gave the eyes and cameras, so the camera view can
# reuse the camera colours and the two panels agree.
prepared = prepare_eye_data_for_plots(eyes, [target, target], lights, cameras)
fig = plt.figure(figsize=(15, 6))
ax_setup = fig.add_subplot(1, 2, 1, projection="3d")
ax_camera = fig.add_subplot(1, 2, 2)
eye_colors, camera_colors = plot_setup(
    ax_setup, prepared["eyes_data"], [target, target], lights, cameras, prepared["cr_3d_lists"]
)

# Right panel: image every (camera, eye) on its own, so each pupil takes its camera's border colour (from the
# setup) and its eye's line style. The four parallel lists feed plot_camera_view_of_eye one view per pupil.
images, image_cameras, border_colors, line_styles = [], [], [], []
for cam_idx, camera in enumerate(cameras):
    for eye_idx, eye in enumerate(eyes):
        images.append(camera.take_image(eye, lights))
        image_cameras.append(camera)
        border_colors.append(camera_colors[cam_idx])  # border = which camera (matches the 3D setup)
        line_styles.append(EYE_STYLES[eye_idx])  # line style = which eye
plot_camera_view_of_eye(
    images,
    image_cameras,
    camera_colors=border_colors,  # per-image border colour (the edge-border feature) = the camera
    pupil_linestyles=line_styles,  # per-image pupil line style = the eye
    ax=ax_camera,
    zoom=True,
    legend=False,
)

# Name both cues so any of the four pupils can be read: border colour = camera, line style = eye; glints by light.
light_colors = create_plot_config().colors.lights
handles = [
    Line2D([], [], color=camera_colors[0], lw=1.2, label="camera 1"),
    Line2D([], [], color=camera_colors[1], lw=1.2, label="camera 2"),
    Line2D([], [], color="black", lw=1.2, ls="-", label="left eye"),
    Line2D([], [], color="black", lw=1.2, ls="--", label="right eye"),
    Line2D([], [], color=light_colors[0], marker="*", ls="none", ms=6, label="glint (light 1)"),
    Line2D([], [], color=light_colors[1], marker="*", ls="none", ms=6, label="glint (light 2)"),
    Line2D([], [], color="gray", marker="+", ls="none", ms=6, label="pupil centre"),
]
ax_camera.legend(handles=handles, loc="upper right", fontsize=7)
ax_camera.set_title("Camera view: two eyes, two cameras", fontsize=10)
plt.show()
