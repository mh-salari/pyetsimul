"""Cornea models compared: spherical vs conic vs toric, on one eye, at one gaze.

Builds three PyEtSimul eyes that differ only in cornea type: a sphere, a conic (asphericity k), and a toric
(corneal astigmatism). It holds the apex radius and everything else fixed, then overlays what one camera sees
of each. The apparent pupil is almost identical across the three because refraction is dominated by the shared
apex curvature, while the glints separate because each reflects off the cornea away from the apex, where the
shapes diverge. So cornea type is a glint-dominated effect; the apparent pupil is robust to it. The printed
spreads quantify this, and the overlay distinguishes the coincident pupils/centres by line style and marker.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from pyetsimul.core import Camera, Eye, Light, create_cornea
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_camera_view_of_eye

TYPES = ["spherical", "conic", "toric"]
PUPIL_COLORS = ["#d62728", "#1f77b4", "#9467bd"]  # red, blue, purple (pupil + centre)
GLINT_COLORS = ["#f1c40f", "#2ca02c", "#e67e22"]  # yellow, green, orange
LINESTYLES = ["-", ":", "--"]
CENTRE_MARKERS = ["+", "x", "."]

base = get_eye_model("PyEtSimul")
radius = base.cornea.anterior_radius  # held fixed across types, so only the cornea shape varies
# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
lights = [Light(position=Position3D(120, 0, 0)), Light(position=Position3D(-120, 0, 0))]  # each gives one glint
target = Position3D(-120, 0, 100)  # eccentric gaze, so the pupil and glints sample the cornea off its apex


def render(cornea_type: str) -> tuple:
    """Image one PyEtSimul eye, with the given cornea type, looking at the target."""
    cornea = create_cornea(cornea_type, base.cornea.center_init, anterior_radius=radius)
    eye = Eye(model=base.copy(cornea=cornea))
    eye.position = Position3D(0, 250, 100)  # place the eye in the scene
    eye.set_rest_orientation_at_target(Position3D(0, 0, 100), up=Position3D(0, 0, 1))  # rest it facing the scene
    eye.look_at(target)  # rotate the eye to gaze at the target
    camera = Camera()
    camera.point_at(eye.position)  # aim the camera at the eye
    return camera.take_image(eye, lights), camera


results = [render(t) for t in TYPES]
images = [image for image, _ in results]
cameras = [camera for _, camera in results]

pupil_centres = np.array([[img.pupil_center.x, img.pupil_center.y] for img in images])
glints = np.array([[img.corneal_reflections[0].x, img.corneal_reflections[0].y] for img in images])
print(f"pupil-centre spread across cornea types: {np.linalg.norm(np.ptp(pupil_centres, axis=0)):.2f} px")
print(f"glint spread across cornea types:        {np.linalg.norm(np.ptp(glints, axis=0)):.2f} px")

fig, ax = plt.subplots(figsize=(8, 8))
plot_camera_view_of_eye(
    images,
    cameras,
    camera_colors=PUPIL_COLORS,
    glint_colors=GLINT_COLORS,
    pupil_linestyles=LINESTYLES,
    center_markers=CENTRE_MARKERS,
    ax=ax,
    zoom=True,
    legend=False,
)

handles = []
for cornea_type, pupil, glint, linestyle, marker in zip(
    TYPES, PUPIL_COLORS, GLINT_COLORS, LINESTYLES, CENTRE_MARKERS, strict=True
):
    handles.extend((
        Line2D([], [], color=pupil, lw=1.0, ls=linestyle, label=f"pupil {cornea_type}"),
        Line2D([], [], color=pupil, marker=marker, ls="none", ms=5, mew=0.9, label=f"pupil centre {cornea_type}"),
        Line2D([], [], color=glint, marker="*", ls="none", ms=6, label=f"glint {cornea_type}"),
    ))

ax.set_title("PyEtSimul cornea types overlaid", fontsize=9)
ax.legend(handles=handles, loc="upper right", fontsize=6)
ax.set_xlabel("X (pixels)", fontsize=8)
ax.set_ylabel("Y (pixels)", fontsize=8)
ax.tick_params(labelsize=7)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.show()
