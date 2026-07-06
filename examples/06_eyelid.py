"""Eyelid openness: how the lid occludes the pupil as it closes.

Renders the same PyEtSimul eye in 3D at three eyelid openness levels (fully open down to nearly closed) at one
gaze direction. As the lid lowers it covers more of the eye, and since the apparent pupil is only the part of
the iris the camera can still see, the pupil opening is progressively clipped. That occlusion is the geometric
reason a low or blinking lid degrades pupil (and glint) detection in video eye tracking. plot_eye_anatomy draws
the globe, cornea, pupil opening, optical and visual axes, and the eyelid opening edge, hiding the pupil points
the lid covers.
"""

import matplotlib.pyplot as plt

from pyetsimul.core import Eye
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_eye_anatomy

OPENNESS = [1.0, 0.6, 0.3]  # fraction the eyelid is open
base = get_eye_model("PyEtSimul")
# Positions are in millimetres; +x right, +y depth, +z up.
target = Position3D(10, 10, -10)  # one gaze direction, shared by all three eyes

fig = plt.figure(figsize=(15, 5.5))
axes = []
for i, openness in enumerate(OPENNESS):
    # A fresh eye per panel so each keeps its own openness; the eyes differ only in how far the lid is open.
    eye = Eye(model=base.copy(eyelid_enabled=True))
    eye.eyelid.openness = openness  # how far the lid is open (1 = fully open)
    eye.set_rest_orientation_at_target(target)  # rest it facing the gaze direction
    eye.look_at(target)  # rotate the eye to gaze at the target
    ax = fig.add_subplot(1, len(OPENNESS), i + 1, projection="3d")
    plot_eye_anatomy(eye, ax=ax)  # hides the pupil points the lid occludes
    for text in ax.texts:  # drop the auto openness label; the title carries it
        text.remove()
    ax.set_title(f"eyelid {openness:.0%} open", fontsize=10)
    axes.append(ax)

# Share one viewing angle and axis box so the three read as the same eye closing.
xlim, ylim, zlim = axes[0].get_xlim(), axes[0].get_ylim(), axes[0].get_zlim()
for ax in axes:
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=-30, azim=45, roll=60)

# One shared legend rather than a crowded one per panel.
handles, labels = axes[0].get_legend_handles_labels()
for ax in axes:
    ax.get_legend().remove()
fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=7)

plt.tight_layout(rect=(0, 0.08, 1, 1))
plt.show()
