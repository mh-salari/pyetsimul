"""Pupil shape models and size decentration: one eye, small vs big, overlaid.

Renders a PyEtSimul eye at two pupil sizes (constricted and dilated), each with the realistic (Wyatt 1995)
and the elliptical pupil model, and with Wildenmann 2013 pupil-size decentration enabled, then overlays what one
camera sees of all four on a single plot. Colour is the size (constricted vs dilated), line style is the shape
model (solid realistic, dashed ellipse). Two things show: the shape model (realistic vs ellipse, at matched
size) and the decentration: as the pupil dilates its centre shifts while the glint stays fixed, so the
pupil-glint vector moves. That moving vector, with no eye rotation, is the pupil-size artifact (PSA).
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.core.pupil_decentration import PupilDecentrationConfig
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_camera_view_of_eye

# Each size carries its plot colour and the marker drawn at its decentred centre; each model its line style.
SIZES = [("constricted", 2.5, "#1f77b4", "+"), ("dilated", 7.0, "#d62728", "x")]  # name, diameter mm, colour, marker
MODELS = [("realistic", "-"), ("elliptical", "--")]  # name, line style

base = get_eye_model("PyEtSimul")
# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
# A glint is the corneal reflection of a light; it does not move with pupil size, so it is the fixed reference
# the pupil centre shifts against. Two lights simply give two glints.
lights = [Light(position=Position3D(120, 0, 0)), Light(position=Position3D(-120, 0, 0))]
camera = Camera()
camera.point_at(Position3D(0, 250, 100))  # aim the camera at where the eye sits


def make_eye(pupil_type: str) -> Eye:
    """A left PyEtSimul eye with the given pupil model and pupil-size decentration on, looking at the target."""
    # Enabling decentration turns on the Wildenmann 2013 model: the pupil centre shifts as the diameter changes
    # (its direction is eye-specific), which is the physical cause of the PSA.
    model = base.copy(pupil_type=pupil_type, decentration_config=PupilDecentrationConfig(enabled=True))
    eye = Eye(model=model, which_eye="left")
    eye.position = Position3D(0, 250, 100)  # place the eye in the scene
    # Rest the eye facing the scene so look_at is a small realistic rotation rather than a ~90 deg swing.
    eye.set_rest_orientation_at_target(Position3D(0, 0, 100), up=Position3D(0, 0, 1))
    eye.look_at(Position3D(-50, 0, 50))  # rotate the eye to gaze at the target
    return eye


# One eye per shape model; each is then imaged at both sizes, so the four views share a scene and the glints.
eyes = {name: make_eye(name) for name, _ in MODELS}
images, colors, linestyles, markers = [], [], [], []
for _, diameter, color, marker in SIZES:
    for model_name, linestyle in MODELS:
        eye = eyes[model_name]
        eye.set_pupil_diameter(diameter)  # decentration moves the centre to match this new size
        images.append(camera.take_image(eye, lights))
        colors.append(color)
        linestyles.append(linestyle)
        markers.append(marker)
cameras = [camera] * len(images)

# The apparent pupil centre moves between the two sizes while the glint does not; that pupil-glint shift, with
# the eye held still, is the PSA expressed in pixels.
constricted, dilated = images[0].pupil_center, images[2].pupil_center
shift = ((constricted.x - dilated.x) ** 2 + (constricted.y - dilated.y) ** 2) ** 0.5
print(f"apparent pupil-centre shift, constricted -> dilated: {shift:.2f} px (the glint stays fixed)")

# Overlay all four: colour separates the sizes, line style the shape models, the marker each size's centre.
fig, ax = plt.subplots(figsize=(8, 8))
plot_camera_view_of_eye(
    images,
    cameras,
    camera_colors=colors,
    pupil_linestyles=linestyles,
    center_markers=markers,
    glint_colors=["gold"],  # the glint is the single fixed reference here, so keep it one colour
    ax=ax,
    zoom=True,
    legend=False,
)

# Build the legend by hand so each entry names its size, shape model, and feature in the same colour as drawn.
handles = []
for size_name, _, color, marker in SIZES:
    for model_name, linestyle in MODELS:
        handles.append(Line2D([], [], color=color, lw=1.0, ls=linestyle, label=f"{size_name} {model_name}"))
    handles.append(Line2D([], [], color=color, marker=marker, ls="none", ms=5, mew=0.9, label=f"{size_name} centre"))
handles.append(Line2D([], [], color="gold", marker="*", ls="none", ms=6, label="glint (fixed)"))

ax.set_title("Pupil shape models with size decentration", fontsize=9)
ax.legend(handles=handles, loc="upper right", fontsize=6)
ax.set_xlabel("X (pixels)", fontsize=8)
ax.set_ylabel("Y (pixels)", fontsize=8)
ax.tick_params(labelsize=7)
for spine in ax.spines.values():
    spine.set_visible(False)
plt.show()
