"""Glint detection noise: how measurement error displaces the corneal reflection.

A glint is the corneal reflection of a light; a detector locates it with some error. PyEtSimul models that error
with GlintNoiseConfig on the camera (gaussian, uniform, a constant offset, or a correlated bias), which
take_image applies to the glint. This renders the same eye through one noise-free camera and several noisy ones,
draws the pupil and the true glint once, and marks each noisy camera's glint, so you see how each model shifts
the corneal reflection relative to the true one.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pyetsimul.camera_noise import GlintNoiseConfig
from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import Position3D
from pyetsimul.visualization import plot_camera_view_of_eye

# One glint-noise model per entry; take_image applies it to the corneal reflection.
NOISE = [
    ("gaussian 2 px", GlintNoiseConfig(noise_type="gaussian", std=2.0, seed=71), "#1f77b4", "o"),
    ("uniform 2.5 px", GlintNoiseConfig(noise_type="uniform", std=2.5, seed=71), "#2ca02c", "s"),
    ("constant +2 px x", GlintNoiseConfig(noise_type="constant_offset", offset_x=2.0, offset_y=0.0), "#9467bd", "^"),
    (
        "correlated bias",
        GlintNoiseConfig(mean=[1.0, -1.5], covariance=[[3.0, 1.2], [1.2, 2.4]], seed=71),
        "#d62728",
        "D",
    ),
]

base = get_eye_model("PyEtSimul")
eye = Eye(model=base)
# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
eye.position = Position3D(0, 250, 100)  # place the eye in the scene
eye.set_rest_orientation_at_target(Position3D(0, 0, 100), up=Position3D(0, 0, 1))  # rest it facing the scene
eye.look_at(Position3D(-50, 0, 50))  # rotate the eye to gaze at the target
lights = [Light(position=Position3D(120, 0, 0))]  # its corneal reflection is the glint

# The noise-free camera gives the pupil and the true, noise-free glint position.
ref_camera = Camera()
ref_camera.point_at(eye.position)
ref_image = ref_camera.take_image(eye, lights)
true_glint = ref_image.corneal_reflections[0]

fig, ax = plt.subplots(figsize=(7, 7))
plot_camera_view_of_eye(ref_image, ref_camera, ax=ax, zoom=True, legend=False, marker_size=14)

handles = [
    Line2D([], [], color="black", lw=0.9, label="pupil"),
    Line2D([], [], color="gold", marker="*", ls="none", ms=9, label="true glint"),
]
for name, config, color, marker in NOISE:
    # Giving the noise model to the camera is what applies it: take_image perturbs the glint it computes.
    camera = Camera(glint_noise_config=config)
    camera.point_at(eye.position)
    glint = camera.take_image(eye, lights).corneal_reflections[0]
    ax.scatter(
        glint.x, glint.y, color=color, marker=marker, s=45, alpha=0.5, zorder=5, edgecolor="black", linewidth=0.3
    )
    handles.append(Line2D([], [], color=color, marker=marker, ls="none", ms=6, mec="black", mew=0.3, label=name))
    offset = ((glint.x - true_glint.x) ** 2 + (glint.y - true_glint.y) ** 2) ** 0.5
    print(f"{name:18s} glint ({glint.x:6.2f}, {glint.y:6.2f})  offset from true {offset:.2f} px")

# The true glint last and opaque, larger than the noisy markers, so it reads as the reference on top.
ax.scatter(true_glint.x, true_glint.y, color="gold", marker="*", s=110, zorder=7, edgecolor="black", linewidth=0.5)

ax.set_title("Glint detection noise", fontsize=10)
ax.legend(handles=handles, loc="upper right", fontsize=7)
ax.set_xlabel("X (pixels)", fontsize=8)
ax.set_ylabel("Y (pixels)", fontsize=8)
ax.tick_params(labelsize=7)
plt.show()
