"""Compare three model eyes by their parameters, what they render, and their shape.

Eye(model="...") selects a complete optical and geometric spec. The first table lists each model's parameters;
the second renders one fixed scene through each and reports the pupil centre and glint; the cross-section figure
draws each model's geometry side by side. A row that differs between the columns shows up as a difference in
shape.
"""

import matplotlib.pyplot as plt
from tabulate import tabulate

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.types import CameraImage, Position3D
from pyetsimul.visualization import plot_eye_cross_section

MODELS = ["PyEtSimul", "et_simul", "gkaModelEye"]
models = {name: get_eye_model(name) for name in MODELS}

# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
# One fixed scene. The eye is given a rest orientation facing the scene (toward -y) so its gaze is a small,
# realistic rotation rather than the ~90 deg swing the default identity rest would force.
eye_position = Position3D(0, 250, 100)  # where the eye sits in the scene
target = Position3D(-50, 0, 50)  # the point the eye gazes at
camera = Camera()
camera.point_at(eye_position)  # aim the camera at the eye
light = Light(position=Position3D(100, 0, 0))  # its corneal reflection is the glint


def render(name: str) -> CameraImage:
    """Render one model: rest it facing the scene, gaze at the target, image it through the camera."""
    eye = Eye(model=name)
    eye.position = eye_position  # place the eye in the scene
    eye.set_rest_orientation_at_target(Position3D(0, 0, 100), up=Position3D(0, 0, 1))  # rest it facing the scene
    eye.look_at(target)  # rotate the eye to gaze at the target
    return camera.take_image(eye, [light])


images = {name: render(name) for name in MODELS}

PARAMETERS = {
    "cornea model": lambda m: type(m.cornea).__name__,
    "anterior radius (mm)": lambda m: f"{m.cornea.anterior_radius:.2f}",
    "conic constant (k)": lambda m: "sphere" if (k := getattr(m.cornea, "anterior_k", None)) is None else f"{k:.2f}",
    "posterior surface": lambda m: "yes" if m.cornea.use_posterior_surface else "no",
    "refractive index (cornea / aqueous)": lambda m: f"{m.cornea.refractive_index:.3f} / {m.n_aqueous_humor:.3f}",
    "angle kappa alpha/beta (deg)": lambda m: f"{m.fovea_alpha_deg}, {m.fovea_beta_deg}",
    "rotation centre": lambda m: type(m.rotation_center).__name__,
    "look-at convention": lambda m: m.look_at_method,
}
OUTPUTS = {
    "pupil centre (px)": lambda img: f"({img.pupil_center.x:.1f}, {img.pupil_center.y:.1f})",
    "glint (px)": lambda img: f"({img.corneal_reflections[0].x:.1f}, {img.corneal_reflections[0].y:.1f})",
}

param_rows = [[label, *(read(models[name]) for name in MODELS)] for label, read in PARAMETERS.items()]
output_rows = [[label, *(read(images[name]) for name in MODELS)] for label, read in OUTPUTS.items()]

print("Model parameters")
print(tabulate(param_rows, headers=["parameter", *MODELS], tablefmt="grid"))
print("\nRendered output (the same scene through each model)")
print(tabulate(output_rows, headers=["output", *MODELS], tablefmt="grid"))

# Each model's geometry drawn side by side, in its own frame.
ROTATION_DESC = {"EyeballCenter": "fixed pivot (eyeball centre)", "RotationCenter": "gaze-dependent pivot"}
LOOKAT_DESC = {
    "visual_axis": "visual axis on target",
    "optical_then_kappa": "optical axis, then kappa",
    "optical_axis_target_direction": "apex-to-target direction",
}
fig, axes = plt.subplots(1, len(MODELS), figsize=(17, 6.5), sharey=True)
for ax, name in zip(axes, MODELS, strict=True):
    model = models[name]
    rotation = ROTATION_DESC.get(type(model.rotation_center).__name__, type(model.rotation_center).__name__)
    look_at = LOOKAT_DESC.get(model.look_at_method, model.look_at_method)
    ax.set_title(
        f"{name}\n"
        f"{type(model.cornea).__name__} | R {model.cornea.anterior_radius:.2f} mm | "
        f"kappa {model.fovea_alpha_deg:.1f}/{model.fovea_beta_deg:.1f} deg\n"
        f"rotation: {rotation}\nlook-at: {look_at}",
        fontsize=8.5,
        pad=10,
    )
    plot_eye_cross_section(Eye(model=name), ax, legend=(ax is axes[0]))
plt.show()
