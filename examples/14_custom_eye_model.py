"""Build a custom eye: an EyeModel is a spec you can construct or copy(), and Eye(model=...) uses it.

An eye in PyEtSimul is defined by an immutable EyeModel: its cornea, angle kappa, rotation centre, look-at
convention and pupil. The named models ("PyEtSimul", "et_simul", "gkaModelEye") are just EyeModels, and you
build your own the same way by setting any of those fields. This constructs a custom eye that differs from the
built-in PyEtSimul across its cornea (a two-surface conic), angle kappa, look-at convention and rotation centre,
renders it, then uses copy() to flatten its cornea and shows the render move.
"""

from tabulate import tabulate

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.core.eye_model import EyeModel, get_eye_model
from pyetsimul.core.rotation_center import EyeballCenter
from pyetsimul.types import Position3D

# Positions are in millimetres; +x right, +y depth (away from the camera), +z up.
target = Position3D(-50, 0, 100)  # an off-axis gaze target, so kappa and the cornea shape show in the render
lights = [Light(position=Position3D(120, 0, 0))]  # its corneal reflection is the glint


def render(model: EyeModel) -> tuple:
    """Image a left eye with the given EyeModel looking at the target; return its pupil centre and glint."""
    eye = Eye(model=model, which_eye="left")
    eye.position = Position3D(0, 250, 100)  # place the eye in the scene
    eye.set_rest_orientation_at_target(Position3D(0, 0, 100), up=Position3D(0, 0, 1))  # rest it facing the scene
    eye.look_at(target)  # rotate the eye to gaze at the target
    camera = Camera()
    camera.point_at(eye.position)  # aim the camera at the eye
    image = camera.take_image(eye, lights)
    return image.pupil_center, image.corneal_reflections[0]


# The built-in PyEtSimul eye is a bare EyeModel of defaults: a single-surface spherical cornea, angle kappa
# 6/2 deg, anatomical Fick rotation centres and the visual-axis look-at.
builtin = get_eye_model("PyEtSimul")

# Build a custom EyeModel from scratch. An EyeModel is far more than a cornea: it also carries the look-at
# convention, the rotation centre, angle kappa and the pupil, so a custom eye can differ from PyEtSimul across
# all of them at once.
my_custom = EyeModel(
    cornea=ConicCornea(anterior_radius=7.76, use_posterior_surface=True),  # a two-surface conic cornea
    fovea_alpha_deg=4.5,  # horizontal angle kappa (deg)
    fovea_beta_deg=1.5,  # vertical angle kappa (deg)
    look_at_method="line_of_sight",  # aim the fovea-to-pupil axis, vs the default visual_axis
    rotation_center=EyeballCenter(),  # a single fixed pivot, vs the default anatomical Fick centres
)

# copy() returns a new EyeModel with fields changed (the model is frozen); here flatten the cornea to R = 8.4 mm.
flatter = my_custom.copy(cornea=ConicCornea(anterior_radius=8.4, use_posterior_surface=True))

# What the custom eye changes, field by field, against the built-in PyEtSimul and the flatter copy.
ROTATION_DESC = {"RotationCenter": "Fick centres", "EyeballCenter": "eyeball centre"}
FIELDS = {
    "cornea model": lambda m: type(m.cornea).__name__,
    "posterior surface": lambda m: "yes" if m.cornea.use_posterior_surface else "no",
    "anterior R (mm)": lambda m: f"{m.cornea.anterior_radius:.2f}",
    "kappa alpha/beta (deg)": lambda m: f"{m.fovea_alpha_deg:.1f} / {m.fovea_beta_deg:.1f}",
    "look-at": lambda m: m.look_at_method,
    "rotation centre": lambda m: ROTATION_DESC.get(type(m.rotation_center).__name__, type(m.rotation_center).__name__),
}
config = {"PyEtSimul": builtin, "my custom": my_custom, "+ flatter copy": flatter}
field_rows = [[label, *(read(m) for m in config.values())] for label, read in FIELDS.items()]
print("What the custom eye changes")
print(tabulate(field_rows, headers=["field", *config]))

# Render each eye at the same off-axis target and read its pupil centre and glint.
renders = {
    "built-in eye model (PyEtSimul)": builtin,
    "my custom eye model": my_custom,
    "+ flatter cornea (copy)": flatter,
}
rendered = {label: render(m) for label, m in renders.items()}
render_rows = [
    [label, f"({pupil.x:.1f}, {pupil.y:.1f})", f"({glint.x:.1f}, {glint.y:.1f})"]
    for label, (pupil, glint) in rendered.items()
]
print("\nRendered output (the same off-axis target through each eye)")
print(tabulate(render_rows, headers=["eye model", "pupil centre (px)", "glint (px)"]))

builtin_pupil, builtin_glint = rendered["built-in eye model (PyEtSimul)"]
custom_pupil, custom_glint = rendered["my custom eye model"]
_, flatter_glint = rendered["+ flatter cornea (copy)"]
pupil_offset = ((custom_pupil.x - builtin_pupil.x) ** 2 + (custom_pupil.y - builtin_pupil.y) ** 2) ** 0.5
glint_offset = ((custom_glint.x - builtin_glint.x) ** 2 + (custom_glint.y - builtin_glint.y) ** 2) ** 0.5
glint_move = ((flatter_glint.x - custom_glint.x) ** 2 + (flatter_glint.y - custom_glint.y) ** 2) ** 0.5
print(
    f"\n-> vs the built-in PyEtSimul the custom eye shifts the pupil centre {pupil_offset:.1f} px "
    f"and the glint {glint_offset:.1f} px; flattening its cornea moves the glint a further {glint_move:.1f} px"
)
