"""Build a custom eye: an EyeModel is a spec you can construct or copy(), and Eye(model=...) uses it.

An eye in PyEtSimul is defined by an immutable EyeModel: its cornea, angle kappa, rotation centre, look-at
convention and pupil. The named models ("PyEtSimul", "et_simul", "gkaModelEye") are just EyeModels, and you
build your own the same way. This constructs an EyeModel that reproduces the validated et_simul eye field by
field, confirms it renders the same pupil and glint as the built-in "et_simul", then uses copy() to change one
field and shows the render move, the point of an editable eye model.
"""

from tabulate import tabulate

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.cornea import SphericalCornea
from pyetsimul.core.eye_model import EyeModel, get_eye_model
from pyetsimul.core.off_axis_pupil import OffAxisPupilConfig
from pyetsimul.core.pupil_decentration import PupilDecentrationConfig
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


# Build an EyeModel from scratch, reproducing the validated et_simul eye: a spherical cornea (no posterior
# surface), angle kappa 6/2 deg, a single fixed eyeball-centre pivot, and the original optical-then-kappa look-at.
my_et_simul = EyeModel(
    cornea=SphericalCornea(
        anterior_radius=7.98,
        refractive_index=1.376,
        use_posterior_surface=False,
        placement_convention="center",
        scale_with_radius=True,
    ),
    axial_length=24.75,
    fovea_alpha_deg=6.0,
    fovea_beta_deg=2.0,
    look_at_method="optical_then_kappa",
    rotation_center=EyeballCenter(),
    decentration_config=PupilDecentrationConfig(enabled=False),
    off_axis_pupil=OffAxisPupilConfig(enabled=False),
    default_pupil_diameter=6.0,
)

# copy() returns a new EyeModel with fields changed (the model is frozen); here flatten the cornea to R = 8.4 mm.
flatter = my_et_simul.copy(
    cornea=SphericalCornea(
        anterior_radius=8.4,
        refractive_index=1.376,
        use_posterior_surface=False,
        placement_convention="center",
        scale_with_radius=True,
    )
)

mine_pupil, mine_glint = render(my_et_simul)
ref_pupil, ref_glint = render(get_eye_model("et_simul"))  # the built-in validated eye
flat_pupil, flat_glint = render(flatter)

rows = [
    [
        "my et_simul (built here)",
        f"({mine_pupil.x:.2f}, {mine_pupil.y:.2f})",
        f"({mine_glint.x:.2f}, {mine_glint.y:.2f})",
    ],
    ["built-in et_simul", f"({ref_pupil.x:.2f}, {ref_pupil.y:.2f})", f"({ref_glint.x:.2f}, {ref_glint.y:.2f})"],
    [
        "+ flatter cornea (copy)",
        f"({flat_pupil.x:.2f}, {flat_pupil.y:.2f})",
        f"({flat_glint.x:.2f}, {flat_glint.y:.2f})",
    ],
]
print(tabulate(rows, headers=["eye model", "pupil centre (px)", "glint (px)"], tablefmt="grid"))

# The reproduction should match the built-in eye; the flatter cornea should move the glint.
match = max(
    abs(mine_pupil.x - ref_pupil.x),
    abs(mine_pupil.y - ref_pupil.y),
    abs(mine_glint.x - ref_glint.x),
    abs(mine_glint.y - ref_glint.y),
)
moved = ((flat_glint.x - mine_glint.x) ** 2 + (flat_glint.y - mine_glint.y) ** 2) ** 0.5
print(f"max difference from the built-in et_simul: {match:.4f} px")
print(f"glint shift from flattening the cornea:    {moved:.2f} px")
