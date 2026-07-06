"""Custom parameter variations: sweep any eye parameter, built-in or your own.

A variation drives DataGenerationStrategy over a range of one eye parameter, rendering the eye at each value.
PyEtSimul ships variations for common anatomy (PupilSizeVariation, CorneaRadiusVariation, AngleKappaVariation),
and GenericEyeVariation sweeps any eye parameter by its dotted path, so a custom sweep needs no new class. This
runs a built-in pupil-size sweep and a custom conic-constant sweep, each generating a small labelled dataset.
"""

from pyetsimul.core import Camera, ConicCornea, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.simulation import DataGenerationStrategy, GenericEyeVariation, PupilSizeVariation
from pyetsimul.types import Position3D

# A generic screen-based eye-tracking setup. Distances in millimetres; frame centred on the screen:
# +x right, +y from the screen toward the eye, +z up.
base = get_eye_model("PyEtSimul")
eye = Eye(model=base.copy(cornea=ConicCornea()))  # a conic cornea, so it has a conic constant to sweep
eye.position = Position3D(0.0, 700.0, 50.0)  # in front of the screen (+y), a little above its centre

camera = Camera()
camera.position = Position3D(0.0, 350.0, -150.0)  # between the screen and the eye, below it, looking up
camera.point_at(eye.position)  # aim the camera at the eye

light = Light(position=Position3D(70.0, 350.0, -140.0))  # its corneal reflection is the glint

# The strategy renders the eye at every value of a variation; save_to_file=False keeps the datasets in memory.
data_gen = DataGenerationStrategy(
    eyes=[eye],
    cameras=[camera],
    lights=[light],
    gaze_target=Position3D(0.0, 0.0, 0.0),  # the eye fixates the screen centre while the parameter varies
    experiment_name="pupil_sweep",
    save_to_file=False,
)

# A built-in parameter variation: sweep the pupil diameter over a range in a number of steps.
pupil_sweep = PupilSizeVariation(diameter_range=[2.5, 7.0], num_steps=6)
pupil_data = data_gen.execute(pupil_sweep)
print(f"{pupil_sweep.describe()}: {pupil_data['total_measurements']} measurements")

# A custom parameter variation: GenericEyeVariation sweeps any eye parameter by its dotted path, here the conic
# constant of the anterior cornea ("cornea.anterior_radius", "fovea_alpha_deg", etc. work the same way). Subclass
# GenericEyeVariation for a named, reusable variation with its own describe().
cornea_k_sweep = GenericEyeVariation("cornea.anterior_k", value_range=[-0.1, -0.5], num_steps=5)
data_gen.set_experiment_name("cornea_k_sweep")
k_data = data_gen.execute(cornea_k_sweep)
print(f"{cornea_k_sweep.describe()}: {k_data['total_measurements']} measurements")
