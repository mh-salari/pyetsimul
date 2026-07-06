"""Custom parameter variations: sweep any eye parameter, built-in or your own.

A variation drives DataGenerationStrategy over a range of one eye parameter, rendering the eye at each value.
PyEtSimul ships variations for common anatomy (PupilSizeVariation, CorneaRadiusVariation, AngleKappaVariation),
and GenericEyeVariation sweeps any eye parameter by its dotted path, so a custom sweep needs no new class. This
runs a built-in pupil-size sweep and a custom conic-constant sweep, reporting how the rendered pupil and glint
move across each sweep.
"""

import math

from tabulate import tabulate

from pyetsimul.core import Camera, ConicCornea, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.simulation import DataGenerationStrategy, GenericEyeVariation, PupilSizeVariation
from pyetsimul.types import Position3D


def sweep_measurements(result: dict) -> list:
    """The per-value measurements of a single-eye, single-camera sweep, in sweep order."""
    return result["data"]["cameras"][0]["eyes"][0]["measurements"]


def apparent_diameter_px(boundary: list) -> float:
    """Equivalent-area diameter of a pupil contour in pixels, from its boundary points (shoelace area)."""
    area = sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in zip(boundary, boundary[1:] + boundary[:1], strict=True))
    return 2.0 * (abs(area) / 2.0 / math.pi) ** 0.5


def distance(p: list, q: list) -> float:
    """Pixel distance between two [x, y] points."""
    return math.hypot(q[0] - p[0], q[1] - p[1])


# execute() renders in parallel with multiprocessing; the __main__ guard keeps spawned workers from re-running this
# file on import (the default start method is spawn on macOS and Windows).
if __name__ == "__main__":
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

    # A built-in parameter variation: sweep the pupil diameter over a range in a number of steps. The eye holds its
    # gaze, so the apparent pupil grows in the image while its centre barely shifts.
    pupil_sweep = PupilSizeVariation(diameter_range=[2.5, 7.0], num_steps=6)
    pupil_data = data_gen.execute(pupil_sweep)

    pupil_ms = sweep_measurements(pupil_data)
    pupil_rows = [
        [
            f"{m['parameter_value']:.2f}",
            f"{apparent_diameter_px(m['pupil_boundary']):.1f}",
            f"({m['pupil_center'][0]:.1f}, {m['pupil_center'][1]:.1f})",
        ]
        for m in pupil_ms
    ]
    print(pupil_sweep.describe())
    print(tabulate(pupil_rows, headers=["pupil diameter (mm)", "apparent diameter (px)", "pupil centre (px)"]))
    grew = f"{apparent_diameter_px(pupil_ms[0]['pupil_boundary']):.1f} -> {apparent_diameter_px(pupil_ms[-1]['pupil_boundary']):.1f}"
    print(
        f"-> apparent pupil diameter grew {grew} px; "
        f"centre shifted {distance(pupil_ms[0]['pupil_center'], pupil_ms[-1]['pupil_center']):.1f} px\n"
    )

    # A custom parameter variation: GenericEyeVariation sweeps any eye parameter by its dotted path, here the conic
    # constant of the anterior cornea ("cornea.anterior_radius", "fovea_alpha_deg", etc. work the same way). The
    # pupil is imaged through the cornea, so reshaping the cornea moves both the reflected glint and the refracted
    # pupil. Subclass GenericEyeVariation for a named, reusable variation with its own describe().
    cornea_k_sweep = GenericEyeVariation("cornea.anterior_k", value_range=[-0.1, -0.5], num_steps=5)
    data_gen.set_experiment_name("cornea_k_sweep")
    k_data = data_gen.execute(cornea_k_sweep)

    k_ms = sweep_measurements(k_data)
    k_rows = [
        [
            f"{m['parameter_value']:.2f}",
            f"({m['corneal_reflections'][0][0]:.1f}, {m['corneal_reflections'][0][1]:.1f})",
            f"({m['pupil_center'][0]:.1f}, {m['pupil_center'][1]:.1f})",
        ]
        for m in k_ms
    ]
    print(cornea_k_sweep.describe())
    print(tabulate(k_rows, headers=["conic k", "glint (px)", "pupil centre (px)"]))
    print(
        f"-> glint moved {distance(k_ms[0]['corneal_reflections'][0], k_ms[-1]['corneal_reflections'][0]):.1f} px; "
        f"pupil centre shifted {distance(k_ms[0]['pupil_center'], k_ms[-1]['pupil_center']):.1f} px"
    )
