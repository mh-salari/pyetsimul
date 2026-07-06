"""Validate a calibrated tracker: how gaze accuracy holds across the screen and as the head moves.

A calibrated gaze model is not equally accurate everywhere: it is best near the calibration targets and degrades
toward the edges of the screen and when the head moves away from where it was calibrated. This calibrates a
tracker on an HV9 grid, then runs two systematic tests. The screen test fixes the head and sweeps the gaze
target over a dense grid on the screen, mapping the gaze error at each point. The observer test fixes the gaze
target and sweeps the eye/head position instead, mapping how the error grows as the head shifts. GazeAccuracyPlotter
draws each error map.
"""

import matplotlib.pyplot as plt

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.gaze_mapping.polynomial import PolynomialGazeModel
from pyetsimul.simulation import DataGenerationStrategy, EyePositionVariation, TargetPositionVariation
from pyetsimul.types import Position3D
from pyetsimul.visualization.gaze_accuracy_plots import GazeAccuracyPlotter

# A generic screen-based eye-tracking setup. Distances in millimetres; frame centred on the screen:
# +x right, +y from the screen toward the eye, +z up.
SCREEN_W, SCREEN_H = 380.0, 300.0  # the display the eye looks at
CAL_HW = 0.88 * SCREEN_W / 2  # the HV9 grid spans 88% x 83% of the screen
CAL_HH = 0.83 * SCREEN_H / 2

# HV9 calibration targets on the screen plane (y = 0): centre, 4 edge midpoints, 4 corners.
HV9 = [
    Position3D(0.0, 0.0, 0.0),
    Position3D(0.0, 0.0, CAL_HH),
    Position3D(0.0, 0.0, -CAL_HH),
    Position3D(-CAL_HW, 0.0, 0.0),
    Position3D(CAL_HW, 0.0, 0.0),
    Position3D(-CAL_HW, 0.0, CAL_HH),
    Position3D(CAL_HW, 0.0, CAL_HH),
    Position3D(-CAL_HW, 0.0, -CAL_HH),
    Position3D(CAL_HW, 0.0, -CAL_HH),
]

base = get_eye_model("PyEtSimul")
eye = Eye(model=base)
eye.position = Position3D(0.0, 700.0, 50.0)  # in front of the screen (+y), a little above its centre

camera = Camera()
camera.position = Position3D(0.0, 350.0, -150.0)  # between the screen and the eye, below it, looking up
camera.point_at(eye.position)  # aim the camera at the eye

light = Light(position=Position3D(70.0, 350.0, -140.0))  # its corneal reflection is the glint

# Calibrate a polynomial gaze model on the HV9 grid: run_calibration renders the eye at each target and fits it.
tracker = PolynomialGazeModel.create(
    cameras=[camera], lights=[light], calib_points=HV9, polynomial="cerrolaza_2008_symmetric"
)
tracker.run_calibration(eye)

# Screen test: head fixed, sweep the gaze target over a dense grid on the screen.
screen_test = TargetPositionVariation(
    grid_center=Position3D(0.0, 0.0, 0.0),
    dx=[-SCREEN_W / 2, SCREEN_W / 2],
    dy=[0.0, 0.0],
    dz=[-SCREEN_H / 2, SCREEN_H / 2],
    grid_size=[16, 1, 16],
)
# Observer test: gaze target fixed, sweep the eye/head position around where it was calibrated.
observer_test = EyePositionVariation(
    center=eye.position,
    dx=[-50.0, 50.0],
    dy=[-50.0, 50.0],
    dz=[0.0, 0.0],
    grid_size=[16, 16, 1],
)

plotter = GazeAccuracyPlotter()
fig, (ax_screen, ax_observer) = plt.subplots(1, 2, figsize=(18, 8))
for variation, name, ax in [(screen_test, "screen", ax_screen), (observer_test, "observer", ax_observer)]:
    # Render the whole variation into a dataset, then score the tracker's gaze error at every point.
    data_gen = DataGenerationStrategy(
        eyes=[eye],
        cameras=[camera],
        lights=[light],
        gaze_target=Position3D(0.0, 0.0, 0.0),
        experiment_name=f"{name}_test",
        save_to_file=False,
        use_refraction=tracker.use_refraction,
    )
    dataset = data_gen.execute(variation)
    results = evaluate_gaze_accuracy(eye_tracker=tracker, dataset=dataset, description=f"{name} test")
    results.pprint(f"{name} test accuracy")
    plotter.plot(results, tracker, f"{name} test", ax=ax)

fig.suptitle("Gaze accuracy: screen sweep and head movement", fontsize=13)
plt.tight_layout()
plt.show()
