"""Calibrate a gaze tracker end to end: fit it on an HV9 grid, test it, and explore it interactively.

A gaze model turns the pupil-glint vector the camera sees into a gaze point on the screen. This builds a
polynomial gaze model, calibrates it on a nine-point HV9 grid (run_calibration renders the eye at each
calibration target and fits the polynomial), reports the calibration accuracy, then generates a fresh grid of
test targets, measures the gaze error there, and opens an interactive plot: move the target with the arrow keys
and watch the predicted gaze track the true one. The calibration data, the test data, and the fitted model are
all produced in this one file.
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.gaze_mapping.polynomial import PolynomialGazeModel
from pyetsimul.simulation import DataGenerationStrategy, TargetPositionVariation
from pyetsimul.types import Position3D, ScreenGeometry
from pyetsimul.visualization.interactive_gaze_plot import create_interactive_gaze_plot

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

# Build a polynomial gaze model and calibrate it: run_calibration renders the eye at each HV9 target and fits the
# pupil-glint -> gaze polynomial. Other models share this interface (Stampe1993GazeModel, HomographyNormalizationGazeModel).
tracker = PolynomialGazeModel.create(
    cameras=[camera],
    lights=[light],
    calib_points=HV9,
    polynomial="cerrolaza_2008_symmetric",
)
tracker.run_calibration(eye)

# Accuracy at the calibration targets themselves.
accuracy_at_calibration_points(tracker, eye=eye).pprint("Calibration accuracy")

# Generate a fresh, denser grid of test targets and measure the gaze error there. save_to_file=False keeps the
# dataset in memory; evaluate_gaze_accuracy predicts gaze at each target and compares it to the true position.
test_grid = TargetPositionVariation(
    grid_center=Position3D(0.0, 0.0, 0.0),
    dx=[-SCREEN_W / 2, SCREEN_W / 2],
    dy=[0.0, 0.0],
    dz=[-SCREEN_H / 2, SCREEN_H / 2],
    grid_size=[11, 1, 9],
)
data_gen = DataGenerationStrategy(
    eyes=[eye],
    cameras=[camera],
    lights=[light],
    gaze_target=Position3D(0.0, 0.0, 0.0),
    experiment_name="test_grid",
    save_to_file=False,
    use_refraction=tracker.use_refraction,
)
test_data = data_gen.execute(test_grid)
evaluate_gaze_accuracy(eye_tracker=tracker, dataset=test_data, description="Test grid").pprint("Test-grid accuracy")

# Interactive check: move the target with the arrow keys and watch the predicted gaze follow the true one.
screen = ScreenGeometry(width=SCREEN_W, height=SCREEN_H, plane="xz")
create_interactive_gaze_plot(
    [eye],
    [tracker.estimate_gaze_at],
    HV9,
    tracker.plane_info,
    [camera],
    [light],
    eye_labels=["eye"],
    screen=screen,
)
