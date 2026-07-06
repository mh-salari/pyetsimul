"""Bring your own gaze model: register a custom polynomial and calibrate with it.

The polynomial gaze models are not a fixed set: you can define your own mapping from the pupil-glint vector to
gaze and register it by name. A PolynomialDescriptor lists the terms and their orders; register_polynomial makes
that mapping available to PolynomialGazeModel.create by name. This registers a custom third-order polynomial,
calibrates a tracker with it on an HV9 grid, and reports the calibration accuracy with an interactive per-target
plot. Change the terms and orders to try your own mapping.
"""

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.gaze_mapping.polynomial import PolynomialDescriptor, PolynomialGazeModel
from pyetsimul.gaze_mapping.polynomial.polynomials import register_polynomial
from pyetsimul.types import Position3D

# A custom third-order polynomial in the normalised pupil-glint vector (x, y). Each entry pairs a term with its
# orders: "x*y" with [2, 1] means x**2 * y, "x" with 3 means x**3, "1" with 0 is the constant.
MY_POLYNOMIAL = PolynomialDescriptor(
    name="my_cubic",
    description="third-order polynomial with cross-terms",
    terms=["x", "y", "x*y", "x*y", "x", "y", "x*y", "x", "y", "1"],
    orders=[3, 3, [2, 1], [1, 2], 2, 2, [1, 1], 1, 1, 0],
)
register_polynomial(MY_POLYNOMIAL)  # "my_cubic" can now be passed to PolynomialGazeModel.create

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

# Calibrate a gaze model that uses the custom polynomial, selected by its registered name.
tracker = PolynomialGazeModel.create(cameras=[camera], lights=[light], calib_points=HV9, polynomial="my_cubic")
tracker.run_calibration(eye)

# Accuracy at the calibration targets, printed and drawn as an interactive per-target plot.
results = accuracy_at_calibration_points(tracker, eye=eye)
results.pprint("Custom-polynomial calibration accuracy")
results.interactive_plot()
