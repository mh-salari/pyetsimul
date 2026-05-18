"""Stampe (1993) HV9 calibration on an EyeLink 1000 Plus desktop setup.

Geometry constants reproduce the lab layout. All distances in mm. Coordinate
system centred on the screen: x right, y away from the screen, z up.
"""

import matplotlib.pyplot as plt

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.gaze_mapping.stampe1993 import Stampe1993GazeModel
from pyetsimul.types import Position3D, ScreenGeometry
from pyetsimul.types.geometry import Point2D
from pyetsimul.visualization.coordinate_utils import prepare_eye_data_for_plots
from pyetsimul.visualization.setup_plots import plot_setup

# ---------------------------------------------------------------------------
# Physical setup (EyeLink 1000 Plus desktop mount)
# ---------------------------------------------------------------------------

# Screen
SCREEN_WIDTH = 531.36
SCREEN_HEIGHT = 298.98
SCREEN_HALF_W = SCREEN_WIDTH / 2
SCREEN_HALF_H = SCREEN_HEIGHT / 2

# Vertical reference: screen centre above the table surface (mm).
SCREEN_CENTRE_ABOVE_TABLE = 345.0

# Eyes (binocular, symmetric ±30 mm from screen centre, 835 mm perpendicular).
EYE_TO_SCREEN = 835.0
EYE_X_RIGHT = 30.0
EYE_X_LEFT = -30.0
EYE_Z = 430.0 - SCREEN_CENTRE_ABOVE_TABLE

# EyeLink 1000 Plus IR camera + illuminator (mounted on a bar below the screen).
CAMERA_X = -100.0
CAMERA_TO_SCREEN = 420.0
CAMERA_Z = 220.0 - SCREEN_CENTRE_ABOVE_TABLE

LIGHT_X = 95.0
LIGHT_TO_SCREEN = 435.0
LIGHT_Z = 230.0 - SCREEN_CENTRE_ABOVE_TABLE

# HV9 grid spans 88% (horizontal) × 83% (vertical) of the screen.
CAL_HALF_W = 0.88 * SCREEN_WIDTH / 2
CAL_HALF_H = 0.83 * SCREEN_HEIGHT / 2

# Order = HV9 paper order: centre + 4 cardinal edges, then 4 corners.
HV9_CALIBRATION_POINTS: list[Position3D] = [
    Position3D(0.0, 0.0, 0.0),
    Position3D(0.0, 0.0, CAL_HALF_H),
    Position3D(0.0, 0.0, -CAL_HALF_H),
    Position3D(-CAL_HALF_W, 0.0, 0.0),
    Position3D(CAL_HALF_W, 0.0, 0.0),
    Position3D(-CAL_HALF_W, 0.0, CAL_HALF_H),
    Position3D(CAL_HALF_W, 0.0, CAL_HALF_H),
    Position3D(-CAL_HALF_W, 0.0, -CAL_HALF_H),
    Position3D(CAL_HALF_W, 0.0, -CAL_HALF_H),
]


def main() -> None:
    """Calibrate one Stampe (1993) model per eye, then report per-target accuracy."""
    right_eye = Eye(cornea=ConicCornea())
    right_eye.position = Position3D(EYE_X_RIGHT, EYE_TO_SCREEN, EYE_Z)

    left_eye = Eye(cornea=ConicCornea())
    left_eye.position = Position3D(EYE_X_LEFT, EYE_TO_SCREEN, EYE_Z)

    camera = Camera()
    camera.position = Position3D(CAMERA_X, CAMERA_TO_SCREEN, CAMERA_Z)
    camera.point_at_binocular(left_eye.position, right_eye.position)

    light = Light(position=Position3D(LIGHT_X, LIGHT_TO_SCREEN, LIGHT_Z))

    et_right = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et_right.run_calibration(right_eye)

    et_left = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et_left.run_calibration(left_eye)

    accuracy_at_calibration_points(et_right, eye=right_eye).pprint("Right Eye — HV9 calibration accuracy")
    accuracy_at_calibration_points(et_left, eye=left_eye).pprint("Left Eye  — HV9 calibration accuracy")

    # 3D setup visualization.
    eyes = [right_eye, left_eye]
    look_at = Position3D(0.0, 0.0, 0.0)
    prepared = prepare_eye_data_for_plots(eyes, [look_at] * len(eyes), [light], [camera])
    calib_points_2d = [Point2D(*et_right.plane_info.extract_2d_coords(pt)) for pt in HV9_CALIBRATION_POINTS]
    screen = ScreenGeometry(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, plane="xz")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(1, 1, 1, projection="3d")
    plot_setup(
        ax,
        prepared["eyes_data"],
        [look_at] * len(eyes),
        [light],
        [camera],
        prepared["cr_3d_lists"],
        calib_points=calib_points_2d,
        screen=screen,
    )
    ax.legend(fontsize=7)
    ax.set_title("Stampe (1993) — HV9 calibration on EyeLink 1000 Plus setup", fontsize=12, fontweight="bold")
    plt.show()


if __name__ == "__main__":
    main()
