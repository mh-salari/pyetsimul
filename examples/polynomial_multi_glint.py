"""Polynomial gaze model: calibration and gaze accuracy evaluation.

Uses the EyeLink 1000 Plus physical setup to calibrate a polynomial gaze model
and evaluate its accuracy. Demonstrates multi-glint support by using two light
sources, where each P-CR vector contributes polynomial features that are
concatenated before regression.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pyetsimul.geometry.plane_detection import PlaneInfo

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.gaze_mapping.polynomial import PolynomialGazeModel
from pyetsimul.simulation import DataGenerationStrategy, EyePositionVariation, TargetPositionVariation
from pyetsimul.types import Position3D, ScreenGeometry
from pyetsimul.types.geometry import Point2D
from pyetsimul.visualization.coordinate_utils import prepare_eye_data_for_plots
from pyetsimul.visualization.gaze_accuracy_plots import GazeAccuracyPlotter
from pyetsimul.visualization.interactive_gaze_plot import create_interactive_gaze_plot
from pyetsimul.visualization.setup_plots import plot_setup

# ---------------------------------------------------------------------------
# EyeLink 1000 Plus physical setup dimensions
# ---------------------------------------------------------------------------
# All distances in mm. Coordinate system centered on the screen:
#   x — horizontal (positive = right)
#   y — depth from screen (positive = away from screen toward the eye)
#   z — vertical (positive = up)
#
# Layout (side view, not to scale):
#   Screen (y=0)  <-475mm->  Camera  <-505mm->  Eye (y=980mm)
#                              (below screen center)

# Screen dimensions
SCREEN_WIDTH = 376
SCREEN_HEIGHT = 301
SCREEN_HALF_W = SCREEN_WIDTH / 2
SCREEN_HALF_H = SCREEN_HEIGHT / 2

# EyeLink HV9 calibration area (88% x 83% of screen)
CAL_AREA_X = 0.88
CAL_AREA_Y = 0.83
CAL_HALF_W = SCREEN_WIDTH * CAL_AREA_X / 2
CAL_HALF_H = SCREEN_HEIGHT * CAL_AREA_Y / 2

# Camera and eye distances from screen plane
EYE_TO_SCREEN = 980
CAMERA_TO_SCREEN = 475

# Real-world heights from ground
SCREEN_BOTTOM_FROM_GROUND = 120
CAMERA_FROM_GROUND = 150
LIGHT_FROM_GROUND = 150
EYE_FROM_GROUND = 420

# Horizontal offsets from screen center
CAMERA_X = -180
LIGHT1_X = CAMERA_X + 265  # IR light 1: 265 mm to the right of camera
LIGHT2_X = CAMERA_X - 50  # IR light 2: 50 mm to the left of camera
EYE_X_RIGHT = 30
EYE_X_LEFT = -30

# Vertical offsets derived from ground heights (screen center is z=0 reference)
SCREEN_CENTER_FROM_GROUND = SCREEN_BOTTOM_FROM_GROUND + SCREEN_HALF_H
CAMERA_Z = CAMERA_FROM_GROUND - SCREEN_CENTER_FROM_GROUND
LIGHT_Z = LIGHT_FROM_GROUND - SCREEN_CENTER_FROM_GROUND
EYE_Z = EYE_FROM_GROUND - SCREEN_CENTER_FROM_GROUND

# HV9 calibration grid: 9 targets on a cross + corners pattern
HV9_CALIBRATION_POINTS: list[Position3D] = [
    Position3D(0.0, 0.0, 0.0),  # Center
    Position3D(0.0, 0.0, CAL_HALF_H),  # Top center
    Position3D(0.0, 0.0, -CAL_HALF_H),  # Bottom center
    Position3D(-CAL_HALF_W, 0.0, 0.0),  # Left center
    Position3D(CAL_HALF_W, 0.0, 0.0),  # Right center
    Position3D(-CAL_HALF_W, 0.0, CAL_HALF_H),  # Top-left
    Position3D(CAL_HALF_W, 0.0, CAL_HALF_H),  # Top-right
    Position3D(-CAL_HALF_W, 0.0, -CAL_HALF_H),  # Bottom-left
    Position3D(CAL_HALF_W, 0.0, -CAL_HALF_H),  # Bottom-right
]

# Polynomial to use for gaze mapping
POLYNOMIAL = "hennessey_2008"


# ---------------------------------------------------------------------------
# Visualization helper
# ---------------------------------------------------------------------------
def plot_physical_setup(
    eyes: list[Eye],
    look_at: Position3D,
    camera: Camera,
    lights: list[Light],
    plane_info: PlaneInfo,
    screen: ScreenGeometry,
) -> None:
    """Show 3D visualization of the physical eye tracking setup."""
    prepared = prepare_eye_data_for_plots(eyes, [look_at] * len(eyes), lights, [camera])
    calib_points_2d = [Point2D(*plane_info.extract_2d_coords(pt)) for pt in HV9_CALIBRATION_POINTS]

    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    plot_setup(
        ax_3d,
        prepared["eyes_data"],
        [look_at] * len(eyes),
        lights,
        [camera],
        prepared["cr_3d_lists"],
        calib_points=calib_points_2d,
        screen=screen,
    )
    ax_3d.legend(fontsize=7)
    title = f"Polynomial Gaze Model — {len(lights)} Light(s)"
    ax_3d.set_title(title, fontsize=12, fontweight="bold")
    plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    """Calibrate polynomial gaze model and evaluate accuracy."""
    # --- Setup: both eyes ---
    right_eye = Eye(cornea=ConicCornea())
    right_eye.position = Position3D(EYE_X_RIGHT, EYE_TO_SCREEN, EYE_Z)

    left_eye = Eye(cornea=ConicCornea())
    left_eye.position = Position3D(EYE_X_LEFT, EYE_TO_SCREEN, EYE_Z)

    # Camera points at the midpoint between the two eyes
    camera = Camera()
    camera.position = Position3D(CAMERA_X, CAMERA_TO_SCREEN, CAMERA_Z)
    midpoint = Position3D(
        (right_eye.position.x + left_eye.position.x) / 2,
        right_eye.position.y,
        right_eye.position.z,
    )
    camera.point_at(midpoint)

    # Two IR lights for multi-glint polynomial regression
    light1 = Light(position=Position3D(LIGHT1_X, CAMERA_TO_SCREEN, LIGHT_Z))
    light2 = Light(position=Position3D(LIGHT2_X, CAMERA_TO_SCREEN, LIGHT_Z))
    lights = [light1, light2]

    screen = ScreenGeometry(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, plane="xz")

    # --- Calibrate: one tracker per eye ---
    et_right = PolynomialGazeModel.create(
        cameras=[camera],
        lights=lights,
        calib_points=HV9_CALIBRATION_POINTS,
        polynomial=POLYNOMIAL,
    )
    et_right.run_calibration(right_eye)

    et_left = PolynomialGazeModel.create(
        cameras=[camera],
        lights=lights,
        calib_points=HV9_CALIBRATION_POINTS,
        polynomial=POLYNOMIAL,
    )
    et_left.run_calibration(left_eye)

    # --- Physical setup visualization ---
    eyes = [right_eye, left_eye]
    look_at = Position3D(0.0, 0.0, 0.0)
    plot_physical_setup(eyes, look_at, camera, lights, et_right.plane_info, screen)

    # --- Calibration accuracy ---
    calib_right = accuracy_at_calibration_points(et_right, eye=right_eye)
    calib_right.pprint(f"Polynomial ({POLYNOMIAL}) — Right Eye Calibration")

    calib_left = accuracy_at_calibration_points(et_left, eye=left_eye)
    calib_left.pprint(f"Polynomial ({POLYNOMIAL}) — Left Eye Calibration")

    # --- Screen test: both eyes, side by side ---
    screen_variation = TargetPositionVariation(
        grid_center=Position3D(0.0, 0.0, 0.0),
        dx=[-SCREEN_HALF_W, SCREEN_HALF_W],
        dy=[0.0, 0.0],
        dz=[-SCREEN_HALF_H, SCREEN_HALF_H],
        grid_size=[16, 1, 16],
    )

    plotter = GazeAccuracyPlotter()
    fig_screen, (ax_screen_r, ax_screen_l) = plt.subplots(1, 2, figsize=(20, 8))

    for eye, et, ax, label in [
        (right_eye, et_right, ax_screen_r, "Right Eye"),
        (left_eye, et_left, ax_screen_l, "Left Eye"),
    ]:
        data_gen = DataGenerationStrategy(
            eyes=[eye],
            cameras=[camera],
            lights=lights,
            gaze_target=Position3D(0.0, 0.0, 0.0),
            experiment_name="screen_test",
            save_to_file=False,
            use_refraction=et.use_refraction,
        )
        dataset = data_gen.execute(screen_variation)
        results = evaluate_gaze_accuracy(eye_tracker=et, dataset=dataset, description=f"Screen test — {label}")
        results.pprint(f"Polynomial ({POLYNOMIAL}) — Screen Test ({label})")
        plotter.plot(results, et, f"Screen Test — {label}", ax=ax)

    fig_screen.suptitle(f"Polynomial ({POLYNOMIAL}) — Screen Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close(fig_screen)

    # --- Head movement test: both eyes, side by side ---
    eye_position_variation = EyePositionVariation(
        center=right_eye.position,
        dx=[-50, 50],
        dy=[-50, 50],
        dz=[0.0, 0.0],
        grid_size=[16, 16, 1],
    )

    fig_head, (ax_head_r, ax_head_l) = plt.subplots(1, 2, figsize=(20, 8))

    for eye, et, ax, label in [
        (right_eye, et_right, ax_head_r, "Right Eye"),
        (left_eye, et_left, ax_head_l, "Left Eye"),
    ]:
        data_gen = DataGenerationStrategy(
            eyes=[eye],
            cameras=[camera],
            lights=lights,
            gaze_target=Position3D(0.0, 0.0, 0.0),
            experiment_name="observer_test",
            save_to_file=False,
            use_refraction=et.use_refraction,
        )
        dataset = data_gen.execute(eye_position_variation)
        results = evaluate_gaze_accuracy(eye_tracker=et, dataset=dataset, description=f"Head movement — {label}")
        results.pprint(f"Polynomial ({POLYNOMIAL}) — Head Movement Test ({label})")
        plotter.plot(results, et, f"Head Movement — {label}", ax=ax)

    fig_head.suptitle(f"Polynomial ({POLYNOMIAL}) — Head Movement Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close(fig_head)

    # --- Combined interactive plot: both eyes ---
    create_interactive_gaze_plot(
        [right_eye, left_eye],
        [et_right.estimate_gaze_at, et_left.estimate_gaze_at],
        HV9_CALIBRATION_POINTS,
        et_right.plane_info,
        [camera],
        lights,
        eye_labels=["Right", "Left"],
        screen=screen,
    )


if __name__ == "__main__":
    main()
