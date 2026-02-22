"""Stampe (1993) gaze model: calibration and gaze accuracy evaluation.

Uses the EyeLink 1000 Plus physical setup to calibrate a Stampe (1993) biquadratic
polynomial gaze model and evaluate its accuracy.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from pyetsimul.geometry.plane_detection import PlaneInfo
from eyelink1000plus_physical_setup import (
    CAMERA_TO_SCREEN,
    CAMERA_X,
    CAMERA_Z,
    EYE_TO_SCREEN,
    EYE_X_LEFT,
    EYE_X_RIGHT,
    EYE_Z,
    HV9_CALIBRATION_POINTS,
    LIGHT_X,
    LIGHT_Z,
    SCREEN_HALF_H,
    SCREEN_HALF_W,
    SCREEN_HEIGHT,
    SCREEN_WIDTH,
)

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.core.cornea import ConicCornea
from pyetsimul.evaluation import accuracy_at_calibration_points
from pyetsimul.evaluation.gaze_accuracy import evaluate_gaze_accuracy
from pyetsimul.gaze_mapping.stampe1993 import Stampe1993GazeModel
from pyetsimul.simulation import DataGenerationStrategy, EyePositionVariation, TargetPositionVariation
from pyetsimul.types import Position3D, ScreenGeometry
from pyetsimul.types.geometry import Point2D
from pyetsimul.visualization.coordinate_utils import prepare_eye_data_for_plots
from pyetsimul.visualization.gaze_accuracy_plots import GazeAccuracyPlotter
from pyetsimul.visualization.interactive_gaze_plot import create_interactive_gaze_plot
from pyetsimul.visualization.setup_plots import plot_setup


def plot_physical_setup(
    eyes: list[Eye],
    look_at: Position3D,
    camera: Camera,
    light: Light,
    plane_info: PlaneInfo,
    screen: ScreenGeometry,
) -> None:
    """Show 3D visualization of the physical eye tracking setup."""
    prepared = prepare_eye_data_for_plots(eyes, [look_at] * len(eyes), [light], [camera])
    calib_points_2d = [Point2D(*plane_info.extract_2d_coords(pt)) for pt in HV9_CALIBRATION_POINTS]

    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    plot_setup(
        ax_3d,
        prepared["eyes_data"],
        [look_at] * len(eyes),
        [light],
        [camera],
        prepared["cr_3d_lists"],
        calib_points=calib_points_2d,
        screen=screen,
    )
    ax_3d.legend(fontsize=7)
    ax_3d.set_title("EyeLink 1000 Plus — Physical Setup", fontsize=12, fontweight="bold")
    plt.show()
    plt.close(fig)


def main() -> None:
    """Calibrate Stampe (1993) model and evaluate gaze accuracy."""
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

    # IR light is mounted on the camera arm
    light = Light(position=Position3D(LIGHT_X, CAMERA_TO_SCREEN, LIGHT_Z))

    # --- Calibrate: one tracker per eye ---
    et_right = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et_right.run_calibration(right_eye)

    et_left = Stampe1993GazeModel.create([camera], [light], HV9_CALIBRATION_POINTS)
    et_left.run_calibration(left_eye)

    # --- Physical setup visualization ---
    eyes = [right_eye, left_eye]
    look_at = Position3D(0.0, 0.0, 0.0)
    prepared = prepare_eye_data_for_plots(eyes, [look_at] * len(eyes), [light], [camera])
    calib_points_2d = [Point2D(*et_right.plane_info.extract_2d_coords(pt)) for pt in HV9_CALIBRATION_POINTS]
    screen = ScreenGeometry(width=SCREEN_WIDTH, height=SCREEN_HEIGHT, plane="xz")

    fig = plt.figure(figsize=(10, 8))
    ax_3d = fig.add_subplot(1, 1, 1, projection="3d")
    plot_setup(
        ax_3d,
        prepared["eyes_data"],
        [look_at] * len(eyes),
        [light],
        [camera],
        prepared["cr_3d_lists"],
        calib_points=calib_points_2d,
        screen=screen,
    )
    ax_3d.legend(fontsize=7)
    ax_3d.set_title("EyeLink 1000 Plus — Physical Setup", fontsize=12, fontweight="bold")
    plt.show()
    plt.close(fig)

    # --- Calibration accuracy ---
    calib_right = accuracy_at_calibration_points(et_right, eye=right_eye)
    calib_right.pprint("Stampe 1993 — Right Eye Calibration")

    calib_left = accuracy_at_calibration_points(et_left, eye=left_eye)
    calib_left.pprint("Stampe 1993 — Left Eye Calibration")

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
            lights=[light],
            gaze_target=Position3D(0.0, 0.0, 0.0),
            experiment_name="screen_test",
            save_to_file=False,
            use_refraction=et.use_refraction,
        )
        dataset = data_gen.execute(screen_variation)
        results = evaluate_gaze_accuracy(eye_tracker=et, dataset=dataset, description=f"Screen test — {label}")
        results.pprint(f"Stampe 1993 — Screen Test ({label})")
        plotter.plot(results, et, f"Screen Test — {label}", ax=ax)

    fig_screen.suptitle("Stampe 1993 — Screen Test", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.close(fig_screen)

    # --- Head movement test: both eyes, side by side ---
    eye_position_variation = EyePositionVariation(
        center=right_eye.position,
        dx=[-50e-3, 50e-3],
        dy=[-50e-3, 50e-3],
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
            lights=[light],
            gaze_target=Position3D(0.0, 0.0, 0.0),
            experiment_name="observer_test",
            save_to_file=False,
            use_refraction=et.use_refraction,
        )
        dataset = data_gen.execute(eye_position_variation)
        results = evaluate_gaze_accuracy(eye_tracker=et, dataset=dataset, description=f"Head movement — {label}")
        results.pprint(f"Stampe 1993 — Head Movement Test ({label})")
        plotter.plot(results, et, f"Head Movement — {label}", ax=ax)

    fig_head.suptitle("Stampe 1993 — Head Movement Test", fontsize=14, fontweight="bold")
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
        [light],
        eye_labels=["Right", "Left"],
        screen=screen,
    )


if __name__ == "__main__":
    main()
