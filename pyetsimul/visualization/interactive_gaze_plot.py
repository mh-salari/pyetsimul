"""Interactive gaze plot for exploring calibration accuracy.

This module provides a standalone interactive plot that visualizes gaze estimation
accuracy at calibration points with a 3D setup view. It computes all calibration
errors internally from a predict function — no pre-computed arrays needed.
"""

import copy
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from pyetsimul.core import Eye

from ..geometry.conversions import calculate_angular_error_degrees
from ..geometry.plane_detection import PlaneInfo
from ..types import GazePrediction, Point2D, Point3D
from .coordinate_utils import prepare_eye_data_for_plots
from .interactive_controls import InteractiveControls
from .setup_plots import plot_setup


def _compute_calibration_errors(
    predict_fn: Callable[[Eye, Point3D], GazePrediction | None],
    eye: Eye,
    calibration_points: list[Point3D],
    plane_info: PlaneInfo,
) -> dict:
    """Compute calibration errors by predicting at each calibration point.

    Returns a dict with arrays needed for the 2D error plot.
    """
    n = len(calibration_points)
    x = np.zeros(n)
    y = np.zeros(n)
    u = np.zeros(n)
    v = np.zeros(n)
    errs_deg = np.zeros(n)
    predicted_points: list[Point3D] = []
    valid_mask = np.zeros(n, dtype=bool)

    for i, target in enumerate(calibration_points):
        coord1, coord2 = plane_info.extract_2d_coords(target)
        x[i] = coord1
        y[i] = coord2

        prediction = predict_fn(eye, target)

        if prediction is not None and prediction.gaze_point is not None:
            gp = prediction.gaze_point
            if not (np.isnan(gp.x) or np.isnan(gp.y) or np.isnan(gp.z)):
                pred_coord1, pred_coord2 = plane_info.extract_2d_coords(gp)
                u[i] = pred_coord1 - coord1
                v[i] = pred_coord2 - coord2
                target_point = Point3D(target.x, target.y, target.z)
                errs_deg[i] = calculate_angular_error_degrees(
                    target_point, Point3D(gp.x, gp.y, gp.z), eye.position
                )
                predicted_points.append(gp)
                valid_mask[i] = True
                continue

        # Failed prediction
        u[i] = np.nan
        v[i] = np.nan
        errs_deg[i] = np.nan
        predicted_points.append(Point3D(np.nan, np.nan, np.nan))

    return {
        "x": x,
        "y": y,
        "u": u,
        "v": v,
        "errs_deg": errs_deg,
        "predicted_points": predicted_points,
        "valid_mask": valid_mask,
    }


def create_interactive_gaze_plot(
    eye: Eye,
    predict_fn: Callable[[Eye, Point3D], GazePrediction | None],
    calibration_points: list[Point3D],
    plane_info: PlaneInfo,
    cameras: list,
    lights: list,
    use_legacy_look_at: bool = False,
) -> plt.Figure:
    """Create interactive gaze plot with keyboard controls.

    Computes calibration errors internally by calling predict_fn at each
    calibration point. Provides real-time exploration of gaze tracking accuracy
    with a 3D setup visualization alongside a 2D error vector plot.

    Args:
        eye: The Eye object to use for gaze prediction.
        predict_fn: Function that predicts gaze given (eye, target_point).
        calibration_points: List of 3D calibration target positions.
        plane_info: Plane detection info for coordinate mapping.
        cameras: List of Camera objects in the setup.
        lights: List of Light objects in the setup.
        use_legacy_look_at: Whether to use legacy look-at behavior.

    Returns:
        The matplotlib Figure containing the interactive gaze plot.

    """
    # Compute calibration errors once for the static overlay
    calib_data = _compute_calibration_errors(predict_fn, eye, calibration_points, plane_info)
    x = calib_data["x"]
    y = calib_data["y"]
    u = calib_data["u"]
    v = calib_data["v"]
    errs_deg = calib_data["errs_deg"]
    predicted_points = calib_data["predicted_points"]
    valid_mask = calib_data["valid_mask"]

    fig = plt.figure(figsize=(24, 10))
    interactive_eye = copy.deepcopy(eye)

    mean_x = sum(pt.x for pt in calibration_points) / len(calibration_points)
    mean_z = sum(pt.z for pt in calibration_points) / len(calibration_points)
    current_target = Point3D(mean_x, 0, mean_z)

    controls = InteractiveControls(interactive_eye, current_target, step_size=10e-3)

    def update_display() -> None:
        """Update both 3D and 2D plots with current target position."""
        fig.clear()

        # Left subplot: 3D eye tracking setup
        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")

        target_3d = Point3D(controls.target_point.x, 0, controls.target_point.z)

        # Convert calibration points to 2D list for plot overlay
        calib_points_2d: list[Point2D] = []
        for pt in calibration_points:
            calib_points_2d.append(Point2D(*plane_info.extract_2d_coords(pt)))

        # Prepare eye data
        prepared_data = prepare_eye_data_for_plots(
            [interactive_eye], [target_3d], lights, cameras, use_legacy_look_at
        )

        # Plot 3D setup
        plot_setup(
            ax_3d,
            prepared_data["eyes_data"],
            [target_3d],
            lights,
            cameras,
            prepared_data["cr_3d_lists"],
            calib_points=calib_points_2d,
        )

        ax_3d.scatter(
            [controls.target_point.x],
            [0],
            [controls.target_point.z],
            c="green",
            s=40,
            marker="x",
            label="Current Target",
        )
        ax_3d.legend()
        ax_3d.set_title("Eye Tracking Setup", fontsize=14, fontweight="bold", pad=20)

        # Right subplot: 2D calibration analysis with real-time prediction
        ax = fig.add_subplot(1, 2, 2)
        ax.set_facecolor("white")

        # Plot all calibration points
        ax.scatter(
            x * 1000,
            y * 1000,
            marker="+",
            s=30,
            c="blue",
            linewidths=1.5,
            alpha=0.8,
            label="Calibration Points",
            zorder=3,
        )

        # Plot original error vectors and predictions
        valid_indices = np.where(valid_mask)[0]
        x_valid, y_valid, u_valid, v_valid, pred_x, pred_y = [], [], [], [], [], []

        for i in valid_indices:
            pred_point = predicted_points[i]
            if isinstance(pred_point, Point3D) and not (
                np.isnan(pred_point.x) or np.isnan(pred_point.y) or np.isnan(pred_point.z)
            ):
                x_valid.append(x[i] * 1000)
                y_valid.append(y[i] * 1000)
                u_valid.append(u[i] * 1000)
                v_valid.append(v[i] * 1000)
                pred_coord1, pred_coord2 = plane_info.extract_2d_coords(
                    Point3D(pred_point.x, pred_point.y, pred_point.z)
                )
                pred_x.append(pred_coord1 * 1000)
                pred_y.append(pred_coord2 * 1000)

        if len(x_valid) > 0:
            for i in range(len(x_valid)):
                ax.arrow(
                    x_valid[i],
                    y_valid[i],
                    u_valid[i],
                    v_valid[i],
                    head_width=2,
                    head_length=1.5,
                    fc="gray",
                    ec="gray",
                    linewidth=1,
                    alpha=0.6,
                )

            ax.scatter(
                pred_x,
                pred_y,
                marker="o",
                s=20,
                c="red",
                alpha=0.7,
                label="Calibration Predictions",
                zorder=4,
            )

        current_prediction = predict_fn(interactive_eye, controls.target_point)

        if current_prediction is not None and current_prediction.gaze_point is not None:
            target_coord1, target_coord2 = plane_info.extract_2d_coords(controls.target_point)
            pred_pos = Point3D(
                current_prediction.gaze_point.x, current_prediction.gaze_point.y, current_prediction.gaze_point.z
            )
            pred_coord1, pred_coord2 = plane_info.extract_2d_coords(pred_pos)

            ax.scatter(
                [target_coord1 * 1000],
                [target_coord2 * 1000],
                marker="x",
                s=60,
                c="green",
                label="Current Target",
                zorder=5,
            )
            ax.scatter(
                [pred_coord1 * 1000],
                [pred_coord2 * 1000],
                marker="x",
                s=40,
                c="orange",
                label="Current Prediction",
                zorder=5,
            )

            error_x = (pred_coord1 - target_coord1) * 1000
            error_y = (pred_coord2 - target_coord2) * 1000
            ax.arrow(
                target_coord1 * 1000,
                target_coord2 * 1000,
                error_x,
                error_y,
                head_width=2,
                head_length=1.5,
                fc="black",
                ec="black",
                linewidth=1,
                alpha=0.6,
                linestyle="--",
            )

            current_error_mm = np.sqrt(error_x**2 + error_y**2)
            current_error_deg = calculate_angular_error_degrees(
                controls.target_point, pred_pos, interactive_eye.position
            )

            valid_errors = errs_deg[valid_mask]
            if len(valid_errors) > 0:
                calib_errors_text = f"Avg: {np.mean(valid_errors):.3f}° | Max: {np.max(valid_errors):.3f}°"
            else:
                calib_errors_text = "No valid calibration points"

            ax.set_title(
                f"Calibration Analysis\n"
                f"Current gaze error: {current_error_mm:.2f}mm ({current_error_deg:.4f}°)\n"
                f"Calibration errors: {calib_errors_text}",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )
        else:
            target_coord1, target_coord2 = plane_info.extract_2d_coords(controls.target_point)
            ax.scatter(
                [target_coord1 * 1000],
                [target_coord2 * 1000],
                marker="x",
                s=60,
                c="green",
                label="Current Target",
                zorder=5,
            )

            valid_errors = errs_deg[valid_mask]
            if len(valid_errors) > 0:
                calib_errors_text = f"Avg: {np.mean(valid_errors):.3f}° | Max: {np.max(valid_errors):.3f}°"
            else:
                calib_errors_text = "No valid calibration points"

            ax.set_title(
                f"Calibration Analysis\n"
                f"Current gaze error: PREDICTION FAILED\n"
                f"Calibration errors: {calib_errors_text}",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )

        ax.set_xlabel(f"{plane_info.primary_axis.upper()} Position (mm)")
        ax.set_ylabel(f"{plane_info.secondary_axis.upper()} Position (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.3)
        fig.canvas.draw()

    controls.set_update_callback(update_display)
    fig.canvas.mpl_connect("key_press_event", controls.handle_key_press)

    update_display()

    return fig
