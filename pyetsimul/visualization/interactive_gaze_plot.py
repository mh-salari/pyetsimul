"""Interactive gaze plot for exploring calibration accuracy.

This module provides a standalone interactive plot that visualizes gaze estimation
accuracy at calibration points with a 3D setup view. It computes all calibration
errors internally from predict functions — no pre-computed arrays needed.

Supports multiple eyes: each eye gets its own predict function, color-coded arrows
and predictions on the 2D panel, and all eyes appear in the 3D setup view.
"""

import copy
from collections.abc import Callable

import matplotlib.pyplot as plt
import numpy as np

from pyetsimul.core import Eye

from ..geometry.conversions import calculate_angular_error_degrees
from ..geometry.plane_detection import PlaneInfo
from ..types import GazePrediction, Point2D, Point3D, ScreenGeometry
from .coordinate_utils import prepare_eye_data_for_plots
from .interactive_controls import InteractiveControls
from .plot_config import create_plot_config
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
                errs_deg[i] = calculate_angular_error_degrees(target_point, Point3D(gp.x, gp.y, gp.z), eye.position)
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
    eyes: list[Eye],
    predict_fns: list[Callable[[Eye, Point3D], GazePrediction | None]],
    calibration_points: list[Point3D],
    plane_info: PlaneInfo,
    cameras: list,
    lights: list,
    use_legacy_look_at: bool = False,
    eye_labels: list[str] | None = None,
    eye_colors: list[str] | None = None,
    screen: ScreenGeometry | None = None,
    show: bool = True,
) -> plt.Figure:
    """Create interactive gaze plot with keyboard controls.

    Computes calibration errors internally by calling each predict_fn at each
    calibration point. Provides real-time exploration of gaze tracking accuracy
    with a 3D setup visualization alongside a 2D error vector plot.

    Supports multiple eyes: each eye/predict_fn pair is color-coded on the plot.

    Args:
        eyes: List of Eye objects to use for gaze prediction.
        predict_fns: List of functions that predict gaze given (eye, target_point).
        calibration_points: List of 3D calibration target positions.
        plane_info: Plane detection info for coordinate mapping.
        cameras: List of Camera objects in the setup.
        lights: List of Light objects in the setup.
        use_legacy_look_at: Whether to use legacy look-at behavior.
        eye_labels: Optional labels for each eye (e.g. ["Right", "Left"]).
            Defaults to "Eye 1", "Eye 2", etc.
        eye_colors: Optional colors for each eye (e.g. ["blue", "green"]).
            Defaults to the config eye color palette.
        screen: Optional ScreenGeometry to draw screen border on the 3D plot.
        show: If True (default), print controls and display with plt.show().
              If False, close the figure from matplotlib's manager and return it
              for saving with fig.savefig().

    Returns:
        The matplotlib Figure.

    """
    config = create_plot_config()
    if eye_colors is None:
        eye_colors = config.colors.eyes
    target_color = config.colors.target
    n_eyes = len(eyes)

    if eye_labels is None:
        eye_labels = [f"Eye {i + 1}" for i in range(n_eyes)]

    # Compute calibration errors once per eye for the static overlay
    all_calib_data = [
        _compute_calibration_errors(predict_fns[i], eyes[i], calibration_points, plane_info) for i in range(n_eyes)
    ]

    fig = plt.figure(figsize=(24, 10))
    interactive_eyes = [copy.deepcopy(eye) for eye in eyes]

    mean_x = sum(pt.x for pt in calibration_points) / len(calibration_points)
    mean_z = sum(pt.z for pt in calibration_points) / len(calibration_points)
    current_target = Point3D(mean_x, 0, mean_z)

    controls = InteractiveControls(interactive_eyes, current_target, step_size=10e-3)

    def update_display() -> None:
        """Update both 3D and 2D plots with current target position."""
        fig.clear()

        # Left subplot: 3D eye tracking setup
        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")

        target_3d = Point3D(controls.target_point.x, 0, controls.target_point.z)

        calib_points_2d: list[Point2D] = []
        for pt in calibration_points:
            calib_points_2d.append(Point2D(*plane_info.extract_2d_coords(pt)))

        prepared_data = prepare_eye_data_for_plots(
            interactive_eyes, [target_3d] * n_eyes, lights, cameras, use_legacy_look_at
        )

        plot_setup(
            ax_3d,
            prepared_data["eyes_data"],
            [target_3d] * n_eyes,
            lights,
            cameras,
            prepared_data["cr_3d_lists"],
            calib_points=calib_points_2d,
            screen=screen,
        )

        ax_3d.scatter(
            [controls.target_point.x],
            [0],
            [controls.target_point.z],
            c=target_color,
            s=40,
            marker="+",
            label="Target",
        )
        ax_3d.legend()
        ax_3d.set_title("Eye Tracking Setup", fontsize=14, fontweight="bold", pad=20)

        # Right subplot: 2D calibration analysis
        ax = fig.add_subplot(1, 2, 2)
        ax.set_facecolor("white")

        # Calibration target points (shared across all eyes)
        calib_x = all_calib_data[0]["x"]
        calib_y = all_calib_data[0]["y"]
        ax.scatter(
            calib_x * 1000,
            calib_y * 1000,
            marker="x",
            s=40,
            c="dimgray",
            linewidths=1.5,
            alpha=0.8,
            label="Calibration Points",
            zorder=3,
        )

        # Static calibration error overlay per eye
        for eye_idx in range(n_eyes):
            calib_data = all_calib_data[eye_idx]
            color = eye_colors[eye_idx % len(eye_colors)]
            label = eye_labels[eye_idx]
            x = calib_data["x"]
            y = calib_data["y"]
            u = calib_data["u"]
            v = calib_data["v"]
            predicted_points = calib_data["predicted_points"]
            valid_mask = calib_data["valid_mask"]

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
                        fc=color,
                        ec=color,
                        linewidth=1,
                        alpha=0.4,
                    )

                ax.scatter(
                    pred_x,
                    pred_y,
                    marker="o",
                    s=12,
                    c=color,
                    alpha=0.5,
                    label=f"{label} Predictions",
                    zorder=4,
                )

        # Real-time prediction per eye
        current_errors_mm = []
        current_errors_deg = []
        for eye_idx in range(n_eyes):
            color = eye_colors[eye_idx % len(eye_colors)]
            prediction = predict_fns[eye_idx](interactive_eyes[eye_idx], controls.target_point)

            if prediction is not None and prediction.gaze_point is not None:
                gp = prediction.gaze_point
                if not (np.isnan(gp.x) or np.isnan(gp.y) or np.isnan(gp.z)):
                    pred_pos = Point3D(gp.x, gp.y, gp.z)
                    target_coord1, target_coord2 = plane_info.extract_2d_coords(controls.target_point)
                    pred_coord1, pred_coord2 = plane_info.extract_2d_coords(pred_pos)

                    ax.scatter(
                        [pred_coord1 * 1000],
                        [pred_coord2 * 1000],
                        marker="x",
                        s=40,
                        c=color,
                        label=f"{eye_labels[eye_idx]} Current",
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
                        fc=color,
                        ec=color,
                        linewidth=1.5,
                        alpha=0.8,
                        linestyle="--",
                    )

                    current_errors_mm.append(np.sqrt(error_x**2 + error_y**2))
                    current_errors_deg.append(
                        calculate_angular_error_degrees(
                            controls.target_point, pred_pos, interactive_eyes[eye_idx].position
                        )
                    )

        # Target marker (shared)
        target_coord1, target_coord2 = plane_info.extract_2d_coords(controls.target_point)
        ax.scatter(
            [target_coord1 * 1000],
            [target_coord2 * 1000],
            marker="+",
            s=60,
            c=target_color,
            label="Target",
            zorder=6,
        )

        # Title with error summary
        if len(current_errors_mm) > 0:
            avg_mm = np.mean(current_errors_mm)
            avg_deg = np.mean(current_errors_deg)
            current_text = f"Current gaze error: {avg_mm:.2f}mm ({avg_deg:.4f}°)"
            if n_eyes > 1:
                current_text += f" avg across {n_eyes} eyes"
        else:
            current_text = "Current gaze error: PREDICTION FAILED"

        # Calibration error summary across all eyes
        all_valid_errors = []
        for eye_idx in range(n_eyes):
            calib_data = all_calib_data[eye_idx]
            valid_errors = calib_data["errs_deg"][calib_data["valid_mask"]]
            all_valid_errors.extend(valid_errors)

        if len(all_valid_errors) > 0:
            all_valid_errors = np.array(all_valid_errors)
            calib_text = (
                f"Calibration errors — Avg: {np.mean(all_valid_errors):.3f}° | Max: {np.max(all_valid_errors):.3f}°"
            )
        else:
            calib_text = "No valid calibration points"

        ax.set_title(
            f"Calibration Analysis\n{current_text}\n{calib_text}",
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

    if show:
        InteractiveControls.print_controls(additional_controls={"Exit": "ESC"})
        plt.show()

    plt.close(fig)

    return fig
