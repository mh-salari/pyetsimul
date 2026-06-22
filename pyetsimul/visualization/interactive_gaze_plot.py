"""Interactive gaze plot for exploring calibration accuracy.

This module provides a standalone interactive plot that visualizes gaze estimation
accuracy at calibration points with a 3D setup view. It computes all calibration
errors internally from predict functions — no pre-computed arrays needed.

Supports multiple eyes: each eye gets its own predict function, color-coded arrows
and predictions on the 2D panel, and all eyes appear in the 3D setup view.
"""

import copy
from collections.abc import Callable
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from pyetsimul.core import Eye

from ..geometry.conversions import calculate_angular_error_degrees
from ..geometry.plane_detection import PlaneInfo
from ..types import GazePrediction, Point2D, Point3D, ScreenGeometry
from .coordinate_utils import prepare_eye_data_for_plots
from .interactive_controls import InteractiveControls
from .plot_config import create_plot_config
from .setup_plots import plot_setup


def compute_calibration_errors(
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


def render_calibration_view(
    ax_3d: "Axes",
    ax_2d: "Axes",
    target_point: Point3D,
    eyes: list[Eye],
    predict_fns: list[Callable[[Eye, Point3D], GazePrediction | None]],
    calibration_points: list[Point3D],
    plane_info: PlaneInfo,
    cameras: list,
    lights: list,
    *,
    cached_calib_data: list[dict] | None = None,
    eye_labels: list[str] | None = None,
    eye_colors: list[str] | None = None,
    target_color: str | None = None,
    screen: ScreenGeometry | None = None,
    ref_bounds_3d: dict | None = None,
    xlim_2d: tuple[float, float] | None = None,
    ylim_2d: tuple[float, float] | None = None,
) -> dict:
    """Render the 3D setup + 2D calibration accuracy view for a given target.

    Both axes are cleared and re-plotted. Designed to be called per-frame from an
    animation, and also used as the rendering primitive of
    ``create_interactive_gaze_plot``. Eye orientation is handled internally via
    ``prepare_eye_data_for_plots`` / ``predict_fns``; the caller passes the eye
    objects but does not need to call ``eye.look_at`` explicitly.

    Args:
        ax_3d: 3D matplotlib axes (projection='3d').
        ax_2d: 2D matplotlib axes.
        target_point: Current gaze target.
        eyes: List of Eye objects.
        predict_fns: One predict function per eye, ``(eye, target) -> GazePrediction``.
        calibration_points: 3D calibration target positions.
        plane_info: Plane detection info for coordinate mapping.
        cameras: List of cameras.
        lights: List of lights.
        cached_calib_data: Optional pre-computed per-eye calibration data (from
            ``compute_calibration_errors``). Skips recomputation when provided.
        eye_labels: Optional per-eye labels (defaults to "Eye 1", "Eye 2", ...).
        eye_colors: Optional per-eye colors (defaults to plot config palette).
        target_color: Optional target marker color (defaults to plot config).
        screen: Optional ScreenGeometry to draw screen border on the 3D plot.
        ref_bounds_3d: Optional fixed 3D axis bounds, passed through to ``plot_setup``.
        xlim_2d: Optional fixed x-limits for the 2D axes.
        ylim_2d: Optional fixed y-limits for the 2D axes.

    Returns:
        Dict with 'errors_mm' and 'errors_deg' lists, one entry per eye that
        produced a valid prediction at ``target_point``.

    """
    config = create_plot_config()
    if eye_colors is None:
        eye_colors = config.colors.eyes
    if target_color is None:
        target_color = config.colors.target
    n_eyes = len(eyes)

    if eye_labels is None:
        eye_labels = [f"Eye {i + 1}" for i in range(n_eyes)]

    # Normalise to Point3D so downstream geometry calls (which strictly typecheck)
    # accept callers passing Position3D — both share .x/.y/.z attributes.
    if not isinstance(target_point, Point3D):
        target_point = Point3D(target_point.x, target_point.y, target_point.z)

    if cached_calib_data is None:
        cached_calib_data = [
            compute_calibration_errors(predict_fns[i], eyes[i], calibration_points, plane_info) for i in range(n_eyes)
        ]

    ax_3d.clear()
    ax_2d.clear()

    # ---------------------------------------------------------------------
    # Left subplot: 3D eye tracking setup
    # ---------------------------------------------------------------------
    target_3d = Point3D(target_point.x, 0, target_point.z)

    calib_points_2d: list[Point2D] = []
    for pt in calibration_points:
        calib_points_2d.append(Point2D(*plane_info.extract_2d_coords(pt)))

    prepared_data = prepare_eye_data_for_plots(eyes, [target_3d] * n_eyes, lights, cameras)

    plot_setup(
        ax_3d,
        prepared_data["eyes_data"],
        [target_3d] * n_eyes,
        lights,
        cameras,
        prepared_data["cr_3d_lists"],
        calib_points=calib_points_2d,
        screen=screen,
        ref_bounds=ref_bounds_3d,
    )

    ax_3d.scatter(
        [target_point.x],
        [0],
        [target_point.z],
        c=target_color,
        s=40,
        marker="+",
        label="Target",
    )
    ax_3d.legend()
    ax_3d.set_title("Eye Tracking Setup", fontsize=14, fontweight="bold", pad=20)

    # ---------------------------------------------------------------------
    # Right subplot: 2D calibration analysis
    # ---------------------------------------------------------------------
    ax_2d.set_facecolor("white")

    # Calibration target points (shared across all eyes)
    calib_x = cached_calib_data[0]["x"]
    calib_y = cached_calib_data[0]["y"]
    ax_2d.scatter(
        calib_x,
        calib_y,
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
        calib_data = cached_calib_data[eye_idx]
        color = eye_colors[eye_idx % len(eye_colors)]
        label = eye_labels[eye_idx]
        x = calib_data["x"]
        y = calib_data["y"]
        predicted_points = calib_data["predicted_points"]
        valid_mask = calib_data["valid_mask"]

        valid_indices = np.where(valid_mask)[0]
        x_valid, y_valid, pred_x, pred_y = [], [], [], []

        for i in valid_indices:
            pred_point = predicted_points[i]
            if isinstance(pred_point, Point3D) and not (
                np.isnan(pred_point.x) or np.isnan(pred_point.y) or np.isnan(pred_point.z)
            ):
                x_valid.append(x[i])
                y_valid.append(y[i])
                pred_coord1, pred_coord2 = plane_info.extract_2d_coords(
                    Point3D(pred_point.x, pred_point.y, pred_point.z)
                )
                pred_x.append(pred_coord1)
                pred_y.append(pred_coord2)

        if len(x_valid) > 0:
            for i in range(len(x_valid)):
                ax_2d.plot(
                    [x_valid[i], pred_x[i]],
                    [y_valid[i], pred_y[i]],
                    color=color,
                    linewidth=1,
                    alpha=0.4,
                    linestyle="--",
                )

            ax_2d.scatter(
                pred_x,
                pred_y,
                marker="o",
                s=12,
                c=color,
                alpha=0.5,
                label=f"{label} Predictions",
                zorder=4,
            )

    # Real-time prediction per eye at the current target
    current_errors_mm: list[float] = []
    current_errors_deg: list[float] = []
    for eye_idx in range(n_eyes):
        color = eye_colors[eye_idx % len(eye_colors)]
        prediction = predict_fns[eye_idx](eyes[eye_idx], target_point)

        if prediction is not None and prediction.gaze_point is not None:
            gp = prediction.gaze_point
            if not (np.isnan(gp.x) or np.isnan(gp.y) or np.isnan(gp.z)):
                pred_pos = Point3D(gp.x, gp.y, gp.z)
                target_coord1, target_coord2 = plane_info.extract_2d_coords(target_point)
                pred_coord1, pred_coord2 = plane_info.extract_2d_coords(pred_pos)

                ax_2d.scatter(
                    [pred_coord1],
                    [pred_coord2],
                    marker="x",
                    s=40,
                    c=color,
                    label=f"{eye_labels[eye_idx]} Current",
                    zorder=5,
                )

                error_x = pred_coord1 - target_coord1
                error_y = pred_coord2 - target_coord2
                ax_2d.plot(
                    [target_coord1, pred_coord1],
                    [target_coord2, pred_coord2],
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    linestyle="--",
                )

                current_errors_mm.append(float(np.sqrt(error_x**2 + error_y**2)))
                current_errors_deg.append(
                    calculate_angular_error_degrees(target_point, pred_pos, eyes[eye_idx].position)
                )

    # Target marker (shared)
    target_coord1, target_coord2 = plane_info.extract_2d_coords(target_point)
    ax_2d.scatter(
        [target_coord1],
        [target_coord2],
        marker="+",
        s=60,
        c=target_color,
        label="Target",
        zorder=6,
    )

    # Title with error summary
    if len(current_errors_mm) > 0:
        avg_mm = float(np.mean(current_errors_mm))
        avg_deg = float(np.mean(current_errors_deg))
        current_text = f"Current gaze error: {avg_deg:.4f}° ({avg_mm:.2f}mm)"
        if n_eyes > 1:
            current_text += f" avg across {n_eyes} eyes"
    else:
        current_text = "Current gaze error: PREDICTION FAILED"

    # Calibration error summary across all eyes
    all_valid_errors: list[float] = []
    for eye_idx in range(n_eyes):
        calib_data = cached_calib_data[eye_idx]
        valid_errors = calib_data["errs_deg"][calib_data["valid_mask"]]
        all_valid_errors.extend(valid_errors)

    if len(all_valid_errors) > 0:
        valid_errs_array = np.array(all_valid_errors)
        calib_text = (
            f"Calibration errors — Avg: {np.mean(valid_errs_array):.3f}° | Max: {np.max(valid_errs_array):.3f}°"
        )
    else:
        calib_text = "No valid calibration points"

    ax_2d.set_title(
        f"Calibration Analysis\n{current_text}\n{calib_text}",
        fontsize=12,
        fontweight="bold",
        pad=20,
    )

    ax_2d.set_xlabel(f"{plane_info.primary_axis.upper()} Position (mm)")
    ax_2d.set_ylabel(f"{plane_info.secondary_axis.upper()} Position (mm)")
    ax_2d.grid(True, alpha=0.3)
    ax_2d.legend()
    ax_2d.set_aspect("equal")
    if xlim_2d is not None:
        ax_2d.set_xlim(*xlim_2d)
    if ylim_2d is not None:
        ax_2d.set_ylim(*ylim_2d)

    return {"errors_mm": current_errors_mm, "errors_deg": current_errors_deg}


def create_interactive_gaze_plot(
    eyes: list[Eye],
    predict_fns: list[Callable[[Eye, Point3D], GazePrediction | None]],
    calibration_points: list[Point3D],
    plane_info: PlaneInfo,
    cameras: list,
    lights: list,
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
    n_eyes = len(eyes)

    # Compute calibration errors once per eye for the static overlay
    all_calib_data = [
        compute_calibration_errors(predict_fns[i], eyes[i], calibration_points, plane_info) for i in range(n_eyes)
    ]

    fig = plt.figure(figsize=(24, 10))
    interactive_eyes = [copy.deepcopy(eye) for eye in eyes]

    mean_x = sum(pt.x for pt in calibration_points) / len(calibration_points)
    mean_z = sum(pt.z for pt in calibration_points) / len(calibration_points)
    current_target = Point3D(mean_x, 0, mean_z)

    controls = InteractiveControls(interactive_eyes, current_target, step_size=10)

    def update_display() -> None:
        """Update both 3D and 2D plots with current target position."""
        fig.clear()
        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")
        ax_2d = fig.add_subplot(1, 2, 2)

        render_calibration_view(
            ax_3d,
            ax_2d,
            controls.target_point,
            interactive_eyes,
            predict_fns,
            calibration_points,
            plane_info,
            cameras,
            lights,
            cached_calib_data=all_calib_data,
            eye_labels=eye_labels,
            eye_colors=eye_colors,
            screen=screen,
        )

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
