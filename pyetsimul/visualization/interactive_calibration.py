"""Interactive calibration plotting for evaluation.

This module provides interactive plotting functions for calibration analysis,
allowing real-time exploration of calibration accuracy and gaze estimation.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from pyetsimul.core import Eye, EyeTracker

from ..geometry.conversions import calculate_angular_error_degrees
from ..geometry.plane_detection import PlaneInfo
from ..types import Point2D, Point3D
from .coordinate_utils import prepare_eye_data_for_plots
from .interactive_controls import InteractiveControls
from .setup_plots import plot_setup


def create_interactive_calibration_plot(
    et: EyeTracker,
    eye: Eye,
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    predicted_points: list,
    valid_mask: np.ndarray,
    errs_deg: np.ndarray,
    plane_info: PlaneInfo,
) -> None:
    """Create interactive calibration plot with keyboard controls.

    Provides real-time exploration of gaze tracking accuracy with 3D setup visualization.
    Allows interactive target positioning to test calibration quality.
    """
    fig = plt.figure(figsize=(24, 10))
    interactive_eye = copy.deepcopy(eye)

    mean_x = sum(pt.x for pt in et.calib_points) / len(et.calib_points)
    mean_z = sum(pt.z for pt in et.calib_points) / len(et.calib_points)
    current_target = Point3D(mean_x, 0, mean_z)

    controls = InteractiveControls(interactive_eye, current_target, step_size=10e-3)

    def update_display() -> None:
        """Update both 3D and 2D plots with current target position."""
        fig.clear()

        # Left subplot: 3D eye tracking setup
        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")

        target_3d = Point3D(controls.target_point.x, 0, controls.target_point.z)

        # Convert calibration points to list format using plane mapping
        calib_points_list: list[Point2D] = []
        for pt in et.calib_points:
            calib_points_list.append(Point2D(*plane_info.extract_2d_coords(pt)))

        # Prepare eye data
        prepared_data = prepare_eye_data_for_plots(
            [interactive_eye], [target_3d], et.lights, et.cameras, et.use_legacy_look_at
        )

        # Plot 3D setup
        plot_setup(
            ax_3d,
            prepared_data["eyes_data"],
            [target_3d],
            et.lights,
            et.cameras,
            prepared_data["cr_3d_lists"],
            calib_points=calib_points_list,
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

        # Add title to 3D subplot
        ax_3d.set_title("Eye Tracking Setup", fontsize=14, fontweight="bold", pad=20)

        # Right subplot: 2D calibration analysis with real-time prediction
        ax = fig.add_subplot(1, 2, 2)
        ax.set_facecolor("white")

        # Plot all calibration points
        ax.scatter(
            X * 1000,
            Y * 1000,
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
                # Check if Point3D has valid coordinates (not NaN)
                x_valid.append(X[i] * 1000)
                y_valid.append(Y[i] * 1000)
                u_valid.append(U[i] * 1000)
                v_valid.append(V[i] * 1000)
                # Use plane coordinates for consistent mapping
                pred_pos = Point3D(pred_point.x, pred_point.y, pred_point.z)
                pred_coord1, pred_coord2 = plane_info.extract_2d_coords(pred_pos)
                pred_x.append(pred_coord1 * 1000)
                pred_y.append(pred_coord2 * 1000)

        if len(x_valid) > 0:
            # Draw arrows from calibration points to predicted gaze points
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

        current_prediction = et.estimate_gaze_at(interactive_eye, controls.target_point)

        if current_prediction is not None and current_prediction.gaze_point is not None:
            # Extract coordinates using plane mapping for consistent display
            target_coord1, target_coord2 = plane_info.extract_2d_coords(controls.target_point)
            pred_pos = Point3D(
                current_prediction.gaze_point.x, current_prediction.gaze_point.y, current_prediction.gaze_point.z
            )
            pred_coord1, pred_coord2 = plane_info.extract_2d_coords(pred_pos)

            # Plot current target and prediction
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

            # Draw error arrow for current prediction
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

            # Calculate current error
            current_error_mm = np.sqrt(error_x**2 + error_y**2)
            # Calculate angular error using full 3D coordinates
            current_error_deg = calculate_angular_error_degrees(
                controls.target_point, pred_pos, interactive_eye.position
            )

            # Create calibration error summary for title
            valid_errors = errs_deg[valid_mask]
            if len(valid_errors) > 0:
                calib_errors_text = f"Avg: {np.mean(valid_errors):.3f}° | Max: {np.max(valid_errors):.3f}°"
            else:
                calib_errors_text = "No valid calibration points"

            # Update title with current error and calibration errors
            ax.set_title(
                f"Interactive Calibration Analysis\n"
                f"Current gaze error: {current_error_mm:.2f}mm ({current_error_deg:.4f}°)\n"
                f"Calibration errors: {calib_errors_text}",
                fontsize=12,
                fontweight="bold",
                pad=20,
            )
        else:
            # Extract coordinates using plane mapping for failed prediction case
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

            # Create calibration error summary for title
            valid_errors = errs_deg[valid_mask]
            if len(valid_errors) > 0:
                calib_errors_text = f"Avg: {np.mean(valid_errors):.3f}° | Max: {np.max(valid_errors):.3f}°"
            else:
                calib_errors_text = "No valid calibration points"

            ax.set_title(
                f"Interactive Calibration Analysis\n"
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

        # Use constrained_layout for better spacing and prevent title cutoff
        plt.subplots_adjust(top=0.9, bottom=0.1, left=0.05, right=0.95, wspace=0.3)
        fig.canvas.draw()

    controls.set_update_callback(update_display)
    fig.canvas.mpl_connect("key_press_event", controls.handle_key_press)

    print("\nINTERACTIVE MODE:")
    InteractiveControls.print_controls(additional_controls={"Exit": "ESC"})
    print("Click on the plot window to focus for keyboard input\n")

    update_display()

    plt.show()
