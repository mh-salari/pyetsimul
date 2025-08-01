"""Calibration analysis for eye tracking systems.

This module analyzes eye tracker calibration accuracy by testing gaze estimation
at the original calibration points to assess calibration quality.
"""

import numpy as np
import copy
from typing import Dict, List
from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import calculate_error_statistics, plt
from ..visualization import prepare_eye_data_for_plots, plot_setup
from ..core import Eye, EyeTracker


def accuracy_at_calibration_points(et: EyeTracker, eye: Eye) -> Dict[str, Dict[str, float]]:
    """Computes gaze error at calibration points to assess calibration quality.

    This function calibrates the eye tracker and then tests its accuracy by
    predicting gaze at each of the original calibration points and measuring
    the prediction errors.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)

    Returns:
        Dictionary with error statistics (mean, max, std, median for both mm and degrees)
    """
    e = eye

    # Calibrate eye tracker using current API
    print("Running calibration...")
    et.run_calibration(e)

    # Get calibration points
    calib_points = et.calib_points  # Shape: (2, N) where N is number of points
    n_points = calib_points.shape[1]

    print(f"Analyzing calibration accuracy at {n_points} points...")

    # Output eye measurements
    apex_pos = e.cornea.get_apex_position()
    apex_cornea_dist = np.linalg.norm(apex_pos - e.cornea.center)
    cornea_pupil_dist = np.linalg.norm(e.cornea.center - e.pupil.pos_pupil)

    print(f"Corneal radius: {apex_cornea_dist * 1e3:.3g} mm")
    print(f"Pupil radius:   {cornea_pupil_dist * 1e3:.3g} mm")

    # Initialize result arrays
    actual_points = []
    predicted_points = []
    U = np.zeros(n_points)
    V = np.zeros(n_points)
    errs_deg = np.zeros(n_points)

    # Print polynomial parameters
    _print_polynomial_parameters(et)

    # Test prediction at each calibration point - side-by-side format
    print(f"\n{'Point':<6} {'Target (mm)':<18} {'Predicted (mm)':<18} {'Error (mm)':<12} {'Error (°)':<10}")
    print("-" * 75)

    for i in range(n_points):
        actual_x = calib_points[0, i]
        actual_y = calib_points[1, i]
        actual_point = np.array([actual_x, actual_y])

        # Get predicted gaze position
        predicted_gaze = et.estimate_gaze_at(e, actual_point)

        actual_points.append(actual_point)

        if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
            predicted_points.append(predicted_gaze.gaze_point)

            # Calculate error vectors
            U[i] = predicted_gaze.gaze_point[0] - actual_x
            V[i] = predicted_gaze.gaze_point[1] - actual_y

            # Compute error in degrees
            errs_deg[i] = calculate_angular_error_degrees(actual_point, predicted_gaze.gaze_point, e.position)

            # Progress output - side-by-side format
            error_mm = np.sqrt(U[i] ** 2 + V[i] ** 2)
            print(
                f"{i + 1:<6} "
                f"({actual_x * 1000:>6.1f}, {actual_y * 1000:>6.1f}){'':<1} "
                f"({predicted_gaze.gaze_point[0] * 1000:>6.1f}, {predicted_gaze.gaze_point[1] * 1000:>6.1f}){'':<1} "
                f"{error_mm * 1000:>8.2f}{'':<4} "
                f"{errs_deg[i]:>8.4f}"
            )
        else:
            predicted_points.append([np.nan, np.nan])
            U[i] = np.nan
            V[i] = np.nan
            errs_deg[i] = np.nan
            print(
                f"{i + 1:<6} "
                f"({actual_x * 1000:>6.1f}, {actual_y * 1000:>6.1f}){'':<7} "
                f"{'FAILED':<18} "
                f"{'--':<12} "
                f"{'--':<10}"
            )

    # Convert to arrays
    actual_points = np.array(actual_points)
    X = actual_points[:, 0]
    Y = actual_points[:, 1]

    # Calculate error statistics only for valid points
    valid_mask = ~(np.isnan(U) | np.isnan(V) | np.isnan(errs_deg))
    n_valid = np.sum(valid_mask)
    n_total = len(U)

    if n_valid > 0:
        errors = calculate_error_statistics(
            U[valid_mask].reshape(1, -1),
            V[valid_mask].reshape(1, -1),
            errs_deg[valid_mask].reshape(1, -1),
        )

        # Display statistics
        print(f"\nCalibration Analysis Results ({n_valid}/{n_total} points successful):")
        print(f"Maximum error {errors['mtr']['max'] * 1e3:.3g} mm ({errors['deg']['max']:.4f}°)")
        print(f"Mean error {errors['mtr']['mean'] * 1e3:.3g} mm ({errors['deg']['mean']:.4f}°)")
        print(f"Standard deviation {errors['mtr']['std'] * 1e3:.3g} mm ({errors['deg']['std']:.4f}°)")

        # Create interactive visualization
        _create_interactive_calibration_plot(et, e, X, Y, U, V, predicted_points, valid_mask, errs_deg)
    else:
        print(f"\nCalibration Analysis Results: ALL {n_total} POINTS FAILED")
        errors = {
            "mtr": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
            "deg": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
        }

    return errors


def _create_interactive_calibration_plot(
    et: EyeTracker,
    eye: Eye,
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    predicted_points: List,
    valid_mask: np.ndarray,
    errs_deg: np.ndarray,
) -> None:
    """Create interactive calibration plot with keyboard controls."""
    # Create figure with two subplots: 3D setup and 2D calibration analysis
    fig = plt.figure(figsize=(20, 8))

    # Create a copy of the eye to avoid modifying the original
    interactive_eye = copy.deepcopy(eye)

    # Interactive state
    current_target = np.array([np.mean(et.calib_points[0, :]), np.mean(et.calib_points[1, :])])
    step_size = 10e-3  # 10mm steps

    def update_display():
        """Update both 3D and 2D plots with current target position."""
        fig.clear()

        # Left subplot: 3D eye tracking setup
        ax_3d = fig.add_subplot(1, 2, 1, projection="3d")

        # Create target for 3D visualization
        target_3d = np.array([current_target[0], 0, current_target[1], 1])

        # Convert calibration points to list format
        calib_points_list = []
        for i in range(et.calib_points.shape[1]):
            calib_points_list.append([et.calib_points[0, i], et.calib_points[1, i]])

        # Prepare eye data
        prepared_data = prepare_eye_data_for_plots(interactive_eye, target_3d, et.lights, et.cameras[0])

        # Plot 3D setup
        plot_setup(
            ax_3d,
            prepared_data["eye_data"],
            target_3d,
            et.lights,
            et.cameras[0],
            prepared_data["cr_3d_list"],
            calib_points=calib_points_list,
        )

        # Add current target position to 3D plot
        ax_3d.scatter(
            [current_target[0]],
            [0],
            [current_target[1]],
            c="green",
            s=40,
            marker="x",
            label="Current Target",
        )
        ax_3d.legend()

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
        X_valid, Y_valid, U_valid, V_valid, pred_x, pred_y = [], [], [], [], [], []

        for i in valid_indices:
            pred_point = predicted_points[i]
            if not np.any(np.isnan(pred_point)):
                X_valid.append(X[i] * 1000)
                Y_valid.append(Y[i] * 1000)
                U_valid.append(U[i] * 1000)
                V_valid.append(V[i] * 1000)
                pred_x.append(pred_point[0] * 1000)
                pred_y.append(pred_point[1] * 1000)

        if len(X_valid) > 0:
            # Draw arrows from calibration points to predicted gaze points
            for i in range(len(X_valid)):
                ax.arrow(
                    X_valid[i],
                    Y_valid[i],
                    U_valid[i],
                    V_valid[i],
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

        # Get real-time prediction for current target
        current_prediction = et.estimate_gaze_at(interactive_eye, current_target)

        if current_prediction is not None and current_prediction.gaze_point is not None:
            # Plot current target and prediction
            ax.scatter(
                [current_target[0] * 1000],
                [current_target[1] * 1000],
                marker="x",
                s=60,
                c="green",
                label="Current Target",
                zorder=5,
            )
            ax.scatter(
                [current_prediction.gaze_point[0] * 1000],
                [current_prediction.gaze_point[1] * 1000],
                marker="x",
                s=40,
                c="orange",
                label="Current Prediction",
                zorder=5,
            )

            # Draw error arrow for current prediction
            error_x = (current_prediction.gaze_point[0] - current_target[0]) * 1000
            error_y = (current_prediction.gaze_point[1] - current_target[1]) * 1000
            ax.arrow(
                current_target[0] * 1000,
                current_target[1] * 1000,
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
            current_error_deg = calculate_angular_error_degrees(
                current_target, current_prediction.gaze_point, interactive_eye.position
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
                f"Calibration errors: {calib_errors_text}"
            )
        else:
            ax.scatter(
                [current_target[0] * 1000],
                [current_target[1] * 1000],
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
                f"Calibration errors: {calib_errors_text}"
            )

        ax.set_xlabel("X Position (mm)")
        ax.set_ylabel("Y Position (mm)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_aspect("equal")

        plt.tight_layout()
        fig.canvas.draw()

    def on_key_press(event):
        """Handle keyboard input for moving target and eye."""
        nonlocal current_target

        if event.key == "escape":
            plt.close(fig)
            return

        # TARGET MOVEMENT (Arrow keys)
        elif event.key == "up":
            current_target[1] += step_size
        elif event.key == "down":
            current_target[1] -= step_size
        elif event.key == "left":
            current_target[0] -= step_size
        elif event.key == "right":
            current_target[0] += step_size

        # EYE MOVEMENT (I/K/J/L/./,)
        elif event.key == "j":
            interactive_eye.trans[0, 3] -= step_size  # Eye left (decrease X)
        elif event.key == "l":
            interactive_eye.trans[0, 3] += step_size  # Eye right (increase X)
        elif event.key == "i":
            interactive_eye.trans[2, 3] += step_size  # Eye up (increase Z)
        elif event.key == "k":
            interactive_eye.trans[2, 3] -= step_size  # Eye down (decrease Z)
        elif event.key == ".":
            interactive_eye.trans[1, 3] -= step_size  # Eye closer to camera (decrease Y)
        elif event.key == ",":
            interactive_eye.trans[1, 3] += step_size  # Eye farther from camera (increase Y)
        else:
            return

        update_display()

    # Connect keyboard handler and display
    fig.canvas.mpl_connect("key_press_event", on_key_press)

    print("\nINTERACTIVE MODE:")
    print("Target Movement (Arrow keys):")
    print("  ↑/↓: Move target up/down")
    print("  ←/→: Move target left/right")
    print()
    print("Eye Movement (I/K/J/L/./):")
    print("  I/K: Move eye up/down")
    print("  J/L: Move eye left/right")
    print("  ./,: Move eye closer/farther from camera")
    print()
    print("Press ESC to exit")
    print("Click on the plot window to focus for keyboard input\n")

    update_display()
    plt.show()


def _print_polynomial_parameters(et: EyeTracker) -> None:
    """Print polynomial parameters from calibrated eye tracker."""
    print("\nPolynomial Parameters:")
    print("-" * 40)

    if hasattr(et, "polynomial_name"):
        print(f"Polynomial type: {et.polynomial_name}")

    if hasattr(et, "state") and et.state:
        # Handle 1D polynomials (shared parameters)
        if "A" in et.state:
            A = et.state["A"]
            print(f"Calibration matrix A shape: {A.shape}")
            print("A matrix:")
            for i, row in enumerate(A):
                coord_name = "X" if i == 0 else "Y"
                print(f"  {coord_name}: [{', '.join(f'{val:8.4f}' for val in row)}]")

        # Handle 2D polynomials (separate parameters)
        elif "A_x" in et.state and "A_y" in et.state:
            A_x = et.state["A_x"]
            A_y = et.state["A_y"]
            print(f"X calibration matrix shape: {A_x.shape}")
            print(f"Y calibration matrix shape: {A_y.shape}")
            print("A_x matrix:")
            for i, row in enumerate(A_x):
                print(f"  [{', '.join(f'{val:8.4f}' for val in row)}]")
            print("A_y matrix:")
            for i, row in enumerate(A_y):
                print(f"  [{', '.join(f'{val:8.4f}' for val in row)}]")
    else:
        print("No calibration parameters found (tracker not calibrated)")
