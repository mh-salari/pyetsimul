"""Calibration analysis for eye tracking systems.

This module analyzes eye tracker calibration accuracy by testing gaze estimation
at the original calibration points to assess calibration quality.
"""

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

from ..core import Eye, EyeTracker
from ..geometry.conversions import calculate_angular_error_degrees
from ..types import Point3D, Position3D
from ..visualization.interactive_gaze_plot import create_interactive_gaze_plot
from .analysis_utils import calculate_error_statistics
from .calibration_utils import pprint_polynomial_parameters


class CalibrationResults:
    """Calibration accuracy results with on-demand visualization."""

    def __init__(self, errors: dict[str, dict[str, float]], plot_data: dict | None = None) -> None:
        """Initialize calibration results.

        Args:
            errors: Dictionary containing error statistics in different units
            plot_data: Internal data needed for on-demand plot creation (None if all points failed)

        """
        self.errors = errors
        self._plot_data = plot_data

    def __str__(self) -> str:
        """Basic string representation of calibration results."""
        mean_error_mm = self.errors["mtr"]["mean"] * 1e3
        mean_error_deg = self.errors["deg"]["mean"]
        return f"CalibrationResults(mean_error={mean_error_deg:.3f}° / {mean_error_mm:.2f}mm)"

    def pprint(self, title: str = "Calibration Accuracy") -> None:
        """Print formatted calibration error statistics."""
        print(f"\n{title}:")

        headers = ["Statistic", "Error (degrees)", "Error (mm)"]
        data = [
            ["Max", f"{self.errors['deg']['max']:.4f}", f"{self.errors['mtr']['max'] * 1e3:.4f}"],
            ["Mean", f"{self.errors['deg']['mean']:.4f}", f"{self.errors['mtr']['mean'] * 1e3:.4f}"],
            ["Std", f"{self.errors['deg']['std']:.4f}", f"{self.errors['mtr']['std'] * 1e3:.4f}"],
            ["Median", f"{self.errors['deg']['median']:.4f}", f"{self.errors['mtr']['median'] * 1e3:.4f}"],
        ]

        print(tabulate(data, headers=headers, tablefmt="grid"))

    def interactive_plot(self, show: bool = True) -> plt.Figure:
        """Create the interactive calibration plot on demand.

        No figure is created until this method is called, preventing figures from
        lurking in matplotlib's global figure manager and appearing unexpectedly.

        Args:
            show: If True (default), display the figure with plt.show() (blocks until closed).
                  If False, return the figure for saving (fig.savefig()) without displaying.
                  The figure is removed from matplotlib's manager to prevent it from
                  appearing unexpectedly in later plt.show() calls.

        Returns:
            The matplotlib Figure.

        """
        if self._plot_data is None:
            raise ValueError("No valid calibration data available for plotting.")

        et = self._plot_data["et"]
        eye = self._plot_data["eye"]
        return create_interactive_gaze_plot(
            [eye],
            [et.estimate_gaze_at],
            et.calib_points,
            et.plane_info,
            et.cameras,
            et.lights,
            et.use_legacy_look_at,
            show=show,
        )


def accuracy_at_calibration_points(et: EyeTracker, eye: Eye) -> CalibrationResults:
    """Computes gaze error at calibration points to assess calibration quality.

    Evaluates calibration accuracy by testing gaze prediction at original calibration targets.
    Provides comprehensive error analysis with both spatial and angular metrics.

    To visualize the results, call calib_results.interactive_plot() on the returned object.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)

    Returns:
        CalibrationResults object with error statistics, printing, and on-demand visualization

    """
    # Ensure eye tracker is calibrated before running analysis
    if not et.algorithm_state.is_calibrated:
        raise ValueError(
            "Eye tracker must be calibrated before running accuracy analysis. Call et.run_calibration(eye) first."
        )

    # Get calibration points and plane info
    calib_points = et.calib_points
    n_points = len(calib_points)

    # Get the eye tracker's plane info for coordinate system
    plane_info = et.plane_info

    print(f"Analyzing calibration accuracy at {n_points} points...")

    # Output eye measurements
    apex_pos = eye.cornea.get_apex_position()
    apecornea_surface_x_dist = np.linalg.norm(apex_pos - eye.cornea.center)
    cornea_pupil_dist = np.linalg.norm(eye.cornea.center - eye.pupil.pos_pupil)

    print(f"Corneal radius: {apecornea_surface_x_dist * 1e3:.3g} mm")
    print(f"Pupil radius:   {cornea_pupil_dist * 1e3:.3g} mm")

    # Initialize result arrays
    actual_points = []
    predicted_points = []
    u = np.zeros(n_points)
    v = np.zeros(n_points)
    errs_deg = np.zeros(n_points)

    # Print polynomial parameters
    pprint_polynomial_parameters(et)

    # Test calibrated polynomial by predicting each calibration point with fresh measurements
    # Use calibrated polynomial to test each calibration point with fresh measurements
    calibration_fit_results = et.test_calibration_fit(eye)

    # Collect data for tabulate
    table_data = []

    for i, (target_position, predicted_gaze) in enumerate(calibration_fit_results):
        # Extract 2D coordinates using plane info for coordinate system consistency
        actual_coord1, actual_coord2 = plane_info.extract_2d_coords(target_position)
        actual_point = Point3D(actual_coord1, actual_coord2, 0.0)  # 2D coordinates in plane

        actual_points.append(actual_point)

        if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
            predicted_points.append(predicted_gaze.gaze_point)

            # Extract predicted coordinates using plane info
            predicted_pos = Position3D(
                predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z
            )
            predicted_coord1, predicted_coord2 = plane_info.extract_2d_coords(predicted_pos)

            # Calculate error vectors using plane coordinates
            u[i] = predicted_coord1 - actual_coord1
            v[i] = predicted_coord2 - actual_coord2

            # Compute error in degrees using full 3D coordinates (convert to Point3D)
            target_point = Point3D(target_position.x, target_position.y, target_position.z)
            predicted_point = Point3D(predicted_pos.x, predicted_pos.y, predicted_pos.z)
            errs_deg[i] = calculate_angular_error_degrees(target_point, predicted_point, eye.position)

            # Collect data for table
            error_mm = np.sqrt(u[i] ** 2 + v[i] ** 2)
            table_data.append([
                i + 1,
                f"({actual_coord1 * 1000:6.1f}, {actual_coord2 * 1000:6.1f})",
                f"({predicted_coord1 * 1000:6.1f}, {predicted_coord2 * 1000:6.1f})",
                f"{error_mm * 1000:8.2f}",
                f"{errs_deg[i]:8.4f}",
            ])
        else:
            predicted_points.append(Point3D(np.nan, np.nan, np.nan))
            u[i] = np.nan
            v[i] = np.nan
            errs_deg[i] = np.nan
            table_data.append([
                i + 1,
                f"({actual_coord1 * 1000:6.1f}, {actual_coord2 * 1000:6.1f})",
                "FAILED",
                "--",
                "--",
            ])

    # Print the results table
    headers = ["Point", "Target (mm)", "Predicted (mm)", "Error (mm)", "Error (°)"]
    print("\nCalibration Point Analysis:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Calculate error statistics only for valid points
    valid_mask = ~(np.isnan(u) | np.isnan(v) | np.isnan(errs_deg))
    n_valid = np.sum(valid_mask)
    n_total = len(u)

    if n_valid > 0:
        errors = calculate_error_statistics(
            u[valid_mask].reshape(1, -1),
            v[valid_mask].reshape(1, -1),
            errs_deg[valid_mask].reshape(1, -1),
        )

        # Display statistics
        print(f"\nCalibration Analysis Results ({n_valid}/{n_total} points successful):")
        print(f"Maximum error {errors['deg']['max']:.4f}° ({errors['mtr']['max'] * 1e3:.3g} mm)")
        print(f"Mean error {errors['deg']['mean']:.4f}° ({errors['mtr']['mean'] * 1e3:.3g} mm)")
        print(f"Standard deviation {errors['deg']['std']:.4f}° ({errors['mtr']['std'] * 1e3:.3g} mm)")

        # Store minimal data for on-demand plot creation via interactive_plot()
        plot_data = {"et": et, "eye": eye}
    else:
        print(f"\nCalibration Analysis Results: ALL {n_total} POINTS FAILED")
        errors = {
            "mtr": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
            "deg": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
        }
        plot_data = None

    return CalibrationResults(errors, plot_data=plot_data)
