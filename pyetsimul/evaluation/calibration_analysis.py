"""Calibration analysis for eye tracking systems.

This module analyzes eye tracker calibration accuracy by testing gaze estimation
at the original calibration points to assess calibration quality.
"""

import numpy as np
from typing import Dict
from tabulate import tabulate

from ..core import Eye, EyeTracker
from ..types import Point3D, Position3D
from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import calculate_error_statistics
from .calibration_utils import pprint_polynomial_parameters
from ..visualization.interactive_calibration import create_interactive_calibration_plot


class CalibrationResults:
    """Calibration accuracy results with printing method."""

    def __init__(self, errors: Dict[str, Dict[str, float]]):
        self.errors = errors

    def __str__(self) -> str:
        """Basic string representation of calibration results."""
        mean_error_mm = self.errors["mtr"]["mean"] * 1e3
        mean_error_deg = self.errors["deg"]["mean"]
        return f"CalibrationResults(mean_error={mean_error_mm:.2f}mm / {mean_error_deg:.3f}°)"

    def pprint(self, title: str = "Calibration Accuracy") -> None:
        """Print formatted calibration error statistics."""
        print(f"\n{title}:")

        headers = ["Statistic", "Error (mm)", "Error (degrees)"]
        data = [
            ["Max", f"{self.errors['mtr']['max'] * 1e3:.4f}", f"{self.errors['deg']['max']:.4f}"],
            ["Mean", f"{self.errors['mtr']['mean'] * 1e3:.4f}", f"{self.errors['deg']['mean']:.4f}"],
            ["Std", f"{self.errors['mtr']['std'] * 1e3:.4f}", f"{self.errors['deg']['std']:.4f}"],
            ["Median", f"{self.errors['mtr']['median'] * 1e3:.4f}", f"{self.errors['deg']['median']:.4f}"],
        ]

        print(tabulate(data, headers=headers, tablefmt="grid"))


def accuracy_at_calibration_points(et: EyeTracker, eye: Eye, interactive_plot: bool = True) -> CalibrationResults:
    """Computes gaze error at calibration points to assess calibration quality.

    Evaluates calibration accuracy by testing gaze prediction at original calibration targets.
    Provides comprehensive error analysis with both spatial and angular metrics.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)
        interactive_plot: Whether to create interactive visualization (default: True)

    Returns:
        CalibrationResults object with error statistics and formatted printing methods
    """
    # Ensure eye tracker is calibrated before running analysis
    if not et.algorithm_state.is_calibrated:
        raise ValueError(
            "Eye tracker must be calibrated before running accuracy analysis. Call et.run_calibration(eye) first."
        )

    # Get calibration points and plane info
    calib_points = et.calib_points  # List[Position3D]
    n_points = len(calib_points)

    # Get the eye tracker's plane info for coordinate system
    plane_info = et.plane_info

    print(f"Analyzing calibration accuracy at {n_points} points...")

    # Output eye measurements
    apex_pos = eye.cornea.get_apex_position()
    apex_cornea_dist = np.linalg.norm(apex_pos - eye.cornea.center)
    cornea_pupil_dist = np.linalg.norm(eye.cornea.center - eye.pupil.pos_pupil)

    print(f"Corneal radius: {apex_cornea_dist * 1e3:.3g} mm")
    print(f"Pupil radius:   {cornea_pupil_dist * 1e3:.3g} mm")

    # Initialize result arrays
    actual_points = []
    predicted_points = []
    U = np.zeros(n_points)
    V = np.zeros(n_points)
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
            U[i] = predicted_coord1 - actual_coord1
            V[i] = predicted_coord2 - actual_coord2

            # Compute error in degrees using full 3D coordinates (convert to Point3D)
            target_point = Point3D(target_position.x, target_position.y, target_position.z)
            predicted_point = Point3D(predicted_pos.x, predicted_pos.y, predicted_pos.z)
            errs_deg[i] = calculate_angular_error_degrees(target_point, predicted_point, eye.position)

            # Collect data for table
            error_mm = np.sqrt(U[i] ** 2 + V[i] ** 2)
            table_data.append(
                [
                    i + 1,
                    f"({actual_coord1 * 1000:6.1f}, {actual_coord2 * 1000:6.1f})",
                    f"({predicted_coord1 * 1000:6.1f}, {predicted_coord2 * 1000:6.1f})",
                    f"{error_mm * 1000:8.2f}",
                    f"{errs_deg[i]:8.4f}",
                ]
            )
        else:
            predicted_points.append(Point3D(np.nan, np.nan, np.nan))
            U[i] = np.nan
            V[i] = np.nan
            errs_deg[i] = np.nan
            table_data.append(
                [i + 1, f"({actual_coord1 * 1000:6.1f}, {actual_coord2 * 1000:6.1f})", "FAILED", "--", "--"]
            )

    # Print the results table
    headers = ["Point", "Target (mm)", "Predicted (mm)", "Error (mm)", "Error (°)"]
    print("\nCalibration Point Analysis:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))

    # Extract coordinates from structured Point3D objects
    X = np.array([pt.x for pt in actual_points])
    Y = np.array([pt.y for pt in actual_points])

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

        # Create interactive visualization if requested
        if interactive_plot:
            create_interactive_calibration_plot(
                et, eye, X, Y, U, V, predicted_points, valid_mask, errs_deg, plane_info
            )
    else:
        print(f"\nCalibration Analysis Results: ALL {n_total} POINTS FAILED")
        errors = {
            "mtr": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
            "deg": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
        }

    return CalibrationResults(errors)
