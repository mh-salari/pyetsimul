"""Observer position analysis for eye tracking systems.

This module analyzes eye tracker accuracy across different observer positions.
The gaze target is fixed while observer position varies.
"""

import numpy as np
from typing import Dict
from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import plot_error_vectors, calculate_error_statistics
from ..core import Eye, EyeTracker
from ..types import Position3D


def accuracy_over_observer_positions(
    et: EyeTracker,
    eye: Eye,
    gaze_target: Position3D,
    movement_range: float = 50e-3,
    grid_size: int = 16,
) -> Dict[str, Dict[str, float]]:
    """Computes gaze error at different observer positions.

    Evaluates gaze tracking robustness by testing accuracy with observer movement.
    Tests calibration quality by varying observer position while keeping target fixed.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)
        gaze_target: Fixed gaze target point in meters (required)
        movement_range: How far to move observer from calibration position in meters (default: 50mm)
        grid_size: Number of grid points per dimension (default: 16)

    Returns:
        Dictionary with error statistics (mean, max, std, median for both mm and degrees)
    """
    e = eye

    # Ensure eye tracker is calibrated before running analysis
    if not et.algorithm_state.is_calibrated:
        raise ValueError(
            "Eye tracker must be calibrated before running accuracy analysis. Call et.run_calibration(eye) first."
        )

    # Define observer position grid - move observer by ±movement_range from calibration position
    calib_x, calib_y, calib_z = e.position.x, e.position.y, e.position.z
    X = np.linspace(calib_x - movement_range, calib_x + movement_range, grid_size)  # ±movement_range from calib X
    Y = np.linspace(calib_y - movement_range, calib_y + movement_range, grid_size)  # ±movement_range from calib Y
    Z = calib_z  # Fix Z to calibration position

    # Initialize result arrays
    U = np.zeros((grid_size, grid_size))
    V = np.zeros((grid_size, grid_size))
    errs_deg = np.zeros((grid_size, grid_size))

    # Output eye measurements
    apex_pos = e.cornea.get_apex_position()
    apex_cornea_dist = np.linalg.norm(apex_pos - e.cornea.center)
    cornea_pupil_dist = np.linalg.norm(e.cornea.center - e.pupil.pos_pupil)

    print(f"Corneal radius: {apex_cornea_dist * 1e3:.3g} mm")
    print(f"Pupil radius:   {cornea_pupil_dist * 1e3:.3g} mm")

    # Get the eye tracker's plane info for coordinate system
    plane_info = et.plane_info

    # Calculate gaze error with observer at different positions
    for i in range(len(X)):
        for j in range(len(Y)):
            # Move observer to test position using Position3D
            e.position = Position3D(X[i], Y[j], Z)

            # Get predicted gaze position directly
            predicted_gaze = et.estimate_gaze_at(e, gaze_target)

            if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
                # Extract coordinates using plane info for error calculation
                target_coord1, target_coord2 = plane_info.extract_2d_coords(gaze_target)
                predicted_pos = Position3D(
                    predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z
                )
                predicted_coord1, predicted_coord2 = plane_info.extract_2d_coords(predicted_pos)

                # Calculate error in mm using the plane's coordinate system
                U[j, i] = predicted_coord1 - target_coord1
                V[j, i] = predicted_coord2 - target_coord2

                # Compute error in degrees using full 3D coordinates
                errs_deg[j, i] = calculate_angular_error_degrees(gaze_target, predicted_pos, e.position)
            else:
                # Handle prediction failure
                U[j, i] = np.nan
                V[j, i] = np.nan
                errs_deg[j, i] = np.nan

        # Progress indicator
        print(".", end="", flush=True)

    print()

    # Calculate error statistics
    errors = calculate_error_statistics(U, V, errs_deg)

    # Display statistics
    print(f"Maximum error {errors['mtr']['max'] * 1e3:.3g} mm")
    print(f"Mean error {errors['mtr']['mean'] * 1e3:.3g} mm")
    print(f"Standard deviation {errors['mtr']['std'] * 1e3:.3g} mm")

    # Plot using shared utility with movement range in title (convert coordinates to mm for display)
    title_prefix = f"Observer Movement ±{movement_range * 1000:.0f}mm"
    plot_error_vectors(
        X,
        Y,
        U,
        V,
        errors,
        title_prefix=title_prefix,
        convert_to_mm=True,
        xlabel="Observer X position (mm)",
        ylabel="Observer Y position (mm)",
    )

    return errors
