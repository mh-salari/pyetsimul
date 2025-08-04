"""Gaze point analysis for eye tracking systems.

This module analyzes eye tracker accuracy across different gaze target points on a screen.
The observer position is fixed while gaze targets vary across screen positions.
"""

import numpy as np
from typing import Optional, Dict
from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import plot_error_vectors, calculate_error_statistics
from ..core import Eye, EyeTracker
from ..types import Position3D, Point2D


def accuracy_over_gaze_points(
    et: EyeTracker,
    eye: Eye,
    grid_center: Position3D,
    observer_pos_test: Optional[Position3D] = None,
    dx: float = 200e-3,
    dy: float = 150e-3,
    grid_size: int = 16,
) -> Dict[str, Dict[str, float]]:
    """Computes gaze error at different gaze target points on screen.

    Evaluates gaze tracking accuracy across a grid of screen positions.
    Tests calibration quality by varying target positions while keeping observer fixed.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)
        observer_pos_test: Observer position for testing (default: same as calib position)
        grid_center: Center point of the gaze target grid (x, y, z) in meters (required)
        dx: Half-width of the grid in x direction in meters (default: 200mm, so grid spans ±200mm)
        dy: Half-height of the grid in y direction in meters (default: 150mm, so grid spans ±150mm)
        grid_size: Number of grid points per dimension (default: 16)

    Returns:
        Dictionary with error statistics (mean, max, std, median for both mm and degrees)
    """
    e = eye

    # Set test position
    if observer_pos_test is None:
        observer_pos_test = e.position

    # Ensure eye tracker is calibrated before running analysis
    if not et.algorithm_state.is_calibrated:
        raise ValueError(
            "Eye tracker must be calibrated before running accuracy analysis. Call et.run_calibration(eye) first."
        )

    # Get the eye tracker's plane info for coordinate system
    plane_info = et.plane_info

    # Define gaze target positions grid using the detected plane
    # Extract the two varying coordinates from the plane info
    coord1_center, coord2_center = plane_info.extract_2d_coords(grid_center)
    coord1_values = np.linspace(coord1_center - dx, coord1_center + dx, grid_size)
    coord2_values = np.linspace(coord2_center - dy, coord2_center + dy, grid_size)

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

    # Set observer position for testing
    e.position = observer_pos_test

    # Main analysis loop - varying gaze target points
    for i in range(len(coord1_values)):
        for j in range(len(coord2_values)):
            # Reconstruct 3D point using plane info
            actual_point_3d = plane_info.reconstruct_3d_point(coord1_values[i], coord2_values[j])
            actual_point = Position3D(actual_point_3d.x, actual_point_3d.y, actual_point_3d.z)
            predicted_gaze = et.estimate_gaze_at(e, actual_point)

            if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
                # Extract coordinates using plane info for error calculation
                actual_coord1, actual_coord2 = plane_info.extract_2d_coords(actual_point)
                predicted_coord1, predicted_coord2 = plane_info.extract_2d_coords(
                    Position3D(predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z)
                )

                # Calculate error in mm using the plane's coordinate system
                U[j, i] = predicted_coord1 - actual_coord1
                V[j, i] = predicted_coord2 - actual_coord2

                # Compute error in degrees using utility function
                # Convert 3D points to 2D screen coordinates using plane mapping
                actual_2d = Point2D(actual_coord1, actual_coord2)
                predicted_2d = Point2D(predicted_coord1, predicted_coord2)
                errs_deg[j, i] = calculate_angular_error_degrees(actual_2d, predicted_2d, observer_pos_test)
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

    # Plot using shared utility (convert coordinates to mm for display)
    plot_error_vectors(
        coord1_values,
        coord2_values,
        U,
        V,
        errors,
        title_prefix="Gaze Point Analysis",
        convert_to_mm=True,
        xlabel=f"{plane_info.primary_axis.upper()} position (mm)",
        ylabel=f"{plane_info.secondary_axis.upper()} position (mm)",
    )

    return errors
