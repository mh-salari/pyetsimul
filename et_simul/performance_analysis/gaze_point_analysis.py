"""Gaze point analysis for eye tracking systems.

This module analyzes eye tracker accuracy across different gaze target points on a screen.
The observer position is fixed while gaze targets vary across screen positions.
"""

import numpy as np
from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import plot_error_vectors, calculate_error_statistics


def accuracy_over_gaze_points(
    et,
    eye,
    observer_pos_test=None,
    grid_center=np.array([0, 200e-3]),
    dx=200e-3,
    dy=150e-3,
    grid_size=16,
):
    """Computes gaze error at different gaze target points on screen.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    Analyzes eye tracker accuracy by testing gaze estimation at various target positions
    in a grid pattern while keeping the observer position fixed.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)
        observer_pos_test: Observer position for testing (default: same as calib position)
        grid_center: Center point of the gaze target grid [x, y] in meters (default: [0, 200mm])
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

    # Calibrate eye tracker using current API
    et.run_calibration(e)

    # Define gaze target positions grid on screen
    X = np.linspace(grid_center[0] - dx, grid_center[0] + dx, grid_size)
    Y = np.linspace(grid_center[1] - dy, grid_center[1] + dy, grid_size)

    # Initialize result arrays
    U = np.zeros((grid_size, grid_size))
    V = np.zeros((grid_size, grid_size))
    errs_deg = np.zeros((grid_size, grid_size))

    # Output eye measurements
    apex_cornea_dist = np.linalg.norm(e.pos_apex - e.pos_cornea)
    cornea_pupil_dist = np.linalg.norm(e.pos_cornea - e.pos_pupil)

    print(f"Corneal radius: {apex_cornea_dist * 1e3:.3g} mm")
    print(f"Pupil radius:   {cornea_pupil_dist * 1e3:.3g} mm")

    # Set observer position for testing
    e.position = observer_pos_test[:3]

    # Main analysis loop - varying gaze target points
    for i in range(len(X)):
        for j in range(len(Y)):
            # Get predicted gaze position directly
            actual_point = np.array([X[i], Y[j]])
            predicted_gaze = et.estimate_gaze_at(e, actual_point)
            
            if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
                # Calculate error in mm
                U[j, i] = predicted_gaze.gaze_point[0] - X[i]
                V[j, i] = predicted_gaze.gaze_point[1] - Y[j]
                
                # Compute error in degrees using utility function
                errs_deg[j, i] = calculate_angular_error_degrees(
                    [X[i], Y[j]], predicted_gaze.gaze_point, observer_pos_test
                )
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
    print(f'Maximum error {errors["mtr"]["max"] * 1e3:.3g} mm')
    print(f'Mean error {errors["mtr"]["mean"] * 1e3:.3g} mm')
    print(f'Standard deviation {errors["mtr"]["std"] * 1e3:.3g} mm')

    # Plot using shared utility (gaze points use meters, no mm conversion)
    plot_error_vectors(X, Y, U, V, errors,
                      title_prefix="Gaze Point Analysis",
                      convert_to_mm=False,
                      xlabel="X position (m)",
                      ylabel="Y position (m)")

    return errors
