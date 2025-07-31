"""Observer position analysis for eye tracking systems.

This module analyzes eye tracker accuracy across different observer positions.
The gaze target is fixed while observer position varies.
"""

import numpy as np
from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import plot_error_vectors, calculate_error_statistics


def accuracy_over_observer_positions(
    et,
    eye,
    gaze_target=np.array([0, 200e-3]),
    movement_range=50e-3,
    grid_size=16,
):
    """Computes gaze error at different observer positions.



    Analyzes eye tracker robustness by testing gaze estimation with the observer
    at various positions while looking at a fixed target point.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)
        gaze_target: Fixed gaze target point [x, y] in meters (default: [0, 200mm])
        movement_range: How far to move observer from calibration position in meters (default: 50mm)
        grid_size: Number of grid points per dimension (default: 16)

    Returns:
        Dictionary with error statistics (mean, max, std, median for both mm and degrees)
    """
    e = eye

    # Calibrate eye tracker at the reference position
    et.run_calibration(e)

    # Define observer position grid - move observer by ±movement_range from calibration position
    calib_x, calib_y, calib_z = e.position
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

    # Calculate gaze error with observer at different positions
    for i in range(len(X)):
        for j in range(len(Y)):
            # Move observer to test position [X[i], Y[j], Z]
            e.position = np.array([X[i], Y[j], Z])

            # Get predicted gaze position directly
            predicted_gaze = et.estimate_gaze_at(e, gaze_target)

            if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
                # Calculate error in mm
                U[j, i] = predicted_gaze.gaze_point[0] - gaze_target[0]
                V[j, i] = predicted_gaze.gaze_point[1] - gaze_target[1]

                # Compute error in degrees using utility function
                errs_deg[j, i] = calculate_angular_error_degrees(gaze_target, predicted_gaze.gaze_point, e.position)
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

    # Plot using shared utility with movement range in title
    title_prefix = f"Movement ±{movement_range * 1000:.0f}mm"
    plot_error_vectors(X, Y, U, V, errors, title_prefix=title_prefix)

    return errors
