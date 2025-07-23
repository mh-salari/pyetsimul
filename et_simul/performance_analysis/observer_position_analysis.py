"""Observer position analysis for eye tracking systems.

This module analyzes eye tracker accuracy across different observer positions.
The gaze target is fixed while observer position varies.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..core import Eye


def accuracy_over_observer_positions(
    et,
    eye=None,
    observer_pos_calib=np.array([0, 550e-3, 350e-3, 1]),
    gaze_target=np.array([0, 200e-3]),
    movement_range=50e-3,
    grid_size=16,
):
    """Computes gaze error at different observer positions.

    This function is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or lat

    Analyzes eye tracker robustness by testing gaze estimation with the observer
    at various positions while looking at a fixed target point.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (if None, creates default eye with r_cornea=7.98e-3)
        observer_pos_calib: Observer position for calibration (default: [0, 550mm, 350mm, 1])
        gaze_target: Fixed gaze target point [x, y] in meters (default: [0, 200mm])
        movement_range: How far to move observer from calibration position in meters (default: 50mm)
        grid_size: Number of grid points per dimension (default: 16)

    Returns:
        Dictionary with error statistics (mean, max, std, median for both mm and degrees)
    """
    # Use provided eye or create default one
    if eye is None:
        e = Eye(rest_pos=np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]]))
        e.trans[:3, 3] = observer_pos_calib[:3]
    else:
        e = eye

    # Calibrate eye tracker at the reference position
    et.run_calibration(e)

    # Define observer position grid - move observer by ±movement_range from calibration position
    calib_x, calib_y, calib_z = observer_pos_calib[:3]
    X = np.linspace(
        calib_x - movement_range, calib_x + movement_range, grid_size
    )  # ±movement_range from calib X
    Y = np.linspace(
        calib_y - movement_range, calib_y + movement_range, grid_size
    )  # ±movement_range from calib Y
    Z = calib_z  # Fix Z to calibration position

    # Initialize result arrays
    U = np.zeros((grid_size, grid_size))
    V = np.zeros((grid_size, grid_size))
    errs_deg = np.zeros((grid_size, grid_size))

    # Output eye measurements
    apex_cornea_dist = np.linalg.norm(e.pos_apex - e.pos_cornea)
    cornea_pupil_dist = np.linalg.norm(e.pos_cornea - e.pos_pupil)

    print(f"Corneal radius: {apex_cornea_dist * 1e3:.3g} mm")
    print(f"Pupil radius:   {cornea_pupil_dist * 1e3:.3g} mm")

    # Calculate gaze error with observer at different positions
    for i in range(len(X)):
        for j in range(len(Y)):
            # Move observer to test position [X[i], Y[j], Z]
            e.trans[:3, 3] = np.array([X[i], Y[j], Z])

            # Calculate gaze error for fixed target point
            U[j, i], V[j, i] = et.calculate_gaze_error(e, gaze_target)

            # Compute error in degrees
            gaze3d_real = np.array([gaze_target[0], 0, gaze_target[1]]) - e.trans[:3, 3]
            gaze3d_measured = (
                np.array([gaze_target[0] + U[j, i], 0, gaze_target[1] + V[j, i]])
                - e.trans[:3, 3]
            )
            gaze3d_real = gaze3d_real / np.linalg.norm(gaze3d_real)
            gaze3d_measured = gaze3d_measured / np.linalg.norm(gaze3d_measured)

            # Angle between vectors in degrees
            dot_product = np.clip(np.dot(gaze3d_real, gaze3d_measured), -1, 1)
            errs_deg[j, i] = 180 / np.pi * np.real(np.arccos(dot_product))

        # Progress indicator
        print(".", end="", flush=True)

    print()

    # Plot gaze error vectors
    plt.figure(figsize=(10, 8))
    q = plt.quiver(
        X * 1000, Y * 1000, U * 1000, V * 1000, scale=150, alpha=0.8, width=0.002
    )

    # Automatically adjust plot limits to include arrow tips
    # Create meshgrid for arrow positions
    XX, YY = np.meshgrid(X * 1000, Y * 1000)
    u_vec, v_vec = U * 1000, V * 1000

    # Calculate arrow tip positions
    arrow_tips_x = XX.flatten() + u_vec.flatten()
    arrow_tips_y = YY.flatten() + v_vec.flatten()

    # Set limits to include both arrow bases and tips with small margin
    all_x = np.concatenate([XX.flatten(), arrow_tips_x])
    all_y = np.concatenate([YY.flatten(), arrow_tips_y])

    x_range = np.max(all_x) - np.min(all_x)
    y_range = np.max(all_y) - np.min(all_y)
    margin_x = x_range * 0.05  # 5% margin
    margin_y = y_range * 0.05  # 5% margin

    plt.xlim(np.min(all_x) - margin_x, np.max(all_x) + margin_x)
    plt.ylim(np.min(all_y) - margin_y, np.max(all_y) + margin_y)

    plt.xlabel("Observer X position (mm)")
    plt.ylabel("Observer Y position (mm)")
    plt.title("Gaze Error Vectors (mm)")
    plt.grid(True, alpha=0.3)

    # Calculate error statistics using numpy
    errs_mtr = np.sqrt(U**2 + V**2).flatten()
    errors = {
        "mtr": {
            "mean": np.mean(errs_mtr),
            "max": np.max(errs_mtr),
            "std": np.std(errs_mtr),
            "median": np.median(errs_mtr),
        },
        "deg": {
            "mean": np.mean(errs_deg.flatten()),
            "max": np.max(errs_deg.flatten()),
            "std": np.std(errs_deg.flatten()),
            "median": np.median(errs_deg.flatten()),
        },
    }

    # Display statistics
    print(f'Maximum error {errors["mtr"]["max"] * 1e3:.3g} mm')
    print(f'Mean error {errors["mtr"]["mean"] * 1e3:.3g} mm')
    print(f'Standard deviation {errors["mtr"]["std"] * 1e3:.3g} mm')

    # Plot title
    title = (
        f'Movement ±{movement_range*1000:.0f}mm: Max {errors["mtr"]["max"] * 1e3:.3g} mm, '
        + f'Mean {errors["mtr"]["mean"] * 1e3:.3g} mm, '
        + f'Std {errors["mtr"]["std"] * 1e3:.3g} mm'
    )
    plt.title(title)
    plt.show()

    return errors
