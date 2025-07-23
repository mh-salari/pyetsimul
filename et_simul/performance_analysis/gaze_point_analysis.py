"""Gaze point analysis for eye tracking systems.

This module analyzes eye tracker accuracy across different gaze target points on a screen.
The observer position is fixed while gaze targets vary across screen positions.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..core import Eye


def accuracy_over_gaze_points(
    et,
    eye=None,
    observer_pos_calib=np.array([0, 550e-3, 350e-3, 1]),
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
        eye: Pre-configured Eye object (if None, creates default eye with r_cornea=7.98e-3)
        observer_pos_calib: Observer position for calibration (default: [0, 550mm, 350mm, 1])
        observer_pos_test: Observer position for testing (default: same as calib position)
        grid_center: Center point of the gaze target grid [x, y] in meters (default: [0, 200mm])
        dx: Half-width of the grid in x direction in meters (default: 200mm, so grid spans ±200mm)
        dy: Half-height of the grid in y direction in meters (default: 150mm, so grid spans ±150mm)
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

    # Set test position
    if observer_pos_test is None:
        observer_pos_test = observer_pos_calib

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
    e.trans[:3, 3] = observer_pos_test[:3]

    # Main analysis loop - varying gaze target points
    for i in range(len(X)):
        for j in range(len(Y)):
            # Calculate gaze error for this target point
            U[j, i], V[j, i] = et.calculate_gaze_error(e, np.array([X[i], Y[j]]))

            # Compute error in degrees
            gaze3d_real = np.array([X[i], 0, Y[j]]) - observer_pos_test[:3]
            gaze3d_measured = (
                np.array([X[i] + U[j, i], 0, Y[j] + V[j, i]]) - observer_pos_test[:3]
            )
            gaze3d_real = gaze3d_real / np.linalg.norm(gaze3d_real)
            gaze3d_measured = gaze3d_measured / np.linalg.norm(gaze3d_measured)

            # Angle between vectors in degrees
            dot_product = np.clip(np.dot(gaze3d_real, gaze3d_measured), -1, 1)
            errs_deg[j, i] = 180 / np.pi * np.real(np.arccos(dot_product))

        # Progress indicator
        print(".", end="", flush=True)

    print()

    # Create visualization
    plt.figure(figsize=(10, 8))
    plt.quiver(X, Y, U, V, width=0.002)

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

    # Plot formatting
    title = (
        f'Maximum error {errors["mtr"]["max"] * 1e3:.3g} mm  '
        + f'Mean error {errors["mtr"]["mean"] * 1e3:.3g} mm  '
        + f'Std dev {errors["mtr"]["std"] * 1e3:.3g} mm'
    )
    plt.title(title)
    plt.xlabel("X position (m)")
    plt.ylabel("Y position (m)")
    plt.show()

    return errors
