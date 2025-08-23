"""Eye movement accuracy analysis."""

import copy
from typing import Optional, Dict
import numpy as np

from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import plot_error_vectors, calculate_error_statistics
from ..core import Eye, EyeTracker
from ..types import Position3D
from ..experimental_designs import EyeMovement


def accuracy_over_eye_positions(
    et: EyeTracker,
    eye: Eye,
    eye_movement: EyeMovement,
    gaze_target: Optional[Position3D] = None,
) -> Dict[str, Dict[str, float]]:
    """Computes gaze error at different eye positions.

    Evaluates gaze tracking robustness across different observer positions.
    Tests calibration stability by varying eye position while keeping gaze target fixed.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)
        eye_movement: EyeMovement specifying eye position pattern
        gaze_target: Fixed gaze target (default: from eye_movement)

    Returns:
        Dictionary with error statistics (mean, max, std, median for both mm and degrees)
    """
    if gaze_target is None:
        gaze_target = eye_movement.gaze_target

    e = copy.deepcopy(eye)

    if not et.algorithm_state.is_calibrated:
        raise ValueError(
            "Eye tracker must be calibrated before running accuracy analysis. Call et.run_calibration(eye) first."
        )

    eye_movement.validate_design()
    eye_positions = eye_movement.generate_eye_positions()

    print(f"Running eye movement analysis with {len(eye_positions)} positions...")

    # Calculate error for all eye positions
    results = []
    for eye_position in eye_positions:
        # Move eye to test position
        e.position = eye_position

        # Get predicted gaze for fixed target
        predicted_gaze = et.estimate_gaze_at(e, gaze_target)

        if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
            predicted_point = Position3D(
                predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z
            )

            # Error vector
            error_x = predicted_point.x - gaze_target.x
            error_y = predicted_point.y - gaze_target.y
            error_z = predicted_point.z - gaze_target.z
            error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)

            # Angular error
            error_angular = calculate_angular_error_degrees(gaze_target, predicted_point, eye_position)

            results.append(
                {
                    "eye_position": eye_position,
                    "gaze_target": gaze_target,
                    "predicted": predicted_point,
                    "error_x": error_x,
                    "error_y": error_y,
                    "error_z": error_z,
                    "error_3d": error_3d,
                    "error_angular": error_angular,
                }
            )
        else:
            results.append(
                {
                    "eye_position": eye_position,
                    "gaze_target": gaze_target,
                    "predicted": None,
                    "error_x": np.nan,
                    "error_y": np.nan,
                    "error_z": np.nan,
                    "error_3d": np.nan,
                    "error_angular": np.nan,
                }
            )

        print(".", end="", flush=True)

    print()

    # Error statistics
    valid_errors = [r["error_3d"] for r in results if not np.isnan(r["error_3d"])]
    valid_angular_errors = [r["error_angular"] for r in results if not np.isnan(r["error_angular"])]

    if valid_errors:
        errors = {
            "mtr": {
                "max": max(valid_errors),
                "mean": np.mean(valid_errors),
                "std": np.std(valid_errors),
                "median": np.median(valid_errors),
            },
            "deg": {
                "max": max(valid_angular_errors) if valid_angular_errors else np.nan,
                "mean": np.mean(valid_angular_errors) if valid_angular_errors else np.nan,
                "std": np.std(valid_angular_errors) if valid_angular_errors else np.nan,
                "median": np.median(valid_angular_errors) if valid_angular_errors else np.nan,
            },
        }
        print(f"Maximum error {errors['mtr']['max'] * 1e3:.3g} mm")
        print(f"Mean error {errors['mtr']['mean'] * 1e3:.3g} mm")
        print(f"Standard deviation {errors['mtr']['std'] * 1e3:.3g} mm")
    else:
        print("No valid predictions found")
        errors = {
            "mtr": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
            "deg": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
        }

    # Dimension-specific plotting
    _plot_results_by_dimension(eye_movement, results, errors, et)

    return errors


def _plot_results_by_dimension(eye_movement, results, errors, et):
    """Plot results using appropriate visualization for movement pattern dimensions."""
    varies_x = eye_movement.dx[0] != eye_movement.dx[1]
    varies_y = eye_movement.dy[0] != eye_movement.dy[1]
    varies_z = eye_movement.dz[0] != eye_movement.dz[1]
    varying_dims = sum([varies_x, varies_y, varies_z])

    if varying_dims == 1:
        _plot_1d_results(eye_movement, results, errors)
    elif varying_dims == 2:
        _plot_2d_results(eye_movement, results, errors, et)
    elif varying_dims == 3:
        _plot_3d_results(eye_movement, results, errors)
    else:
        print("Warning: No varying dimensions detected, skipping plots")


def _plot_1d_results(eye_movement, results, errors):
    """Plot 1D results with error vectors along the line."""
    print("1D plotting: TODO - implement vector style similar to 2D")


def _plot_2d_results(eye_movement, results, errors, et):
    """Plot 2D results using existing vector plot approach."""
    plane_info = et.plane_info

    # Determine grid structure
    x_grid_size, y_grid_size, z_grid_size = eye_movement.grid_size
    varies_x = eye_movement.dx[0] != eye_movement.dx[1]
    varies_y = eye_movement.dy[0] != eye_movement.dy[1]
    varies_z = eye_movement.dz[0] != eye_movement.dz[1]

    # Generate coordinate arrays for the varying dimensions
    dx_min, dx_max = eye_movement.dx
    dy_min, dy_max = eye_movement.dy
    dz_min, dz_max = eye_movement.dz

    x_values = (
        [eye_movement.eye_center.x + dx_min]
        if dx_min == dx_max
        else np.linspace(eye_movement.eye_center.x + dx_min, eye_movement.eye_center.x + dx_max, x_grid_size)
    )
    y_values = (
        [eye_movement.eye_center.y + dy_min]
        if dy_min == dy_max
        else np.linspace(eye_movement.eye_center.y + dy_min, eye_movement.eye_center.y + dy_max, y_grid_size)
    )
    z_values = (
        [eye_movement.eye_center.z + dz_min]
        if dz_min == dz_max
        else np.linspace(eye_movement.eye_center.z + dz_min, eye_movement.eye_center.z + dz_max, z_grid_size)
    )

    # Use the varying dimensions for coordinate arrays
    if varies_x and varies_y:
        coord1_size, coord2_size = x_grid_size, y_grid_size
        coord1_values, coord2_values = x_values, y_values
    elif varies_x and varies_z:
        coord1_size, coord2_size = x_grid_size, z_grid_size
        coord1_values, coord2_values = x_values, z_values
    elif varies_y and varies_z:
        coord1_size, coord2_size = y_grid_size, z_grid_size
        coord1_values, coord2_values = y_values, z_values
    else:
        print("Warning: 2D plotting requires exactly 2 varying dimensions")
        return

    print(f"DEBUG: coord1_size={coord1_size}, coord2_size={coord2_size}")
    print(f"DEBUG: len(coord1_values)={len(coord1_values)}, len(coord2_values)={len(coord2_values)}")
    print(f"DEBUG: len(results)={len(results)}")

    # Convert results to 2D arrays for plotting
    U = np.zeros((coord2_size, coord1_size))
    V = np.zeros((coord2_size, coord1_size))
    errs_deg = np.zeros((coord2_size, coord1_size))

    idx = 0
    for i in range(coord1_size):
        for j in range(coord2_size):
            if idx < len(results) and results[idx]["predicted"] is not None:
                gaze_target = results[idx]["gaze_target"]
                predicted = results[idx]["predicted"]

                # Extract plane coordinates
                target_coord1, target_coord2 = plane_info.extract_2d_coords(gaze_target)
                predicted_coord1, predicted_coord2 = plane_info.extract_2d_coords(predicted)

                # Calculate error in plane coordinates (error = predicted - target)
                U[j, i] = predicted_coord1 - target_coord1
                V[j, i] = predicted_coord2 - target_coord2

                # Angular error
                errs_deg[j, i] = results[idx]["error_angular"]
            else:
                U[j, i] = np.nan
                V[j, i] = np.nan
                errs_deg[j, i] = np.nan
            idx += 1

    # Recalculate 2D-specific error statistics for compatibility with existing plot function
    errors_2d = calculate_error_statistics(U, V, errs_deg)

    # Use existing plot function
    plot_error_vectors(
        coord1_values,
        coord2_values,
        U,
        V,
        errors_2d,
        title_prefix="Eye Movement Analysis",
        convert_to_mm=True,
        xlabel=f"Eye {plane_info.primary_axis.upper()} position (mm)",
        ylabel=f"Eye {plane_info.secondary_axis.upper()} position (mm)",
    )


def _plot_3d_results(eye_movement, results, errors):
    """Plot 3D results with error vectors in 3D space."""
    print("3D plotting: TODO - implement 3D vector visualization similar to 2D style")
