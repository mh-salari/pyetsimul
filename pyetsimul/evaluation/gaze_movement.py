"""Gaze movement accuracy analysis."""

import copy
from typing import Optional, Dict
import numpy as np
from tqdm import tqdm

from ..geometry.conversions import calculate_angular_error_degrees
from .analysis_utils import plot_error_vectors, plot_error_vectors_3d, calculate_error_statistics
from ..core import Eye, EyeTracker
from ..types import Position3D
from ..experimental_designs import GazeMovement


def accuracy_over_gaze_points(
    et: EyeTracker,
    eye: Eye,
    gaze_movement: GazeMovement,
    observer_pos_test: Optional[Position3D] = None,
) -> Dict[str, Dict[str, float]]:
    """Computes gaze error at different gaze target points.

    Returns error statistics and generates dimension-appropriate plots.

    Args:
        et: Eye tracker structure
        eye: Pre-configured Eye object (required)
        gaze_movement: GazeMovement specifying gaze target pattern
        observer_pos_test: Observer position for testing (default: same as calib position)

    Returns:
        Dictionary with error statistics (mean, max, std, median for both mm and degrees)
    """
    if observer_pos_test is None:
        if eye.position is not None:
            observer_pos_test = eye.position
        else:
            raise ValueError("Observer position must be provided or eye must have a position set")

    e = copy.deepcopy(eye)

    if not et.algorithm_state.is_calibrated:
        raise ValueError(
            "Eye tracker must be calibrated before running accuracy analysis. Call et.run_calibration(eye) first."
        )

    gaze_movement.validate_design()
    target_positions = gaze_movement.generate_target_positions()

    e.position = observer_pos_test

    # Calculate error for all targets
    results = []
    for actual_point in tqdm(target_positions, desc="Gaze point analysis"):
        predicted_gaze = et.estimate_gaze_at(e, actual_point)

        if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
            predicted_point = Position3D(
                predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z
            )

            # Error vector
            error_x = predicted_point.x - actual_point.x
            error_y = predicted_point.y - actual_point.y
            error_z = predicted_point.z - actual_point.z
            error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)

            # Angular error
            error_angular = calculate_angular_error_degrees(actual_point, predicted_point, observer_pos_test)

            results.append(
                {
                    "actual": actual_point,
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
                    "actual": actual_point,
                    "predicted": None,
                    "error_x": np.nan,
                    "error_y": np.nan,
                    "error_z": np.nan,
                    "error_3d": np.nan,
                    "error_angular": np.nan,
                }
            )

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
    _plot_results_by_dimension(gaze_movement, results, errors, et, observer_pos_test)

    return errors


def _plot_results_by_dimension(gaze_movement, results, errors, et, observer_pos_test):
    """Select and execute plotting function based on varying dimensions."""
    varies_x = gaze_movement.dx[0] != gaze_movement.dx[1]
    varies_y = gaze_movement.dy[0] != gaze_movement.dy[1]
    varies_z = gaze_movement.dz[0] != gaze_movement.dz[1]
    varying_dims = sum([varies_x, varies_y, varies_z])

    if varying_dims == 1:
        _plot_1d_results(gaze_movement, results, errors)
    elif varying_dims == 2:
        _plot_2d_results(gaze_movement, results, errors, et, observer_pos_test)
    elif varying_dims == 3:
        _plot_3d_results(gaze_movement, results, errors)
    else:
        print("Warning: No varying dimensions detected, skipping plots")


def _plot_1d_results(gaze_movement, results, errors):
    """Plot 1D results with error vectors in 3D space."""
    # Determine which axis varies
    varies_x = gaze_movement.dx[0] != gaze_movement.dx[1]
    varies_y = gaze_movement.dy[0] != gaze_movement.dy[1]
    varies_z = gaze_movement.dz[0] != gaze_movement.dz[1]

    if varies_x:
        varying_axis = "X"
    elif varies_y:
        varying_axis = "Y"
    else:  # must be Z if we're in 1D function
        varying_axis = "Z"

    # Extract positions and error vectors for valid results
    positions = []
    error_vectors = []
    angular_errors = []

    for result in results:
        if result["predicted"] is not None:
            # Target position
            actual = result["actual"]
            positions.append([actual.x, actual.y, actual.z])

            # Error vector (predicted - actual)
            error_vectors.append([result["error_x"], result["error_y"], result["error_z"]])

            # Angular error
            angular_errors.append(result["error_angular"])

    if not positions:
        print("Warning: No valid data for 1D plotting")
        return

    positions = np.array(positions)
    error_vectors = np.array(error_vectors)
    angular_errors = np.array(angular_errors)

    # Use the 3D plotting function for 1D data
    plot_error_vectors_3d(
        positions=positions,
        error_vectors=error_vectors,
        angular_errors=angular_errors,
        errors=errors,
        title_prefix=f"Gaze Point Analysis (1D - {varying_axis} axis)",
        convert_to_mm=True,
        position_labels=("Gaze X", "Gaze Y", "Gaze Z"),
    )


def _plot_2d_results(gaze_movement, results, errors, et, observer_pos_test):
    """Plot 2D results using existing vector plot approach."""
    plane_info = et.plane_info

    # Determine grid structure
    x_grid_size, y_grid_size, z_grid_size = gaze_movement.grid_size
    varies_x = gaze_movement.dx[0] != gaze_movement.dx[1]
    varies_y = gaze_movement.dy[0] != gaze_movement.dy[1]
    varies_z = gaze_movement.dz[0] != gaze_movement.dz[1]

    if varies_x and varies_y:
        coord1_size, coord2_size = x_grid_size, y_grid_size
    elif varies_x and varies_z:
        coord1_size, coord2_size = x_grid_size, z_grid_size
    elif varies_y and varies_z:
        coord1_size, coord2_size = y_grid_size, z_grid_size
    else:
        print("Warning: 2D plotting requires exactly 2 varying dimensions")
        return

    # Generate coordinate arrays
    dx_min, dx_max = gaze_movement.dx
    dy_min, dy_max = gaze_movement.dy
    dz_min, dz_max = gaze_movement.dz

    x_values = (
        [gaze_movement.grid_center.x + dx_min]
        if dx_min == dx_max
        else np.linspace(gaze_movement.grid_center.x + dx_min, gaze_movement.grid_center.x + dx_max, x_grid_size)
    )
    y_values = (
        [gaze_movement.grid_center.y + dy_min]
        if dy_min == dy_max
        else np.linspace(gaze_movement.grid_center.y + dy_min, gaze_movement.grid_center.y + dy_max, y_grid_size)
    )
    z_values = (
        [gaze_movement.grid_center.z + dz_min]
        if dz_min == dz_max
        else np.linspace(gaze_movement.grid_center.z + dz_min, gaze_movement.grid_center.z + dz_max, z_grid_size)
    )

    # Map coordinates based on plane_info
    coord_map = {"x": x_values, "y": y_values, "z": z_values}
    coord1_values = coord_map[plane_info.primary_axis.lower()]
    coord2_values = coord_map[plane_info.secondary_axis.lower()]

    # Convert results to 2D arrays for plotting
    U = np.zeros((coord2_size, coord1_size))
    V = np.zeros((coord2_size, coord1_size))
    errs_deg = np.zeros((coord2_size, coord1_size))

    idx = 0
    for j in range(coord2_size):
        for i in range(coord1_size):
            if idx < len(results) and results[idx]["predicted"] is not None:
                actual = results[idx]["actual"]
                predicted = results[idx]["predicted"]

                # Extract plane coordinates
                actual_coord1, actual_coord2 = plane_info.extract_2d_coords(actual)
                predicted_coord1, predicted_coord2 = plane_info.extract_2d_coords(predicted)

                # Calculate error in plane coordinates
                U[j, i] = predicted_coord1 - actual_coord1
                V[j, i] = predicted_coord2 - actual_coord2

                # Angular error
                errs_deg[j, i] = calculate_angular_error_degrees(actual, predicted, observer_pos_test)
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
        title_prefix="Gaze Point Analysis",
        convert_to_mm=True,
        xlabel=f"{plane_info.primary_axis.upper()} position (mm)",
        ylabel=f"{plane_info.secondary_axis.upper()} position (mm)",
    )


def _plot_3d_results(gaze_movement, results, errors):
    """Plot 3D results with error vectors in 3D space."""
    # Extract positions and error vectors for valid results
    positions = []
    error_vectors = []
    angular_errors = []

    for result in results:
        if result["predicted"] is not None:
            # Target position
            actual = result["actual"]
            positions.append([actual.x, actual.y, actual.z])

            # Error vector (predicted - actual)
            error_vectors.append([result["error_x"], result["error_y"], result["error_z"]])

            # Angular error
            angular_errors.append(result["error_angular"])

    if not positions:
        print("Warning: No valid data for 3D plotting")
        return

    positions = np.array(positions)
    error_vectors = np.array(error_vectors)
    angular_errors = np.array(angular_errors)

    # Use the new 3D plotting function
    plot_error_vectors_3d(
        positions=positions,
        error_vectors=error_vectors,
        angular_errors=angular_errors,
        errors=errors,
        title_prefix="Gaze Point Analysis (3D)",
        convert_to_mm=True,
        position_labels=("Gaze X", "Gaze Y", "Gaze Z"),
    )
