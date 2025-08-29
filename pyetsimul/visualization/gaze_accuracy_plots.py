"""Gaze accuracy visualization utilities.

Provides specialized plotting functionality for gaze accuracy analysis results.
"""

import numpy as np

from .analysis_plots import plot_error_vectors_2d, plot_error_vectors_3d
from ..experiment_framework.data_generation.core import TargetVariation


def detect_variation_plane(variation):
    """Detect which 2D plane a parameter variation occurs in.

    Analyzes variation ranges to determine which two axes actually vary.

    Args:
        variation: Parameter variation object with dx, dy, dz attributes

    Returns:
        tuple: (primary_axis, secondary_axis, primary_range, secondary_range)
               where axes are 'x', 'y', 'z' and ranges are [min, max]
    """
    if not hasattr(variation, "grid") or not hasattr(variation.grid, "dx"):
        raise ValueError("Variation object must have grid with dx, dy, dz attributes")

    grid = variation.grid

    TOLERANCE = 1e-10
    x_varies = abs(grid.dx[1] - grid.dx[0]) > TOLERANCE
    y_varies = abs(grid.dy[1] - grid.dy[0]) > TOLERANCE
    z_varies = abs(grid.dz[1] - grid.dz[0]) > TOLERANCE

    varying_axes = []
    if x_varies:
        varying_axes.append(("x", grid.dx))
    if y_varies:
        varying_axes.append(("y", grid.dy))
    if z_varies:
        varying_axes.append(("z", grid.dz))

    if len(varying_axes) == 0:
        return "x", "y", [0.0, 0.0], [0.0, 0.0]
    elif len(varying_axes) == 1:
        axis, variation_range = varying_axes[0]
        return axis, axis, variation_range, [0.0, 0.0]
    else:
        # Two or more axes vary - use first two
        return varying_axes[0][0], varying_axes[1][0], varying_axes[0][1], varying_axes[1][1]


def extract_variation_coords(position, primary_axis, secondary_axis):
    """Extract 2D coordinates from 3D position based on variation axes."""
    coords = {"x": position.x, "y": position.y, "z": position.z}
    return coords[primary_axis], coords[secondary_axis]


class GazeAccuracyPlotter:
    """Handles plotting for gaze accuracy analysis results."""

    def plot(self, gaze_result, eye_tracker, title_prefix: str = "Gaze Accuracy Analysis", plot_mode: str = "auto"):
        """Plot gaze accuracy results with flexible 2D/3D visualization.

        Args:
            gaze_result: GazeAccuracyResult instance with analysis data
            eye_tracker: Eye tracker instance (used to access plane_info for 2D plotting)
            title_prefix: Title prefix for the plot
            plot_mode: "2d", "3d", or "auto" (auto chooses 2D for 2D trackers, 3D otherwise)
        """
        if gaze_result.successful_predictions == 0:
            print("No successful predictions to plot")
            return

        if gaze_result.variation is None:
            print("No variation information available for plotting")
            return

        if plot_mode == "auto":
            has_plane_info = hasattr(eye_tracker, "plane_info") and eye_tracker.plane_info is not None
            plot_mode = "2d" if has_plane_info else "3d"

        if plot_mode == "2d":
            self._plot_2d(gaze_result, eye_tracker, title_prefix)
        else:
            self._plot_3d(gaze_result, eye_tracker, title_prefix)

    def _plot_3d(self, gaze_result, eye_tracker, title_prefix):
        """Create 3D visualization using existing plotting utilities."""
        positions = []
        error_vectors = []
        angular_errors = []

        for data_idx in range(len(gaze_result.ground_truth_points)):
            if gaze_result.predicted_points[data_idx] is not None:
                actual = gaze_result.ground_truth_points[data_idx]
                predicted = gaze_result.predicted_points[data_idx]

                if isinstance(gaze_result.variation, TargetVariation):
                    pos = [actual.x, actual.y, actual.z]
                else:
                    observer = gaze_result.observer_positions[data_idx]
                    pos = [observer.x, observer.y, observer.z]

                error_vec = [predicted.x - actual.x, predicted.y - actual.y, predicted.z - actual.z]

                positions.append(pos)
                error_vectors.append(error_vec)
                angular_errors.append(gaze_result.errors_angular[data_idx])

        positions = np.array(positions)
        error_vectors = np.array(error_vectors)
        angular_errors = np.array(angular_errors)

        plot_error_vectors_3d(
            positions=positions,
            error_vectors=error_vectors,
            angular_errors=angular_errors,
            errors=gaze_result.error_stats,
            title_prefix=title_prefix,
            convert_to_mm=True,
            position_labels=("X position", "Y position", "Z position"),
        )

    def _plot_2d(self, gaze_result, eye_tracker, title_prefix):
        """Create 2D visualization using natural coordinates from plane detection."""
        if not hasattr(eye_tracker, "plane_info") or eye_tracker.plane_info is None:
            print("Warning: Eye tracker lacks plane info, using 3D visualization")
            self._plot_3d(gaze_result, eye_tracker, title_prefix)
            return

        plane_info = eye_tracker.plane_info

        if isinstance(gaze_result.variation, TargetVariation):
            pos_primary_axis = plane_info.primary_axis
            pos_secondary_axis = plane_info.secondary_axis
            coordinate_system = "target_plane"
        else:
            var_primary, var_secondary, _, _ = detect_variation_plane(gaze_result.variation)
            pos_primary_axis = var_primary
            pos_secondary_axis = var_secondary
            coordinate_system = "observer_variation"

        positions_2d = []
        error_vectors_2d = []
        angular_errors = []

        for data_idx in range(len(gaze_result.ground_truth_points)):
            if gaze_result.predicted_points[data_idx] is not None:
                actual = gaze_result.ground_truth_points[data_idx]
                predicted = gaze_result.predicted_points[data_idx]

                if coordinate_system == "target_plane":
                    pos_2d = plane_info.extract_2d_coords(actual)
                else:
                    observer = gaze_result.observer_positions[data_idx]
                    pos_2d = extract_variation_coords(observer, pos_primary_axis, pos_secondary_axis)

                actual_2d = plane_info.extract_2d_coords(actual)
                predicted_2d = plane_info.extract_2d_coords(predicted)
                error_2d = (predicted_2d[0] - actual_2d[0], predicted_2d[1] - actual_2d[1])

                positions_2d.append(pos_2d)
                error_vectors_2d.append(error_2d)
                angular_errors.append(gaze_result.errors_angular[data_idx])

        if not positions_2d:
            print("No valid data points for 2D plotting")
            return

        positions_array = np.array(positions_2d)
        error_vectors_array = np.array(error_vectors_2d)

        X = positions_array[:, 0]
        Y = positions_array[:, 1]
        U = error_vectors_array[:, 0]
        V = error_vectors_array[:, 1]

        if coordinate_system == "target_plane":
            primary_label = f"Target {pos_primary_axis.upper()} position"
            secondary_label = f"Target {pos_secondary_axis.upper()} position"
        else:
            primary_label = f"Observer {pos_primary_axis.upper()} position"
            secondary_label = f"Observer {pos_secondary_axis.upper()} position"

        plot_error_vectors_2d(
            X=X,
            Y=Y,
            U=U,
            V=V,
            errors=gaze_result.error_stats,
            angular_errors=angular_errors,
            title_prefix=title_prefix,
            convert_to_mm=True,
            xlabel=f"{primary_label} (mm)",
            ylabel=f"{secondary_label} (mm)",
        )
