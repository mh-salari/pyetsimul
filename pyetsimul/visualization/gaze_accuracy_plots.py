"""Gaze accuracy visualization utilities.

Provides specialized plotting functionality for gaze accuracy analysis results.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..simulation.core import ParameterVariation

import matplotlib.pyplot as plt
import numpy as np

from ..core import EyeTracker
from ..evaluation.gaze_accuracy import GazeAccuracyResult
from ..simulation.core import TargetVariation
from ..types import Position3D
from .analysis_plots import plot_error_vectors_2d, plot_error_vectors_3d


def detect_variation_plane(variation: "ParameterVariation") -> tuple[str, str, list[float], list[float]]:
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

    tolerance = 1e-10
    x_varies = abs(grid.dx[1] - grid.dx[0]) > tolerance
    y_varies = abs(grid.dy[1] - grid.dy[0]) > tolerance
    z_varies = abs(grid.dz[1] - grid.dz[0]) > tolerance

    varying_axes = []
    if x_varies:
        varying_axes.append(("x", grid.dx))
    if y_varies:
        varying_axes.append(("y", grid.dy))
    if z_varies:
        varying_axes.append(("z", grid.dz))

    if len(varying_axes) == 0:
        return "x", "y", [0.0, 0.0], [0.0, 0.0]
    if len(varying_axes) == 1:
        axis, variation_range = varying_axes[0]
        return axis, axis, variation_range, [0.0, 0.0]
    # Two or more axes vary - use first two
    return varying_axes[0][0], varying_axes[1][0], varying_axes[0][1], varying_axes[1][1]


def extract_variation_coords(position: Position3D, primary_axis: str, secondary_axis: str) -> tuple[float, float]:
    """Extract 2D coordinates from 3D position based on variation axes."""
    coords = {"x": position.x, "y": position.y, "z": position.z}
    return coords[primary_axis], coords[secondary_axis]


class GazeAccuracyPlotter:
    """Handles plotting for gaze accuracy analysis results."""

    def plot(
        self,
        gaze_result: GazeAccuracyResult,
        eye_tracker: EyeTracker,
        title_prefix: str = "Gaze Accuracy Analysis",
        plot_mode: str = "auto",
        show: bool = True,
    ) -> plt.Figure:
        """Plot gaze accuracy results with flexible 2D/3D visualization.

        Args:
            gaze_result: GazeAccuracyResult instance with analysis data
            eye_tracker: Eye tracker instance (used to access plane_info for 2D plotting)
            title_prefix: Title prefix for the plot
            plot_mode: "2d", "3d", or "auto" (auto chooses 2D for 2D trackers, 3D otherwise)
            show: If True (default), display the figure with plt.show() (blocks until closed).
                  If False, return the figure for saving (fig.savefig()) without displaying.
                  The figure is removed from matplotlib's manager to prevent it from
                  appearing unexpectedly in later plt.show() calls.

        Returns:
            The matplotlib Figure.

        """
        if gaze_result.successful_predictions == 0:
            raise ValueError("No successful predictions to plot")

        if gaze_result.variation is None:
            raise ValueError("No variation information available for plotting")

        if plot_mode == "auto":
            has_plane_info = hasattr(eye_tracker, "plane_info") and eye_tracker.plane_info is not None
            plot_mode = "2d" if has_plane_info else "3d"

        if plot_mode == "2d":
            fig = self._plot_2d(gaze_result, eye_tracker, title_prefix)
        else:
            fig = GazeAccuracyPlotter._plot_3d(gaze_result, title_prefix)

        if show:
            plt.show()

        plt.close(fig)

        return fig

    @staticmethod
    def _plot_3d(gaze_result: GazeAccuracyResult, title_prefix: str) -> plt.Figure:
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

        return plot_error_vectors_3d(
            positions=positions,
            error_vectors=error_vectors,
            angular_errors=angular_errors,
            errors=gaze_result.error_stats,
            title_prefix=title_prefix,
            convert_to_mm=True,
            position_labels=("X position", "Y position", "Z position"),
        )

    def _plot_2d(self, gaze_result: GazeAccuracyResult, eye_tracker: EyeTracker, title_prefix: str) -> plt.Figure:
        """Create 2D visualization using natural coordinates from plane detection."""
        if not hasattr(eye_tracker, "plane_info") or eye_tracker.plane_info is None:
            print("Warning: Eye tracker lacks plane info, using 3D visualization")
            return self._plot_3d(gaze_result, title_prefix)

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
            raise ValueError("No valid data points for 2D plotting")

        positions_array = np.array(positions_2d)
        error_vectors_array = np.array(error_vectors_2d)

        x = positions_array[:, 0]
        y = positions_array[:, 1]
        u = error_vectors_array[:, 0]
        v = error_vectors_array[:, 1]

        if coordinate_system == "target_plane":
            primary_label = f"Target {pos_primary_axis.upper()} position"
            secondary_label = f"Target {pos_secondary_axis.upper()} position"
        else:
            primary_label = f"Observer {pos_primary_axis.upper()} position"
            secondary_label = f"Observer {pos_secondary_axis.upper()} position"

        return plot_error_vectors_2d(
            X=x,
            Y=y,
            U=u,
            V=v,
            errors=gaze_result.error_stats,
            title_prefix=title_prefix,
            convert_to_mm=True,
            xlabel=f"{primary_label} (mm)",
            ylabel=f"{secondary_label} (mm)",
        )
