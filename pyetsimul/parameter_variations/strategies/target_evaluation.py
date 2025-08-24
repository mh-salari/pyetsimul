"""Target position evaluation strategy for parameter variations."""

import copy
from typing import Dict
import numpy as np
from tqdm import tqdm

from ...core import Eye, EyeTracker
from ...types import Position3D
from ...geometry.conversions import calculate_angular_error_degrees
from ...evaluation.analysis_utils import calculate_error_statistics, plot_error_vectors, plot_error_vectors_3d
from ..core import ParameterVariation, VariationStrategy


class TargetPositionEvaluationStrategy(VariationStrategy):
    """Evaluates gaze accuracy across target position variations."""

    def __init__(self, observer_position: Position3D = None, enable_plotting: bool = True):
        self.observer_position = observer_position
        self.enable_plotting = enable_plotting

    def execute(self, eye: Eye, et: EyeTracker, variation: ParameterVariation) -> Dict[str, Dict[str, float]]:
        """Run accuracy evaluation across all target position values."""

        if not et.algorithm_state.is_calibrated:
            raise ValueError("Eye tracker must be calibrated before evaluation")

        e = copy.deepcopy(eye)

        # Set observer position
        if self.observer_position is not None:
            e.position = self.observer_position
        elif e.position is None:
            raise ValueError("Observer position must be provided or eye must have a position set")

        target_positions = variation.generate_values()
        results = []

        for target_position in tqdm(target_positions, desc="Gaze point analysis"):
            predicted_gaze = et.estimate_gaze_at(e, target_position)

            if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
                predicted_point = Position3D(
                    predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z
                )

                error_x = predicted_point.x - target_position.x
                error_y = predicted_point.y - target_position.y
                error_z = predicted_point.z - target_position.z
                error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)

                error_angular = calculate_angular_error_degrees(target_position, predicted_point, e.position)

                results.append(
                    {
                        "actual": target_position,
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
                        "actual": target_position,
                        "predicted": None,
                        "error_x": np.nan,
                        "error_y": np.nan,
                        "error_z": np.nan,
                        "error_3d": np.nan,
                        "error_angular": np.nan,
                    }
                )

        statistics = self._calculate_statistics(results)

        # Add plotting if enabled
        if self.enable_plotting:
            self._plot_results(variation, results, statistics, et, e.position)

        return statistics

    def _calculate_statistics(self, results: list) -> Dict[str, Dict[str, float]]:
        """Calculate error statistics from results."""
        valid_3d = [r["error_3d"] for r in results if not np.isnan(r["error_3d"])]
        valid_angular = [r["error_angular"] for r in results if not np.isnan(r["error_angular"])]

        if valid_3d:
            return {
                "mtr": {
                    "max": max(valid_3d),
                    "mean": np.mean(valid_3d),
                    "std": np.std(valid_3d),
                    "median": np.median(valid_3d),
                },
                "deg": {
                    "max": max(valid_angular) if valid_angular else np.nan,
                    "mean": np.mean(valid_angular) if valid_angular else np.nan,
                    "std": np.std(valid_angular) if valid_angular else np.nan,
                    "median": np.median(valid_angular) if valid_angular else np.nan,
                },
            }
        else:
            return {
                "mtr": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
                "deg": {"max": np.nan, "mean": np.nan, "std": np.nan, "median": np.nan},
            }

    def _plot_results(self, variation, results, statistics, et, observer_pos):
        """Generate plots based on target position variation dimensions."""
        # Determine which dimensions vary
        varies_x = variation.dx[0] != variation.dx[1]
        varies_y = variation.dy[0] != variation.dy[1]
        varies_z = variation.dz[0] != variation.dz[1]
        varying_dims = sum([varies_x, varies_y, varies_z])

        if varying_dims == 2:
            self._plot_2d_results(variation, results, statistics, et, observer_pos)
        elif varying_dims == 1:
            self._plot_1d_results(variation, results, statistics)
        elif varying_dims == 3:
            self._plot_3d_results(variation, results, statistics)

    def _plot_2d_results(self, variation, results, statistics, et, observer_pos):
        """Plot 2D target position results."""
        plane_info = et.plane_info

        # Determine grid structure
        x_grid_size, y_grid_size, z_grid_size = variation.grid_size
        varies_x = variation.dx[0] != variation.dx[1]
        varies_y = variation.dy[0] != variation.dy[1]
        varies_z = variation.dz[0] != variation.dz[1]

        if varies_x and varies_y:
            coord1_size, coord2_size = x_grid_size, y_grid_size
        elif varies_x and varies_z:
            coord1_size, coord2_size = x_grid_size, z_grid_size
        elif varies_y and varies_z:
            coord1_size, coord2_size = y_grid_size, z_grid_size
        else:
            return

        # Generate coordinate arrays
        dx_min, dx_max = variation.dx
        dy_min, dy_max = variation.dy
        dz_min, dz_max = variation.dz

        x_values = (
            [variation.grid_center.x + dx_min]
            if dx_min == dx_max
            else np.linspace(variation.grid_center.x + dx_min, variation.grid_center.x + dx_max, x_grid_size)
        )
        y_values = (
            [variation.grid_center.y + dy_min]
            if dy_min == dy_max
            else np.linspace(variation.grid_center.y + dy_min, variation.grid_center.y + dy_max, y_grid_size)
        )
        z_values = (
            [variation.grid_center.z + dz_min]
            if dz_min == dz_max
            else np.linspace(variation.grid_center.z + dz_min, variation.grid_center.z + dz_max, z_grid_size)
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
                if idx < len(results) and not np.isnan(results[idx]["error_3d"]):
                    actual = results[idx]["actual"]
                    predicted = results[idx]["predicted"]

                    # Extract plane coordinates
                    actual_coord1, actual_coord2 = plane_info.extract_2d_coords(actual)
                    predicted_coord1, predicted_coord2 = plane_info.extract_2d_coords(predicted)

                    # Calculate error in plane coordinates
                    U[j, i] = predicted_coord1 - actual_coord1
                    V[j, i] = predicted_coord2 - actual_coord2
                    errs_deg[j, i] = results[idx]["error_angular"]
                else:
                    U[j, i] = np.nan
                    V[j, i] = np.nan
                    errs_deg[j, i] = np.nan
                idx += 1

        # Calculate 2D-specific error statistics for compatibility
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

    def _plot_1d_results(self, variation, results, statistics):
        """Plot 1D target position results using 3D plotting."""
        # Determine which axis varies
        varies_x = variation.dx[0] != variation.dx[1]
        varies_y = variation.dy[0] != variation.dy[1]
        varies_z = variation.dz[0] != variation.dz[1]

        if varies_x:
            varying_axis = "X"
        elif varies_y:
            varying_axis = "Y"
        elif varies_z:
            varying_axis = "Z"
        else:
            raise ValueError("No varying dimension found")

        # Extract positions and error vectors for valid results
        positions = []
        error_vectors = []
        angular_errors = []

        for result in results:
            if not np.isnan(result["error_3d"]):
                actual = result["actual"]
                positions.append([actual.x, actual.y, actual.z])
                error_vectors.append([result["error_x"], result["error_y"], result["error_z"]])
                angular_errors.append(result["error_angular"])

        if positions:
            positions = np.array(positions)
            error_vectors = np.array(error_vectors)
            angular_errors = np.array(angular_errors)

            plot_error_vectors_3d(
                positions=positions,
                error_vectors=error_vectors,
                angular_errors=angular_errors,
                errors=statistics,
                title_prefix=f"Gaze Point Analysis (1D - {varying_axis} axis)",
                convert_to_mm=True,
                position_labels=("Gaze X", "Gaze Y", "Gaze Z"),
            )

    def _plot_3d_results(self, variation, results, statistics):
        """Plot 3D target position results."""
        # Extract positions and error vectors for valid results
        positions = []
        error_vectors = []
        angular_errors = []

        for result in results:
            if not np.isnan(result["error_3d"]):
                actual = result["actual"]
                positions.append([actual.x, actual.y, actual.z])
                error_vectors.append([result["error_x"], result["error_y"], result["error_z"]])
                angular_errors.append(result["error_angular"])

        if positions:
            positions = np.array(positions)
            error_vectors = np.array(error_vectors)
            angular_errors = np.array(angular_errors)

            plot_error_vectors_3d(
                positions=positions,
                error_vectors=error_vectors,
                angular_errors=angular_errors,
                errors=statistics,
                title_prefix="Gaze Point Analysis (3D)",
                convert_to_mm=True,
                position_labels=("Gaze X", "Gaze Y", "Gaze Z"),
            )
