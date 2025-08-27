"""Accuracy evaluation strategy for parameter variations."""

import copy
from typing import Dict
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from ...core import Eye, EyeTracker
from ...types import Position3D
from ...geometry.conversions import calculate_angular_error_degrees
from ...evaluation.analysis_utils import plot_error_vectors, plot_error_vectors_3d
from ..core import ParameterVariation, VariationStrategy


class ParameterVariationResults:
    """Parameter variation results with printing method."""

    def __init__(self, errors: Dict[str, Dict[str, float]], test_type: str, test_count: int):
        self.errors = errors
        self.test_type = test_type
        self.test_count = test_count

    def __str__(self) -> str:
        """Basic string representation of parameter variation results."""
        mean_error_mm = self.errors["mtr"]["mean"] * 1e3
        mean_error_deg = self.errors["deg"]["mean"]
        return f"ParameterVariationResults({self.test_type}, {self.test_count} tests, mean_error={mean_error_mm:.2f}mm / {mean_error_deg:.3f}°)"

    def pprint(self, title: str = "Parameter Variation Results") -> None:
        """Print formatted parameter variation error statistics."""
        print(f"\n{title}:")

        headers = ["Statistic", "Error (mm)", "Error (degrees)"]
        data = [
            ["Max", f"{self.errors['mtr']['max'] * 1e3:.4f}", f"{self.errors['deg']['max']:.4f}"],
            ["Mean", f"{self.errors['mtr']['mean'] * 1e3:.4f}", f"{self.errors['deg']['mean']:.4f}"],
            ["Std", f"{self.errors['mtr']['std'] * 1e3:.4f}", f"{self.errors['deg']['std']:.4f}"],
            ["Median", f"{self.errors['mtr']['median'] * 1e3:.4f}", f"{self.errors['deg']['median']:.4f}"],
        ]

        print(tabulate(data, headers=headers, tablefmt="grid"))


class EyePositionEvaluationStrategy(VariationStrategy):
    """Evaluates gaze accuracy across eye position variations."""

    def __init__(self, gaze_target: Position3D, enable_plotting: bool = True):
        self.gaze_target = gaze_target
        self.enable_plotting = enable_plotting

    def execute(self, eye: Eye, et: EyeTracker, variation: ParameterVariation) -> Dict[str, Dict[str, float]]:
        """Run accuracy evaluation across all parameter values."""

        if not et.algorithm_state.is_calibrated:
            raise ValueError("Eye tracker must be calibrated before evaluation")

        e = copy.deepcopy(eye)
        values = variation.generate_values()
        results = []

        for value in tqdm(values, desc=f"Evaluating {variation.param_name}"):
            variation.apply_to_eye(e, value)

            predicted_gaze = et.estimate_gaze_at(e, self.gaze_target)

            if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
                predicted_point = Position3D(
                    predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z
                )

                error_x = predicted_point.x - self.gaze_target.x
                error_y = predicted_point.y - self.gaze_target.y
                error_z = predicted_point.z - self.gaze_target.z
                error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)

                error_angular = calculate_angular_error_degrees(self.gaze_target, predicted_point, e.position)

                results.append(
                    {
                        "eye_position": value,
                        "gaze_target": self.gaze_target,
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
                        "eye_position": value,
                        "gaze_target": self.gaze_target,
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
            self._plot_results(variation, results, statistics, et)

        return ParameterVariationResults(statistics, "eye_position", len(values))

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

    def _plot_results(self, variation, results, statistics, et):
        """Generate plots based on eye position variation dimensions."""
        # Determine which dimensions vary
        varies_x = variation.dx[0] != variation.dx[1]
        varies_y = variation.dy[0] != variation.dy[1]
        varies_z = variation.dz[0] != variation.dz[1]
        varying_dims = sum([varies_x, varies_y, varies_z])

        if varying_dims == 2:
            self._plot_2d_results(variation, results, statistics, et)
        elif varying_dims == 1:
            self._plot_1d_results(variation, results, statistics)
        elif varying_dims == 3:
            self._plot_3d_results(variation, results, statistics)

    def _plot_2d_results(self, variation, results, statistics, et):
        """Plot 2D eye position results."""
        # This is adapted from the old eye_movement plotting logic
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

        # Generate coordinate arrays for the varying dimensions
        dx_min, dx_max = variation.dx
        dy_min, dy_max = variation.dy
        dz_min, dz_max = variation.dz

        x_values = (
            [variation.center.x + dx_min]
            if dx_min == dx_max
            else np.linspace(variation.center.x + dx_min, variation.center.x + dx_max, x_grid_size)
        )
        y_values = (
            [variation.center.y + dy_min]
            if dy_min == dy_max
            else np.linspace(variation.center.y + dy_min, variation.center.y + dy_max, y_grid_size)
        )
        z_values = (
            [variation.center.z + dz_min]
            if dz_min == dz_max
            else np.linspace(variation.center.z + dz_min, variation.center.z + dz_max, z_grid_size)
        )

        # Use the varying dimensions for coordinate arrays
        if varies_x and varies_y:
            coord1_values, coord2_values = x_values, y_values
        elif varies_x and varies_z:
            coord1_values, coord2_values = x_values, z_values
        elif varies_y and varies_z:
            coord1_values, coord2_values = y_values, z_values

        # Convert results to 2D arrays for plotting
        U = np.zeros((coord2_size, coord1_size))
        V = np.zeros((coord2_size, coord1_size))
        errs_deg = np.zeros((coord2_size, coord1_size))

        idx = 0
        for i in range(coord1_size):
            for j in range(coord2_size):
                if idx < len(results) and not np.isnan(results[idx]["error_3d"]):
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

        # Use existing plot function
        plot_error_vectors(
            coord1_values,
            coord2_values,
            U,
            V,
            statistics,
            title_prefix="Eye Movement Analysis",
            convert_to_mm=True,
            xlabel=f"Eye {plane_info.primary_axis.upper()} position (mm)",
            ylabel=f"Eye {plane_info.secondary_axis.upper()} position (mm)",
        )

    def _plot_1d_results(self, variation, results, statistics):
        """Plot 1D eye position results using 3D plotting."""
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

        # Extract eye positions and error vectors for valid results
        positions = []
        error_vectors = []
        angular_errors = []

        for result in results:
            if not np.isnan(result["error_3d"]):
                # Eye position
                eye_pos = result["eye_position"]
                positions.append([eye_pos.x, eye_pos.y, eye_pos.z])

                # Error vector (predicted gaze - target gaze)
                error_vectors.append([result["error_x"], result["error_y"], result["error_z"]])

                # Angular error
                angular_errors.append(result["error_angular"])

        if positions:
            positions = np.array(positions)
            error_vectors = np.array(error_vectors)
            angular_errors = np.array(angular_errors)

            # Use the 3D plotting function for 1D data
            plot_error_vectors_3d(
                positions=positions,
                error_vectors=error_vectors,
                angular_errors=angular_errors,
                errors=statistics,
                title_prefix=f"Eye Movement Analysis (1D - {varying_axis} axis)",
                convert_to_mm=True,
                position_labels=("Eye X", "Eye Y", "Eye Z"),
            )

    def _plot_3d_results(self, variation, results, statistics):
        """Plot 3D eye position results with error vectors in 3D space."""
        # Extract eye positions and error vectors for valid results
        positions = []
        error_vectors = []
        angular_errors = []

        for result in results:
            if not np.isnan(result["error_3d"]):
                # Eye position
                eye_pos = result["eye_position"]
                positions.append([eye_pos.x, eye_pos.y, eye_pos.z])

                # Error vector (predicted gaze - target gaze)
                error_vectors.append([result["error_x"], result["error_y"], result["error_z"]])

                # Angular error
                angular_errors.append(result["error_angular"])

        if positions:
            positions = np.array(positions)
            error_vectors = np.array(error_vectors)
            angular_errors = np.array(angular_errors)

            # Use the new 3D plotting function
            plot_error_vectors_3d(
                positions=positions,
                error_vectors=error_vectors,
                angular_errors=angular_errors,
                errors=statistics,
                title_prefix="Eye Movement Analysis (3D)",
                convert_to_mm=True,
                position_labels=("Eye X", "Eye Y", "Eye Z"),
            )
