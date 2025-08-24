"""Algorithm comparison strategy for parameter variations."""

import copy
from typing import Dict, List, Any
import numpy as np
from tqdm import tqdm
from tabulate import tabulate

from ...core import Eye, EyeTracker
from ...types import Position3D
from ...geometry.conversions import calculate_angular_error_degrees
from ..core import ParameterVariation, VariationStrategy


class AlgorithmComparisonStrategy(VariationStrategy):
    """Compare multiple algorithms across parameter variations."""

    def __init__(self, algorithms: List[EyeTracker], algorithm_names: List[str] = None):
        """Initialize algorithm comparison strategy.

        Args:
            algorithms: List of EyeTracker instances to compare
            algorithm_names: Optional names for algorithms (defaults to indices)
        """
        if not algorithms:
            raise ValueError("Must provide at least one algorithm")

        self.algorithms = algorithms
        self.algorithm_names = algorithm_names or [f"Algorithm_{i}" for i in range(len(algorithms))]

        if len(self.algorithm_names) != len(self.algorithms):
            raise ValueError("Number of algorithm names must match number of algorithms")

    def execute(self, eye: Eye, variation: ParameterVariation, gaze_target: Position3D = None) -> Dict[str, Any]:
        """Compare all algorithms across parameter variation values.

        Args:
            eye: Eye object for simulation
            variation: Parameter variation to test across
            gaze_target: Target gaze point (for eye position variations)

        Returns:
            Dictionary with comparison results for each algorithm
        """
        # Ensure all algorithms are calibrated
        for i, algorithm in enumerate(self.algorithms):
            if not algorithm.algorithm_state.is_calibrated:
                print(f"Calibrating {self.algorithm_names[i]}...")
                algorithm.run_calibration(eye)

        values = variation.generate_values()
        results = {name: [] for name in self.algorithm_names}

        print(f"Comparing {len(self.algorithms)} algorithms across {len(values)} parameter values...")

        for value in tqdm(values, desc="Testing parameter values"):
            e = copy.deepcopy(eye)

            # Apply parameter variation
            if variation.param_name == "eye_position":
                variation.apply_to_eye(e, value)
                if gaze_target:
                    e.look_at(gaze_target)
                target_point = gaze_target
            elif variation.param_name == "target_position":
                e.look_at(value)  # value is the target position
                target_point = value
            else:
                raise ValueError(f"Unsupported parameter variation: {variation.param_name}")

            # Test each algorithm at this parameter value
            for i, algorithm in enumerate(self.algorithms):
                algorithm_name = self.algorithm_names[i]

                try:
                    # Estimate gaze with this algorithm
                    predicted_gaze = algorithm.estimate_gaze_at(e, target_point)

                    if predicted_gaze is not None and predicted_gaze.gaze_point is not None:
                        predicted_point = Position3D(
                            predicted_gaze.gaze_point.x, predicted_gaze.gaze_point.y, predicted_gaze.gaze_point.z
                        )

                        # Calculate errors
                        error_x = predicted_point.x - target_point.x
                        error_y = predicted_point.y - target_point.y
                        error_z = predicted_point.z - target_point.z
                        euclidean_error = np.sqrt(error_x**2 + error_y**2 + error_z**2)

                        error_angular = calculate_angular_error_degrees(target_point, predicted_point, e.position)

                        result = {
                            "parameter_value": value,
                            "target_point": target_point,
                            "predicted_point": predicted_point,
                            "error_x": error_x,
                            "error_y": error_y,
                            "error_z": error_z,
                            "euclidean_error": euclidean_error,
                            "error_angular": error_angular,
                            "success": True,
                        }
                    else:
                        result = {
                            "parameter_value": value,
                            "target_point": target_point,
                            "predicted_point": None,
                            "error_x": np.nan,
                            "error_y": np.nan,
                            "error_z": np.nan,
                            "euclidean_error": np.nan,
                            "error_angular": np.nan,
                            "success": False,
                        }

                except Exception as e:
                    result = {
                        "parameter_value": value,
                        "target_point": target_point,
                        "predicted_point": None,
                        "error_x": np.nan,
                        "error_y": np.nan,
                        "error_z": np.nan,
                        "euclidean_error": np.nan,
                        "error_angular": np.nan,
                        "success": False,
                        "error": str(e),
                    }

                results[algorithm_name].append(result)

        # Calculate summary statistics for each algorithm
        summary = self._calculate_summary_statistics(results)

        return {
            "algorithms": self.algorithm_names,
            "parameter_variation": variation.param_name,
            "total_tests": len(values),
            "detailed_results": results,
            "summary_statistics": summary,
        }

    def _calculate_summary_statistics(self, results: Dict[str, List[Dict]]) -> Dict[str, Dict]:
        """Calculate summary statistics for each algorithm."""
        summary = {}

        for algorithm_name, algorithm_results in results.items():
            # Extract valid errors
            valid_euclidean = [
                r["euclidean_error"] for r in algorithm_results if r["success"] and not np.isnan(r["euclidean_error"])
            ]
            valid_angular = [
                r["error_angular"] for r in algorithm_results if r["success"] and not np.isnan(r["error_angular"])
            ]

            success_rate = sum(1 for r in algorithm_results if r["success"]) / len(algorithm_results)

            if valid_euclidean:
                summary[algorithm_name] = {
                    "success_rate": success_rate,
                    "position_error": {
                        "mean": np.mean(valid_euclidean),
                        "std": np.std(valid_euclidean),
                        "median": np.median(valid_euclidean),
                        "max": np.max(valid_euclidean),
                        "min": np.min(valid_euclidean),
                    },
                    "angular_error": {
                        "mean": np.mean(valid_angular) if valid_angular else np.nan,
                        "std": np.std(valid_angular) if valid_angular else np.nan,
                        "median": np.median(valid_angular) if valid_angular else np.nan,
                        "max": np.max(valid_angular) if valid_angular else np.nan,
                        "min": np.min(valid_angular) if valid_angular else np.nan,
                    },
                    "total_points": len(algorithm_results),
                    "successful_points": len(valid_euclidean),
                }
            else:
                summary[algorithm_name] = {
                    "success_rate": success_rate,
                    "position_error": {"mean": np.nan, "std": np.nan, "median": np.nan, "max": np.nan, "min": np.nan},
                    "angular_error": {"mean": np.nan, "std": np.nan, "median": np.nan, "max": np.nan, "min": np.nan},
                    "total_points": len(algorithm_results),
                    "successful_points": 0,
                }

        return summary

    def print_results(self, results: Dict[str, Any], test_name: str) -> None:
        """Print algorithm comparison results in tabulated format.

        Args:
            results: Results dictionary from execute()
            test_name: Name of the test for display
        """
        headers = ["Algorithm", "Error (mm)", "Error (deg)", "Tests"]

        # Add sub-headers for error statistics
        sub_headers = ["", "Mean ± Std (Max | Median)", "Mean ± Std (Max | Median)", "Success/Total"]

        data = []
        for algorithm_name in results["algorithms"]:
            stats = results["summary_statistics"][algorithm_name]

            # Format position error statistics
            if not np.isnan(stats["position_error"]["mean"]):
                position_error = (
                    f"{stats['position_error']['mean'] * 1000:.2f} ± "
                    f"{stats['position_error']['std'] * 1000:.2f} "
                    f"({stats['position_error']['max'] * 1000:.2f} | "
                    f"{stats['position_error']['median'] * 1000:.2f})"
                )
            else:
                position_error = "N/A"

            # Format angular error statistics
            if not np.isnan(stats["angular_error"]["mean"]):
                angular_error = (
                    f"{stats['angular_error']['mean']:.3f} ± "
                    f"{stats['angular_error']['std']:.3f} "
                    f"({stats['angular_error']['max']:.3f} | "
                    f"{stats['angular_error']['median']:.3f})"
                )
            else:
                angular_error = "N/A"

            data.append(
                [
                    algorithm_name,
                    position_error,
                    angular_error,
                    f"{stats['successful_points']}/{stats['total_points']}",
                ]
            )

        # Print the table with descriptive title
        print(f"\n{test_name} ({results['total_tests']} test points):")
        print(tabulate([sub_headers] + data, headers=headers, tablefmt="grid"))
        print()
