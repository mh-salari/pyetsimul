"""Generic gaze accuracy evaluation that works with any pre-generated dataset."""

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
from tqdm import tqdm

from pyetsimul.log import info, warning

from ..core import EyeTracker
from ..geometry.conversions import calculate_angular_error_degrees
from ..types import CameraImage, EyeMeasurement, Point2D, Point3D, Position3D, PupilData
from .analysis_utils import _compute_stats


@dataclass
class GazeAccuracyResult:
    """Results from gaze accuracy evaluation."""

    errors_3d: list[float]  # 3D distances in mm
    errors_angular: list[float]  # Angular errors in degrees
    predicted_points: list[Position3D | None]  # Predicted gaze points
    ground_truth_points: list[Position3D]  # Ground truth gaze points
    observer_positions: list[Position3D]  # Observer eye positions

    # Statistics
    error_stats: dict[str, dict[str, float]]  # Mean, max, std, median for mtr and deg
    total_measurements: int
    successful_predictions: int

    # For plotting
    variation: Any | None = None  # Store the parameter variation for plotting

    def pprint(self, title: str = "Gaze Accuracy Results") -> None:
        """Print formatted results matching existing style."""
        info(f"\n{title}")
        info("-" * len(title))

        success_rate = (self.successful_predictions / self.total_measurements) * 100
        info(f"Success rate: {self.successful_predictions}/{self.total_measurements} ({success_rate:.1f}%)")

        if self.successful_predictions > 0:
            # Display both mm and degrees together
            max_mm = self.error_stats["mm"]["max"]
            max_deg = self.error_stats["deg"]["max"]
            mean_mm = self.error_stats["mm"]["mean"]
            mean_deg = self.error_stats["deg"]["mean"]
            std_mm = self.error_stats["mm"]["std"]
            std_deg = self.error_stats["deg"]["std"]

            info(f"Mean error {mean_deg:.2f}° ({mean_mm:.2f} mm)")
            info(f"Standard deviation {std_deg:.2f}° ({std_mm:.2f} mm)")
            info(f"Maximum error {max_deg:.2f}° ({max_mm:.2f} mm)")
        else:
            warning("No successful predictions to analyze")


def evaluate_gaze_accuracy(
    eye_tracker: EyeTracker,
    dataset: dict[str, Any],
    description: str = "Evaluating gaze accuracy",
    camera_id: int = 0,
    eye_id: int = 0,
) -> GazeAccuracyResult:
    """Evaluate gaze accuracy using pre-generated dataset.

    This function takes a calibrated eye tracker and a dataset (in-memory dictionary)
    and evaluates accuracy by:
    1. Reconstructing EyeMeasurement objects from stored pupil/glint data
    2. Feeding them directly to the tracker's prediction stage
    3. Comparing predictions against ground truth

    Args:
        eye_tracker: Calibrated eye tracker to evaluate with
        dataset: In-memory dataset from DataGenerationStrategy.execute()
        camera_id: Which camera's data to use (default: 0)
        eye_id: Which eye's data to use (default: 0)
        description: Progress description

    Returns:
        GazeAccuracyResult with errors and statistics

    """
    if not eye_tracker.algorithm_state.is_calibrated:
        raise ValueError("Eye tracker must be calibrated before evaluation")

    # Extract dataset components
    _, measurements, variation, camera_resolution = _extract_dataset_components(dataset, camera_id, eye_id)

    # Process all measurements
    (errors_3d, errors_angular, predicted_points, ground_truth_points, observer_positions, successful_predictions) = (
        _process_measurements(eye_tracker, measurements, camera_resolution, description)
    )

    # Calculate final statistics
    error_stats = _calculate_error_statistics(errors_3d, errors_angular)

    return GazeAccuracyResult(
        errors_3d=errors_3d,
        errors_angular=errors_angular,
        predicted_points=predicted_points,
        ground_truth_points=ground_truth_points,
        observer_positions=observer_positions,
        error_stats=error_stats,
        total_measurements=len(measurements),
        successful_predictions=successful_predictions,
        variation=variation,
    )


def _extract_dataset_components(
    dataset: dict[str, Any], camera_id: int, eye_id: int
) -> tuple[dict, list[dict], dict | None, Point2D]:
    """Extract camera data, measurements, and variation from dataset."""
    camera_data = dataset["data"]["cameras"][camera_id]
    eye_data = camera_data["eyes"][eye_id]
    measurements = eye_data["measurements"]

    # Extract variation information for plotting (if available)
    variation = dataset.get("parameter_variation")

    # Extract camera resolution from dataset
    camera_params = dataset["data"]["cameras"][camera_id]["camera_parameters"]
    camera_resolution = Point2D(x=int(camera_params["resolution"]["x"]), y=int(camera_params["resolution"]["y"]))

    return camera_data, measurements, variation, camera_resolution


def _process_measurements(
    eye_tracker: EyeTracker, measurements: list[dict], camera_resolution: Point2D, description: str
) -> tuple[list[float], list[float], list[Position3D | None], list[Point3D], list[Position3D], int]:
    """Process all measurements and calculate errors."""
    errors_3d = []
    errors_angular = []
    predicted_points = []
    ground_truth_points = []
    observer_positions = []
    successful_predictions = 0

    for measurement in tqdm(measurements, desc=description, leave=False):
        # Extract ground truth data
        ground_truth, eye_position = _extract_ground_truth_data(measurement)
        if ground_truth is None:
            continue
        ground_truth = cast("Point3D", ground_truth)
        eye_position = cast("Position3D", eye_position)

        ground_truth_points.append(ground_truth)
        observer_positions.append(eye_position)

        # Process single measurement
        prediction = _process_single_measurement(measurement, camera_resolution, eye_tracker)

        # Calculate errors and update results
        _update_results_with_prediction(
            prediction, ground_truth, eye_position, predicted_points, errors_3d, errors_angular
        )

        if prediction is not None and prediction.gaze_point is not None:
            successful_predictions += 1

    return errors_3d, errors_angular, predicted_points, ground_truth_points, observer_positions, successful_predictions


def _extract_ground_truth_data(measurement: dict) -> tuple[Point3D | None, Position3D | None]:
    """Extract ground truth gaze target and eye position from measurement."""
    if measurement["gaze_target"] is None:
        return None, None

    ground_truth = Point3D(
        measurement["gaze_target"]["x"], measurement["gaze_target"]["y"], measurement["gaze_target"]["z"]
    )
    eye_position = Position3D(
        measurement["eye_state"]["position"]["x"],
        measurement["eye_state"]["position"]["y"],
        measurement["eye_state"]["position"]["z"],
    )

    return ground_truth, eye_position


def _process_single_measurement(
    measurement: dict, camera_resolution: Point2D, eye_tracker: EyeTracker
) -> Position3D | None:
    """Process a single measurement and return gaze prediction."""
    # Extract measurement data
    pupil_center = measurement["pupil_center"]
    pupil_boundary = measurement["pupil_boundary"]
    glint_positions = measurement["corneal_reflections"]

    if pupil_center is None or any(g is None for g in glint_positions):
        return None

    # Reconstruct EyeMeasurement object from stored data
    boundary_points = None
    if pupil_boundary:
        # Convert to numpy array format expected by PupilData
        boundary_points = np.array([[p[0], p[1]] for p in pupil_boundary]).T  # 2xM matrix

    pupil_data = PupilData(center=Point2D(pupil_center[0], pupil_center[1]), boundary_points=boundary_points)

    # Create CameraImage from stored data
    corneal_reflections = [Point2D(g[0], g[1]) if g is not None else None for g in glint_positions]
    camera_image = CameraImage(
        corneal_reflections=corneal_reflections,
        pupil_boundary=None,  # Not needed for prediction
        pupil_center=pupil_data.center,
        resolution=camera_resolution,
        glint_sizes_px=measurement.get("glint_sizes_px"),
    )

    eye_measurement = EyeMeasurement(camera_image=camera_image, pupil_data=pupil_data)

    # Use eye tracker's prediction method directly (bypassing simulation)
    return eye_tracker.predict_gaze(eye_measurement)


def _update_results_with_prediction(
    prediction: Position3D | None,
    ground_truth: Point3D,
    eye_position: Position3D,
    predicted_points: list,
    errors_3d: list,
    errors_angular: list,
) -> None:
    """Update result lists with prediction outcome."""
    if prediction is not None and prediction.gaze_point is not None:
        predicted_point = prediction.gaze_point
        predicted_points.append(predicted_point)

        # Calculate 3D error
        error_3d = np.sqrt(
            (predicted_point.x - ground_truth.x) ** 2
            + (predicted_point.y - ground_truth.y) ** 2
            + (predicted_point.z - ground_truth.z) ** 2
        )
        errors_3d.append(error_3d)

        # Calculate angular error
        error_angular = calculate_angular_error_degrees(ground_truth, predicted_point, eye_position)
        errors_angular.append(error_angular)
    else:
        predicted_points.append(None)
        errors_3d.append(np.nan)
        errors_angular.append(np.nan)


def _calculate_error_statistics(errors_3d: list[float], errors_angular: list[float]) -> dict[str, dict[str, float]]:
    """Calculate error statistics from 3D and angular errors."""
    valid_errors_3d = [e for e in errors_3d if not np.isnan(e)]
    valid_errors_angular = [e for e in errors_angular if not np.isnan(e)]

    nan_stats = {"mean": np.nan, "max": np.nan, "std": np.nan, "median": np.nan}

    if not valid_errors_3d:
        return {"mm": nan_stats, "deg": nan_stats}

    return {
        "mm": _compute_stats(np.array(valid_errors_3d)),
        "deg": _compute_stats(np.array(valid_errors_angular)) if valid_errors_angular else nan_stats,
    }
