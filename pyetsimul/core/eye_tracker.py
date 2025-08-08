"""Eye tracker module.

This module provides the EyeTracker class that represents a complete eye tracking
system with cameras, lights, calibration points, and algorithm functions.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple
from abc import ABC, abstractmethod

from .camera import Camera
from .light import Light
from .eye import Eye
from ..types import Position3D, Point3D, GazePrediction, EyeMeasurement, PupilData


@dataclass
class EyeTracker(ABC):
    """Abstract base class for eye tracking systems.

    Provides unified interface for different eye tracking algorithms.
    Manages cameras, lights, calibration points, and measurement collection.
    Implements common workflow for calibration and gaze estimation.
    """

    # Physical components
    cameras: List[Camera] = field(default_factory=list)
    lights: List[Light] = field(default_factory=list)

    # Calibration points for gaze tracking
    calib_points: List[Position3D] = field(default_factory=list)

    # Algorithm state/parameters
    state: Dict[str, Any] = field(default_factory=dict)

    # Refraction setting
    use_refraction: bool = True

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Algorithm name identifier - must be implemented by subclasses."""
        pass

    def add_camera(self, camera: Camera) -> None:
        """Add a camera to the eye tracker.

        Manages camera collection for multi-camera eye tracking setups.
        """
        self.cameras.append(camera)

    def add_light(self, light: Light) -> None:
        """Add a light to the eye tracker.

        Manages light collection for corneal reflection detection.
        """
        self.lights.append(light)

    def add_calibration_point(self, point: Position3D) -> None:
        """Add a calibration point to the eye tracker.

        Builds calibration grid for gaze tracking accuracy.
        """
        self.calib_points.append(point)

    def set_calibration_points(self, points: List[Position3D]) -> None:
        """Set all calibration points at once.

        Replaces entire calibration grid with new point collection.
        """
        self.calib_points = points.copy()

    def run_calibration(self, eye: Eye) -> "EyeTracker":
        """Run the complete calibration workflow.

        Manages data collection and algorithm-specific calibration.
        Collects measurements at all calibration points and calls algorithm calibration.

        Args:
            eye: Eye object to calibrate with

        Returns:
            Self for method chaining
        """
        calibration_measurements = self._collect_calibration_measurements(eye)
        self.calibrate(calibration_measurements)
        return self

    def _collect_calibration_measurements(self, eye: Eye) -> List[EyeMeasurement]:
        """Helper to collect measurements for each calibration point.

        Gathers calibration data and reports detection failures.
        Ensures reproducible results with fixed random seed.
        """
        measurements = []
        np.random.seed(0)  # For reproducible results

        n_points = len(self.calib_points)
        failed_points = []

        print(f"Collecting calibration data at {n_points} points...")

        for i, calib_point in enumerate(self.calib_points):
            # Make eye look at calibration point
            eye.look_at(calib_point)

            # Take images from all cameras (for now use first camera)
            # TODO: Support multi-camera setups properly
            if not self.cameras:
                raise ValueError("No cameras available for calibration")

            camera_image = self.cameras[0].take_image(eye, self.lights, use_refraction=self.use_refraction)

            # Create pupil data from camera image
            pupil_data = PupilData(boundary_points=camera_image.pupil_boundary, center=camera_image.pupil_center)

            # Create eye measurement
            measurement = EyeMeasurement(
                camera_image=camera_image,
                pupil_data=pupil_data,
                gaze_direction=Point3D(calib_point.x, calib_point.y, calib_point.z),
            )
            measurements.append(measurement)

            # Check for detection failures
            if camera_image.pupil_center is None:
                failed_points.append((i + 1, calib_point, "PUPIL CENTER not detected"))
            elif not camera_image.corneal_reflections or camera_image.corneal_reflections[0] is None:
                failed_points.append((i + 1, calib_point, "CR not detected"))

        # Summary of failed points
        if failed_points:
            print(f"\n⚠️  WARNING: {len(failed_points)}/{n_points} calibration points failed:")
            for point_num, point, reason in failed_points:
                print(f"  Point {point_num} ({point.x * 1000:.0f}mm, {point.z * 1000:.0f}mm): {reason}")
            print(f"  Calibration will proceed with {n_points - len(failed_points)} valid points.\n")

        return measurements

    def estimate_gaze_at(self, eye: Eye, look_at_pos: Point3D) -> Optional[GazePrediction]:
        """Estimate gaze position when eye looks at a target.

        Implements complete gaze estimation pipeline: eye movement → camera → prediction.
        Delegates to algorithm-specific prediction method.

        Args:
            eye: Eye object
            look_at_pos: 3D position where eye should look

        Returns:
            GazePrediction with estimated gaze and intermediate values
        """
        # Make eye look at target position
        target = Position3D(look_at_pos.x, look_at_pos.y, look_at_pos.z)
        eye.look_at(target)

        # For now, use first camera (TODO: support multi-camera)
        camera_image = self.cameras[0].take_image(eye, self.lights, use_refraction=self.use_refraction)

        # Create EyeMeasurement from camera image
        pupil_data = PupilData(boundary_points=camera_image.pupil_boundary, center=camera_image.pupil_center)
        measurement = EyeMeasurement(
            camera_image=camera_image,
            pupil_data=pupil_data,
            timestamp=None,  # Could add timestamp if needed
        )

        # Get gaze prediction - returns None if prediction fails
        return self.predict_gaze(measurement)

    def calculate_gaze_error(self, eye: Eye, look_at_pos: Point3D) -> Tuple[float, float]:
        """Calculate gaze estimation error.

        Evaluates gaze tracking accuracy by comparing prediction to known target.
        Returns error in meters or NaN if estimation fails.

        Args:
            eye: Eye object
            look_at_pos: 3D position where eye should look

        Returns:
            Tuple of (u, v) gaze error in meters, or (NaN, NaN) if estimation fails
        """
        gaze_prediction = self.estimate_gaze_at(eye, look_at_pos)

        if gaze_prediction is not None and gaze_prediction.gaze_point is not None:
            u = gaze_prediction.gaze_point.x - look_at_pos.x
            v = gaze_prediction.gaze_point.y - look_at_pos.y
            return u, v
        else:
            return np.nan, np.nan

    @abstractmethod
    def calibrate(self, calibration_measurements: List[EyeMeasurement]) -> None:
        """Calibrate the eye tracker using collected data.

        Abstract interface for algorithm-specific calibration implementation.
        Each eye tracker type must implement its specific calibration algorithm.

        Args:
            calibration_measurements: List of eye measurements collected at each calibration point
        """
        pass

    def test_calibration_fit(self, eye: Eye) -> List[Tuple[Position3D, Optional[GazePrediction]]]:
        """Test calibrated polynomial by predicting each calibration point.

        Validates calibration quality by testing full pipeline on known targets.
        Tests: target → eye movement → camera → polynomial → prediction.

        Args:
            eye: Eye object to use for measurements

        Returns:
            List of (target_position, prediction) tuples for each calibration point
        """
        results = []

        for target_position in self.calib_points:
            # Make eye look at calibration point
            eye.look_at(target_position)

            # Take fresh camera measurement
            camera_image = self.cameras[0].take_image(eye, self.lights, use_refraction=self.use_refraction)

            # Create measurement from camera image
            pupil_data = PupilData(boundary_points=camera_image.pupil_boundary, center=camera_image.pupil_center)
            measurement = EyeMeasurement(
                camera_image=camera_image,
                pupil_data=pupil_data,
                timestamp=None,
            )

            # Use calibrated polynomial to predict gaze
            prediction = self.predict_gaze(measurement)
            results.append((target_position, prediction))

        return results

    @abstractmethod
    def predict_gaze(self, measurement: EyeMeasurement) -> Optional[GazePrediction]:
        """Predict gaze position from eye measurement.

        Abstract interface for algorithm-specific gaze prediction implementation.
        Each eye tracker type must implement its specific gaze prediction algorithm.

        Args:
            measurement: EyeMeasurement containing pupil and corneal reflection data

        Returns:
            GazePrediction with estimated gaze position or None if prediction fails
        """
        pass
