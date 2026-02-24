"""Eye tracker module.

This module provides the EyeTracker class that represents a complete eye tracking
system with cameras, lights, calibration points, and algorithm functions.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pyetsimul.log import info, table, warning

from ..types import EyeMeasurement, GazePrediction, Point3D, Position3D, PupilData
from ..utils import validate_eye_camera_setup
from .camera import Camera
from .eye import Eye
from .light import Light


@dataclass
class EyeTracker(ABC):
    """Abstract base class for eye tracking systems.

    Provides unified interface for different eye tracking algorithms.
    Manages cameras, lights, calibration points, and measurement collection.
    Implements common workflow for calibration and gaze estimation.
    """

    # Physical components
    cameras: list[Camera] = field(default_factory=list)
    lights: list[Light] = field(default_factory=list)

    # Calibration points for gaze tracking
    calib_points: list[Position3D] = field(default_factory=list)

    # Algorithm state/parameters
    state: dict[str, Any] = field(default_factory=dict)

    # Refraction setting
    use_refraction: bool = True

    # Pupil center calculation method: "ellipse" (default) or "center_of_mass"
    pupil_center_method: str = "ellipse"

    # Calibration diagnostics: list of (point_number, position, reason) for failed points
    failed_calibration_points: list[tuple[int, Position3D, str]] = field(default_factory=list)

    # MATLAB compatibility mode for eye rotation
    use_legacy_look_at: bool = False

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Algorithm name identifier - must be implemented by subclasses."""

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

    def set_calibration_points(self, points: list[Position3D]) -> None:
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
        # Validate eye-camera setup if cameras are present
        if self.cameras:
            for camera in self.cameras:
                validate_eye_camera_setup(eye.rest_orientation, camera.trans.get_rotation())

        calibration_measurements = self._collect_calibration_measurements(eye)
        self.calibrate(calibration_measurements)
        return self

    def _collect_calibration_measurements(self, eye: Eye) -> list[EyeMeasurement]:
        """Helper to collect measurements for each calibration point.

        Gathers calibration data and reports detection failures.
        """
        measurements = []

        n_points = len(self.calib_points)
        failed_points = []

        info(f"Collecting calibration data at {n_points} points...")

        for i, calib_point in enumerate(self.calib_points):
            # Make eye look at calibration point
            eye.look_at(calib_point, legacy=self.use_legacy_look_at)

            # Take images from all cameras (currently uses first camera)
            # TODO: Multi-camera support requires algorithm-specific implementation
            if not self.cameras:
                raise ValueError("No cameras available for calibration")

            camera_image = self.cameras[0].take_image(
                eye, self.lights, use_refraction=self.use_refraction, center_method=self.pupil_center_method
            )

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

        # Store and report failed points
        self.failed_calibration_points = failed_points
        if failed_points:
            warning(f"\n{len(failed_points)}/{n_points} calibration points failed:")
            for point_num, point, reason in failed_points:
                warning(f"  Point {point_num} ({point.x:.0f}mm, {point.z:.0f}mm): {reason}")
            warning(f"  Calibration will proceed with {n_points - len(failed_points)} valid points.\n")

        return measurements

    def estimate_gaze_at(self, eye: Eye, look_at_pos: Point3D) -> GazePrediction | None:
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
        eye.look_at(target, legacy=self.use_legacy_look_at)

        # Use first camera (TODO: multi-camera support is algorithm-dependent)
        camera_image = self.cameras[0].take_image(
            eye, self.lights, use_refraction=self.use_refraction, center_method=self.pupil_center_method
        )

        # Create EyeMeasurement from camera image
        pupil_data = PupilData(boundary_points=camera_image.pupil_boundary, center=camera_image.pupil_center)
        measurement = EyeMeasurement(
            camera_image=camera_image,
            pupil_data=pupil_data,
            timestamp=None,  # Could add timestamp if needed
        )

        # Get gaze prediction - returns None if prediction fails
        return self.predict_gaze(measurement)

    def calculate_gaze_error(self, eye: Eye, look_at_pos: Point3D) -> tuple[float, float]:
        """Calculate gaze estimation error.

        Evaluates gaze tracking accuracy by comparing prediction to known target.
        Returns error in mm or NaN if estimation fails.

        Args:
            eye: Eye object
            look_at_pos: 3D position where eye should look

        Returns:
            Tuple of (u, v) gaze error in mm, or (NaN, NaN) if estimation fails

        """
        gaze_prediction = self.estimate_gaze_at(eye, look_at_pos)

        if gaze_prediction is not None and gaze_prediction.gaze_point is not None:
            u = gaze_prediction.gaze_point.x - look_at_pos.x
            v = gaze_prediction.gaze_point.y - look_at_pos.y
            return u, v
        return np.nan, np.nan

    @abstractmethod
    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibrate the eye tracker using collected data.

        Abstract interface for algorithm-specific calibration implementation.
        Each eye tracker type must implement its specific calibration algorithm.

        Args:
            calibration_measurements: List of eye measurements collected at each calibration point

        """

    def test_calibration_fit(self, eye: Eye) -> list[tuple[Position3D, GazePrediction | None]]:
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
            eye.look_at(target_position, legacy=self.use_legacy_look_at)

            # Take fresh camera measurement
            camera_image = self.cameras[0].take_image(
                eye, self.lights, use_refraction=self.use_refraction, center_method=self.pupil_center_method
            )

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
    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction | None:
        """Predict gaze position from eye measurement.

        Abstract interface for algorithm-specific gaze prediction implementation.
        Each eye tracker type must implement its specific gaze prediction algorithm.

        Args:
            measurement: EyeMeasurement containing pupil and corneal reflection data

        Returns:
            GazePrediction with estimated gaze position or None if prediction fails

        """

    def __str__(self) -> str:
        """Basic string representation of the eye tracker."""
        try:
            calibrated = self.algorithm_state.is_calibrated
        except AttributeError:
            calibrated = False
        return f"{self.__class__.__name__}(algorithm={self.algorithm_name}, cameras={len(self.cameras)}, lights={len(self.lights)}, calibrated={calibrated})"

    def pprint(self, eye: "Eye | None" = None) -> None:
        """Print detailed eye tracker parameters in a formatted table.

        Args:
            eye: Optional Eye instance to include eye position in the summary.

        """
        # Check calibration status
        try:
            calibrated = self.algorithm_state.is_calibrated
        except AttributeError:
            calibrated = False

        calib_points = len(self.calib_points) if self.calib_points else 0

        data = []

        # Add eye position if provided
        if eye is not None:
            pos = eye.position
            data.append(["Eye position (mm)", f"({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})"])

        data.extend([
            ["Algorithm", self.algorithm_name],
            ["Cameras", str(len(self.cameras))],
            ["Lights", str(len(self.lights))],
            ["Calibration points", str(calib_points)],
            ["Calibration status", "Calibrated" if calibrated else "Not calibrated"],
            ["Use refraction", "Yes" if self.use_refraction else "No"],
            ["Legacy look_at mode", "Yes" if self.use_legacy_look_at else "No"],
        ])

        # Add algorithm-specific configuration
        if hasattr(self, "algorithm_state") and hasattr(self.algorithm_state, "config"):
            config = self.algorithm_state.config
            if hasattr(config, "method"):
                data.append(["Algorithm method", config.method])
            if hasattr(config, "degree"):
                data.append(["Polynomial degree", str(config.degree)])

        # Add camera details
        if self.cameras:
            for i, cam in enumerate(self.cameras):
                pos = cam.position
                data.append([
                    f"Camera {i + 1} position (mm)",
                    f"({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})",
                ])

        # Add light details
        if self.lights:
            for i, light in enumerate(self.lights):
                pos = light.position
                data.append([
                    f"Light {i + 1} position (mm)",
                    f"({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})",
                ])

        headers = ["Parameter", "Value"]
        info(f"{self.__class__.__name__} Configuration:")
        table(data, headers=headers, tablefmt="grid")
