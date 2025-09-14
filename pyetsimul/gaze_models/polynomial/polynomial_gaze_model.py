"""Polynomial gaze model eye tracker.

Pupil-CR tracker with polynomial gaze model.
"""

import time

import numpy as np

from pyetsimul.core import Camera, EyeTracker, Light
from pyetsimul.types.algorithms import GazePrediction, PolynomialGazeModelState
from pyetsimul.types.geometry import Point3D, Position3D
from pyetsimul.types.imaging import EyeMeasurement

from ...geometry.plane_detection import PlaneInfo, detect_calibration_plane, summarize_plane_detection
from .polynomials import get_polynomial


class PolynomialGazeModel(EyeTracker):
    """Polynomial gaze model Pupil-CR eye tracker.

    Maps pupil-corneal reflection vectors to screen coordinates using polynomial functions.
    Supports various polynomial types from eye tracking literature.
    """

    def __init__(self, polynomial: str, **kwargs: object) -> None:
        """Initialize polynomial gaze model tracker with polynomial specification.

        Sets up polynomial function and algorithm state for gaze tracking.

        Args:
            polynomial: Polynomial type to use
            **kwargs: Arguments passed to parent EyeTracker

        """
        super().__init__(**kwargs)
        self.algorithm_state = PolynomialGazeModelState()
        self.polynomial_name = polynomial
        self.polynomial_func = get_polynomial(polynomial)
        self.plane_info = None

    @property
    def algorithm_name(self) -> str:
        """Algorithm name identifier."""
        return "polynomial"

    @classmethod
    def create(
        cls,
        cameras: list,
        lights: list,
        calib_points: list[Position3D],
        polynomial: str,
        use_refraction: bool = True,
    ) -> "PolynomialGazeModel":
        """Create polynomial gaze model eye tracker setup.

        Factory method for configuring complete polynomial gaze model tracker with all components.

        Args:
            cameras: List of Camera objects
            lights: List of Light objects
            calib_points: List of calibration target positions
            polynomial: Polynomial type to use
            use_refraction: Whether to use refraction model (default True)

        Returns:
            PolynomialGazeModel: Configured polynomial gaze model eye tracker

        """
        return cls(
            polynomial=polynomial,
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
        )

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibration function for pupil-CR polynomial gaze model.

        Detects calibration plane and performs polynomial least squares calibration.
        Automatically handles different polynomial feature types.

        Args:
            calibration_measurements: List of EyeMeasurement objects from calibration

        """
        # Detect calibration plane for coordinate system
        self.plane_info = detect_calibration_plane(self.calib_points)
        print(summarize_plane_detection(self.calib_points, self.plane_info))

        # Determine polynomial feature type for appropriate calibration
        test_features = self.polynomial_func(0.0, 0.0)

        if test_features.uses_different_xy_features:
            self._calibrate_different_xy(calibration_measurements)
        else:
            self._calibrate_same_xy(calibration_measurements)

    def _calibrate_same_xy(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibrate with polynomial using same features for both coordinates.

        Uses single polynomial for both X and Y gaze components.
        """
        # Determine feature vector size from first valid measurement
        feature_size = None
        for measurement in calibration_measurements:
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )
            if pc is not None and cr is not None:
                pcr = pc - cr
                poly_features = self.polynomial_func(pcr.x, pcr.y)
                feature_size = poly_features.feature_count
                break

        if feature_size is None:
            raise ValueError("No valid calibration data found")

        # Build feature matrix for all calibration points
        feature_matrix = np.zeros((feature_size, len(self.calib_points)))

        for i, measurement in enumerate(calibration_measurements):
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )

            if pc is not None and cr is not None:
                pcr = pc - cr
                poly_features = self.polynomial_func(pcr.x, pcr.y)
                feature_matrix[:, i] = poly_features.features

        # Map 3D calibration points to 2D plane coordinates
        calib_coords_2d = [self.plane_info.extract_2d_coords(pt) for pt in self.calib_points]
        calib_points_array = np.array(calib_coords_2d).T

        # Solve least squares: calib_points = A @ feature_matrix → A = calib_points @ pinv(feature_matrix)
        calibration_matrix = calib_points_array @ np.linalg.pinv(feature_matrix)

        # Store coefficients for both X and Y gaze components
        self.algorithm_state.x_coefficients = calibration_matrix[0:1, :].flatten()
        self.algorithm_state.y_coefficients = calibration_matrix[1:2, :].flatten()
        self.algorithm_state.is_calibrated = True

    def _calibrate_different_xy(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibrate with polynomial using different features for each coordinate.

        Uses independent polynomials for X and Y gaze components.
        """
        # Determine polynomial structure from first valid measurement
        poly_features = None
        for measurement in calibration_measurements:
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )
            if pc is not None and cr is not None:
                pcr = pc - cr
                poly_features = self.polynomial_func(pcr.x, pcr.y)
                break

        if poly_features is None:
            raise ValueError("No valid calibration data found")

        # Build separate feature matrices for X and Y coordinates
        if poly_features.features.dtype == object:
            # Handle object arrays (mixed-length coordinates)
            coord_x_size = len(poly_features.features[0])
            coord_y_size = len(poly_features.features[1])
            feature_matrix_x = np.zeros((coord_x_size, len(self.calib_points)))
            feature_matrix_y = np.zeros((coord_y_size, len(self.calib_points)))
        else:
            # Handle regular 2D arrays
            _, feature_size = poly_features.features.shape
            feature_matrix_x = np.zeros((feature_size, len(self.calib_points)))
            feature_matrix_y = np.zeros((feature_size, len(self.calib_points)))

        for i, measurement in enumerate(calibration_measurements):
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )

            if pc is not None and cr is not None:
                pcr = pc - cr
                poly_features = self.polynomial_func(pcr.x, pcr.y)
                if poly_features.features.dtype == object:
                    # Handle object arrays (mixed-length coordinates)
                    feature_matrix_x[:, i] = poly_features.features[0]  # X coordinate features
                    feature_matrix_y[:, i] = poly_features.features[1]  # Y coordinate features
                else:
                    # Handle regular 2D arrays
                    feature_matrix_x[:, i] = poly_features.features[0, :]  # X coordinate features
                    feature_matrix_y[:, i] = poly_features.features[1, :]  # Y coordinate features

        # Map 3D calibration points to 2D plane coordinates
        calib_coords_2d = [self.plane_info.extract_2d_coords(pt) for pt in self.calib_points]
        calib_points_array = np.array(calib_coords_2d).T

        # Solve separate least squares problems for X and Y coordinates
        coeff_x = calib_points_array[0:1, :] @ np.linalg.pinv(feature_matrix_x)  # X coordinate coefficients
        coeff_y = calib_points_array[1:2, :] @ np.linalg.pinv(feature_matrix_y)  # Y coordinate coefficients

        # Store separate coefficients for X and Y gaze components
        self.algorithm_state.x_coefficients = coeff_x.flatten()
        self.algorithm_state.y_coefficients = coeff_y.flatten()
        self.algorithm_state.is_calibrated = True

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction:
        """Predict gaze from pupil-corneal reflection vector.

        Applies calibrated polynomial model to predict screen gaze coordinates.
        Handles missing data with appropriate confidence scoring.

        Args:
            measurement: EyeMeasurement object containing pupil and corneal reflection data

        Returns:
            GazePrediction: Structured prediction result.

        """
        start_time = time.time()

        # Extract pupil center and corneal reflection from measurement
        pc = measurement.pupil_data.center
        cr = measurement.camera_image.corneal_reflections[0] if measurement.camera_image.corneal_reflections else None
        polynomial_name = self.polynomial_name

        # Prepare intermediate results for debugging and analysis
        intermediate_results = {"pc": pc, "cr": cr, "polynomial_name": polynomial_name}

        if pc is not None and cr is not None:
            # Calculate pupil-corneal reflection vector
            pcr_vector = pc - cr
            intermediate_results["pcr_vector"] = pcr_vector

            # Generate polynomial features and predict gaze coordinates
            poly_features = self.polynomial_func(pcr_vector.x, pcr_vector.y)
            intermediate_results["feature_vector"] = poly_features.features
            intermediate_results["polynomial_info"] = poly_features

            # Apply calibrated polynomial model with plane coordinate system
            gaze_point = poly_features.predict(
                self.algorithm_state.x_coefficients, self.algorithm_state.y_coefficients, self.plane_info
            )

            confidence = 1.0  # High confidence for successful polynomial prediction
        else:
            # Handle missing pupil or corneal reflection data
            gaze_point = Point3D(0.0, 0.0, 0.0)
            confidence = 0.0

        processing_time = time.time() - start_time

        return GazePrediction(
            gaze_point=gaze_point,
            confidence=confidence,
            algorithm_name=f"polynomial_{polynomial_name}",
            processing_time=processing_time,
            intermediate_results=intermediate_results,
        )

    def serialize(self) -> dict:
        """Serialize eye tracker to dictionary.

        Saves all eye tracker parameters, calibration data, and algorithm state
        for later restoration and validation.

        Returns:
            dict: Complete eye tracker state including hardware configuration

        """
        return {
            "polynomial_name": self.polynomial_name,
            "algorithm_state": self.algorithm_state.serialize(),
            "plane_info": self.plane_info.serialize() if self.plane_info else None,
            "cameras": [camera.serialize() for camera in self.cameras],
            "lights": [light.serialize() for light in self.lights],
            "calib_points": [point.serialize() for point in self.calib_points],
            "use_refraction": self.use_refraction,
            "use_legacy_look_at": self.use_legacy_look_at,
            "state": self.state,  # Additional state from parent class
        }

    @classmethod
    def deserialize(cls, data: dict) -> "PolynomialGazeModel":
        """Deserialize eye tracker from dictionary.

        Restores complete eye tracker state including calibration and hardware configuration.
        The restored tracker is ready to use without recalibration.

        Args:
            data: Dictionary from serialize() method

        Returns:
            PolynomialGazeModel: Fully configured and calibrated eye tracker

        """
        # Deserialize hardware components
        cameras = [Camera.deserialize(cam_data) for cam_data in data["cameras"]]
        lights = [Light.deserialize(light_data) for light_data in data["lights"]]
        calib_points = [Position3D.deserialize(pt_data) for pt_data in data["calib_points"]]

        # Create tracker instance
        tracker = cls(
            polynomial=data["polynomial_name"],
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=data["use_refraction"],
        )

        # Restore algorithm state and plane info
        tracker.algorithm_state = PolynomialGazeModelState.deserialize(data["algorithm_state"])
        if data["plane_info"]:
            tracker.plane_info = PlaneInfo.deserialize(data["plane_info"])

        # Restore parent class state and legacy mode
        tracker.use_legacy_look_at = data["use_legacy_look_at"]
        tracker.state = data["state"]

        return tracker
