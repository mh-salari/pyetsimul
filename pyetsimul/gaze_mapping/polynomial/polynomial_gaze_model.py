"""Polynomial gaze model eye tracker.

Pupil-CR tracker with polynomial gaze model.
"""

import time

import numpy as np

from pyetsimul.core import Camera, EyeTracker, Light
from pyetsimul.log import info
from pyetsimul.types.algorithms import GazePrediction
from pyetsimul.types.geometry import Point2D, Position3D
from pyetsimul.types.imaging import EyeMeasurement

from ...geometry.plane_detection import PlaneInfo, detect_calibration_plane, summarize_plane_detection
from .polynomial_features import PolynomialFeatures
from .polynomial_state import PolynomialGazeModelState
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
        pupil_center_method: str = "ellipse",
    ) -> "PolynomialGazeModel":
        """Create polynomial gaze model eye tracker setup.

        Factory method for configuring complete polynomial gaze model tracker with all components.

        Args:
            cameras: List of Camera objects
            lights: List of Light objects
            calib_points: List of calibration target positions
            polynomial: Polynomial type to use
            use_refraction: Whether to use refraction model (default True)
            pupil_center_method: Pupil center method: "ellipse", "convex_hull" or "center_of_mass"

        Returns:
            PolynomialGazeModel: Configured polynomial gaze model eye tracker

        """
        return cls(
            polynomial=polynomial,
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
            pupil_center_method=pupil_center_method,
        )

    @staticmethod
    def _pcr_vectors(measurement: EyeMeasurement) -> list[Point2D] | None:
        """P-CR vectors of a measurement: the pupil center minus each corneal reflection.

        Returns None if the pupil center or any corneal reflection is missing.
        """
        pc = measurement.pupil_data.center
        corneal_reflections = measurement.camera_image.corneal_reflections
        if pc is None or not corneal_reflections or any(cr is None for cr in corneal_reflections):
            return None
        return [pc - cr for cr in corneal_reflections]

    def _glint_features(self, pcr_vectors: list[Point2D]) -> list[PolynomialFeatures]:
        """Polynomial features of each P-CR vector, one entry per glint.

        The same polynomial is applied to every glint, so a single-glint polynomial also works with
        several light sources.
        """
        return [self.polynomial_func(pcr_vector.x, pcr_vector.y) for pcr_vector in pcr_vectors]

    @staticmethod
    def _concatenate_same_xy(glint_features: list[PolynomialFeatures]) -> np.ndarray:
        """Feature vector shared by both gaze coordinates, concatenated across the glints.

        For example, with Cerrolaza [x², y², xy, x, y, 1] and 2 glints, the
        concatenated feature vector becomes:
        [x1², y1², x1y1, x1, y1, 1, x2², y2², x2y2, x2, y2, 1]
        """
        return np.concatenate([features.features for features in glint_features])

    @staticmethod
    def _concatenate_different_xy(glint_features: list[PolynomialFeatures]) -> tuple[np.ndarray, np.ndarray]:
        """Feature vectors of the X and the Y gaze coordinate, each concatenated across the glints."""
        return (
            np.concatenate([features.features[0] for features in glint_features]),
            np.concatenate([features.features[1] for features in glint_features]),
        )

    def _compute_glint_features(self, measurement: EyeMeasurement) -> np.ndarray | None:
        """Feature vector of a measurement for a polynomial with the same features for X and Y.

        Returns None if the pupil center or any corneal reflection is missing.
        """
        pcr_vectors = self._pcr_vectors(measurement)
        if pcr_vectors is None:
            return None
        return self._concatenate_same_xy(self._glint_features(pcr_vectors))

    def _compute_glint_features_different_xy(
        self, measurement: EyeMeasurement
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Feature vectors (X, Y) of a measurement for a polynomial with different features for X and Y.

        Returns None if the pupil center or any corneal reflection is missing.
        """
        pcr_vectors = self._pcr_vectors(measurement)
        if pcr_vectors is None:
            return None
        return self._concatenate_different_xy(self._glint_features(pcr_vectors))

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibration function for pupil-CR polynomial gaze model.

        Detects calibration plane and performs polynomial least squares calibration.
        Automatically handles different polynomial feature types.

        Args:
            calibration_measurements: List of EyeMeasurement objects from calibration

        """
        # Detect calibration plane for coordinate system
        self.plane_info = detect_calibration_plane(self.calib_points)
        info(summarize_plane_detection(self.calib_points, self.plane_info))

        # Determine polynomial feature type for appropriate calibration
        test_features = self.polynomial_func(0.0, 0.0)

        if test_features.uses_different_xy_features:
            self._calibrate_different_xy(calibration_measurements)
        else:
            self._calibrate_same_xy(calibration_measurements)

    def _calibrate_same_xy(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibrate with polynomial using same features for both coordinates.

        Uses single polynomial for both X and Y gaze components.
        Features are computed per-glint and concatenated across all available glints.
        """
        # Determine feature vector size from first valid measurement
        feature_size = None
        for measurement in calibration_measurements:
            features = self._compute_glint_features(measurement)
            if features is not None:
                feature_size = len(features)
                break

        if feature_size is None:
            raise ValueError("No valid calibration data found")

        # Build feature matrix for all calibration points
        feature_matrix = np.zeros((feature_size, len(self.calib_points)))

        for i, measurement in enumerate(calibration_measurements):
            features = self._compute_glint_features(measurement)
            if features is not None:
                feature_matrix[:, i] = features

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
        Features are computed per-glint and concatenated across all available glints,
        with X and Y feature sets concatenated separately.
        """
        # Determine feature sizes from first valid measurement
        x_feature_size = None
        y_feature_size = None
        for measurement in calibration_measurements:
            result = self._compute_glint_features_different_xy(measurement)
            if result is not None:
                x_features, y_features = result
                x_feature_size = len(x_features)
                y_feature_size = len(y_features)
                break

        if x_feature_size is None:
            raise ValueError("No valid calibration data found")

        # Build separate feature matrices for X and Y coordinates
        feature_matrix_x = np.zeros((x_feature_size, len(self.calib_points)))
        feature_matrix_y = np.zeros((y_feature_size, len(self.calib_points)))

        for i, measurement in enumerate(calibration_measurements):
            result = self._compute_glint_features_different_xy(measurement)
            if result is not None:
                x_features, y_features = result
                feature_matrix_x[:, i] = x_features  # X coordinate features
                feature_matrix_y[:, i] = y_features  # Y coordinate features

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

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction | None:
        """Predict gaze from pupil-corneal reflection vectors.

        Applies calibrated polynomial model to predict screen gaze coordinates.
        Computes P-CR vectors for all available glints and concatenates the
        polynomial features before applying the calibrated coefficients.

        Args:
            measurement: EyeMeasurement object containing pupil and corneal reflection data

        Returns:
            GazePrediction, or None when the pupil center or a corneal reflection is missing.

        Raises:
            ValueError: If the tracker has not been calibrated

        """
        self._require_calibration()
        start_time = time.perf_counter()

        pcr_vectors = self._pcr_vectors(measurement)
        if pcr_vectors is None:
            return None
        glint_features = self._glint_features(pcr_vectors)

        # Intermediate results for debugging and analysis
        intermediate_results = {
            "pc": measurement.pupil_data.center,
            "corneal_reflections": measurement.camera_image.corneal_reflections,
            "polynomial_name": self.polynomial_name,
            "pcr_vectors": pcr_vectors,
            "polynomial_info": glint_features[0],
        }

        if glint_features[0].uses_same_xy_features:
            feature_vector = self._concatenate_same_xy(glint_features)
            intermediate_results["feature_vector"] = feature_vector
            coefficient_matrix = np.vstack([
                self.algorithm_state.x_coefficients,
                self.algorithm_state.y_coefficients,
            ])
            gaze_2d = coefficient_matrix @ feature_vector
        else:
            feature_vector_x, feature_vector_y = self._concatenate_different_xy(glint_features)
            intermediate_results["feature_vector_x"] = feature_vector_x
            intermediate_results["feature_vector_y"] = feature_vector_y
            gaze_2d = np.array([
                self.algorithm_state.x_coefficients @ feature_vector_x,
                self.algorithm_state.y_coefficients @ feature_vector_y,
            ])

        return GazePrediction(
            gaze_point=self.plane_info.reconstruct_3d_point(gaze_2d[0], gaze_2d[1]),
            confidence=1.0,
            algorithm_name=f"polynomial_{self.polynomial_name}",
            processing_time=time.perf_counter() - start_time,
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
            "pupil_center_method": self.pupil_center_method,
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
            pupil_center_method=data["pupil_center_method"],
        )

        # Restore algorithm state and plane info
        tracker.algorithm_state = PolynomialGazeModelState.deserialize(data["algorithm_state"])
        if data["plane_info"]:
            tracker.plane_info = PlaneInfo.deserialize(data["plane_info"])

        return tracker
