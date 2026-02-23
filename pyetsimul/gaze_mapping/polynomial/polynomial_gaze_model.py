"""Polynomial gaze model eye tracker.

Pupil-CR tracker with polynomial gaze model.
"""

import time

import numpy as np

from pyetsimul.core import Camera, EyeTracker, Light
from pyetsimul.log import info
from pyetsimul.types.algorithms import GazePrediction
from pyetsimul.types.geometry import Point3D, Position3D
from pyetsimul.types.imaging import EyeMeasurement

from ...geometry.plane_detection import PlaneInfo, detect_calibration_plane, summarize_plane_detection
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

    def _compute_glint_features(self, measurement: EyeMeasurement) -> np.ndarray | None:
        """Compute polynomial features for all available glints and concatenate them.

        Applies the same polynomial to each P-CR vector (pupil center minus each
        corneal reflection) and concatenates the feature vectors. This allows any
        single-glint polynomial to automatically leverage multiple light sources.

        For example, with Cerrolaza [x², y², xy, x, y, 1] and 2 glints, the
        concatenated feature vector becomes:
        [x1², y1², x1y1, x1, y1, 1, x2², y2², x2y2, x2, y2, 1]

        Args:
            measurement: EyeMeasurement with pupil data and corneal reflections

        Returns:
            Concatenated feature array for same_xy polynomials, or None if
            pupil center or any corneal reflection is missing.

        """
        pc = measurement.pupil_data.center
        if pc is None:
            return None

        corneal_reflections = measurement.camera_image.corneal_reflections
        if not corneal_reflections:
            return None

        # Apply polynomial to each P-CR vector and collect feature arrays
        all_features = []
        for cr in corneal_reflections:
            if cr is None:
                return None
            pcr = pc - cr
            poly_features = self.polynomial_func(pcr.x, pcr.y)
            all_features.append(poly_features.features)

        return np.concatenate(all_features)

    def _compute_glint_features_different_xy(
        self, measurement: EyeMeasurement
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """Compute polynomial features for all glints with different X/Y feature sets.

        Same as _compute_glint_features but handles polynomials that use independent
        feature sets for X and Y gaze coordinates. Concatenates X features and Y
        features separately across all glints.

        Args:
            measurement: EyeMeasurement with pupil data and corneal reflections

        Returns:
            Tuple of (x_features, y_features) concatenated across all glints,
            or None if pupil center or any corneal reflection is missing.

        """
        pc = measurement.pupil_data.center
        if pc is None:
            return None

        corneal_reflections = measurement.camera_image.corneal_reflections
        if not corneal_reflections:
            return None

        # Apply polynomial to each P-CR vector and collect per-coordinate features
        all_x_features = []
        all_y_features = []
        for cr in corneal_reflections:
            if cr is None:
                return None
            pcr = pc - cr
            poly_features = self.polynomial_func(pcr.x, pcr.y)
            if poly_features.features.dtype == object:
                # Handle object arrays (mixed-length coordinates)
                all_x_features.append(poly_features.features[0])
                all_y_features.append(poly_features.features[1])
            else:
                # Handle regular 2D arrays
                all_x_features.append(poly_features.features[0, :])
                all_y_features.append(poly_features.features[1, :])

        return np.concatenate(all_x_features), np.concatenate(all_y_features)

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

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction:
        """Predict gaze from pupil-corneal reflection vectors.

        Applies calibrated polynomial model to predict screen gaze coordinates.
        Computes P-CR vectors for all available glints and concatenates the
        polynomial features before applying the calibrated coefficients.
        Handles missing data with appropriate confidence scoring.

        Args:
            measurement: EyeMeasurement object containing pupil and corneal reflection data

        Returns:
            GazePrediction: Structured prediction result.

        """
        start_time = time.time()

        # Extract pupil center and corneal reflections from measurement
        pc = measurement.pupil_data.center
        corneal_reflections = measurement.camera_image.corneal_reflections
        polynomial_name = self.polynomial_name

        # Prepare intermediate results for debugging and analysis
        intermediate_results = {
            "pc": pc,
            "corneal_reflections": corneal_reflections,
            "polynomial_name": polynomial_name,
        }

        if pc is None or not corneal_reflections:
            # Handle missing pupil or corneal reflection data
            gaze_point = Point3D(0.0, 0.0, 0.0)
            confidence = 0.0

        else:
            # Calculate P-CR vectors for all glints and collect polynomial features
            pcr_vectors = []
            all_features = []
            has_missing_cr = False

            for cr in corneal_reflections:
                if cr is None:
                    has_missing_cr = True
                    break
                pcr_vector = pc - cr
                pcr_vectors.append(pcr_vector)
                poly_features = self.polynomial_func(pcr_vector.x, pcr_vector.y)
                all_features.append(poly_features)

            intermediate_results["pcr_vectors"] = pcr_vectors

            if has_missing_cr or not all_features:
                # Handle missing corneal reflection in one of the glints
                gaze_point = Point3D(0.0, 0.0, 0.0)
                confidence = 0.0
            else:
                intermediate_results["polynomial_info"] = all_features[0]

                # Determine polynomial feature type and predict gaze coordinates
                if all_features[0].uses_same_xy_features:
                    # Same features for X and Y — single concatenated vector
                    concatenated_features = np.concatenate([f.features for f in all_features])
                    intermediate_results["feature_vector"] = concatenated_features

                    # Apply calibrated polynomial model with plane coordinate system
                    coefficient_matrix = np.vstack([
                        self.algorithm_state.x_coefficients,
                        self.algorithm_state.y_coefficients,
                    ])
                    gaze_2d = coefficient_matrix @ concatenated_features
                else:
                    # Different features for X and Y — concatenate per-coordinate
                    all_x_features = []
                    all_y_features = []
                    for f in all_features:
                        if f.features.dtype == object:
                            # Handle object arrays (mixed-length coordinates)
                            all_x_features.append(f.features[0])
                            all_y_features.append(f.features[1])
                        else:
                            # Handle regular 2D arrays
                            all_x_features.append(f.features[0, :])
                            all_y_features.append(f.features[1, :])
                    concat_x = np.concatenate(all_x_features)
                    concat_y = np.concatenate(all_y_features)
                    intermediate_results["feature_vector_x"] = concat_x
                    intermediate_results["feature_vector_y"] = concat_y

                    # Apply separate calibrated coefficients for X and Y coordinates
                    gaze_2d = np.array([
                        self.algorithm_state.x_coefficients @ concat_x,
                        self.algorithm_state.y_coefficients @ concat_y,
                    ])

                gaze_point = self.plane_info.reconstruct_3d_point(gaze_2d[0], gaze_2d[1])

                confidence = 1.0  # High confidence for successful polynomial prediction

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
