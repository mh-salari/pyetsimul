"""Interpolation eye tracker.

Pupil-CR tracker with polynomial interpolation.
"""

from et_simul.core import EyeTracker
from et_simul.types.algorithms import InterpolationConfig, InterpolationState, GazePrediction
from et_simul.types.imaging import EyeMeasurement
from et_simul.types.geometry import Point3D, Position3D
from .polynomials import get_polynomial
from ...geometry.plane_detection import detect_calibration_plane, summarize_plane_detection
import time
import numpy as np
from typing import List


class InterpolationTracker(EyeTracker):
    """Polynomial interpolation Pupil-CR eye tracker.

    Maps pupil-corneal reflection vectors to screen coordinates using polynomial functions.
    Supports various polynomial types from eye tracking literature.
    """

    def __init__(self, polynomial: str, config: InterpolationConfig = None, **kwargs):
        """Initialize interpolation tracker with structured configuration.

        Sets up polynomial function and algorithm state for gaze tracking.

        Args:
            polynomial: Polynomial type to use
            config: InterpolationConfig instance
            **kwargs: Arguments passed to parent EyeTracker
        """
        super().__init__(**kwargs)
        self.config = config or InterpolationConfig()
        self.algorithm_state = InterpolationState()
        self.polynomial_name = polynomial
        self.polynomial_func = get_polynomial(polynomial)
        self.plane_info = None

    @property
    def algorithm_name(self) -> str:
        """Algorithm name identifier."""
        return "interpolation"

    @classmethod
    def create(
        cls,
        cameras: List,
        lights: List,
        calib_points: List[Position3D],
        polynomial: str,
        config: InterpolationConfig = None,
        use_refraction: bool = True,
    ) -> "InterpolationTracker":
        """Create interpolation eye tracker setup.

        Factory method for configuring complete interpolation tracker with all components.

        Args:
            cameras: List of Camera objects
            lights: List of Light objects
            calib_points: List of calibration target positions
            polynomial: Polynomial type to use
            config: InterpolationConfig instance (optional)
            use_refraction: Whether to use refraction model (default True)

        Returns:
            InterpolationTracker: Configured interpolation eye tracker
        """
        return cls(
            polynomial=polynomial,
            config=config,
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
        )

    def calibrate(self, calib_data: List[EyeMeasurement]) -> None:
        """Calibration function for pupil-CR interpolation.

        Detects calibration plane and performs polynomial least squares calibration.
        Automatically handles 1D and 2D polynomial types.

        Args:
            calib_data: List of EyeMeasurement objects from calibration
        """
        # Detect calibration plane for coordinate system
        self.plane_info = detect_calibration_plane(self.calib_points)
        print(summarize_plane_detection(self.calib_points, self.plane_info))

        # Determine polynomial type (1D vs 2D) for appropriate calibration
        test_features = self.polynomial_func(0.0, 0.0)

        if test_features.is_2d:
            self._calibrate_2d(calib_data)
        else:
            self._calibrate_1d(calib_data)

    def _calibrate_1d(self, calib_data: List[EyeMeasurement]) -> None:
        """Calibrate with 1D polynomial (shared features for both coordinates).

        Uses single polynomial for both X and Y gaze components.
        """
        # Determine feature vector size from first valid measurement
        feature_size = None
        for i, measurement in enumerate(calib_data):
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
        X = np.zeros((feature_size, len(self.calib_points)))

        for i, measurement in enumerate(calib_data):
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )

            if pc is not None and cr is not None:
                pcr = pc - cr
                poly_features = self.polynomial_func(pcr.x, pcr.y)
                X[:, i] = poly_features.features

        # Map 3D calibration points to 2D plane coordinates
        calib_coords_2d = [self.plane_info.extract_2d_coords(pt) for pt in self.calib_points]
        calib_points_array = np.array(calib_coords_2d).T

        # Solve least squares: calib_points = A @ X → A = calib_points @ pinv(X)
        calibration_matrix = calib_points_array @ np.linalg.pinv(X)

        # Store coefficients for both X and Y gaze components
        self.algorithm_state.x_coefficients = calibration_matrix[0:1, :].flatten()
        self.algorithm_state.y_coefficients = calibration_matrix[1:2, :].flatten()
        self.algorithm_state.is_calibrated = True

    def _calibrate_2d(self, calib_data: List[EyeMeasurement]) -> None:
        """Calibrate with 2D polynomial (separate features for each coordinate).

        Uses independent polynomials for X and Y gaze components.
        """
        # Determine polynomial structure from first valid measurement
        poly_features = None
        for i, measurement in enumerate(calib_data):
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
        num_coords, feature_size = poly_features.features.shape
        X_x = np.zeros((feature_size, len(self.calib_points)))
        X_y = np.zeros((feature_size, len(self.calib_points)))

        for i, measurement in enumerate(calib_data):
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )

            if pc is not None and cr is not None:
                pcr = pc - cr
                poly_features = self.polynomial_func(pcr.x, pcr.y)
                X_x[:, i] = poly_features.features[0, :]  # X coordinate features
                X_y[:, i] = poly_features.features[1, :]  # Y coordinate features

        # Map 3D calibration points to 2D plane coordinates
        calib_coords_2d = [self.plane_info.extract_2d_coords(pt) for pt in self.calib_points]
        calib_points_array = np.array(calib_coords_2d).T

        # Solve separate least squares problems for X and Y coordinates
        A_x = calib_points_array[0:1, :] @ np.linalg.pinv(X_x)  # X coordinate coefficients
        A_y = calib_points_array[1:2, :] @ np.linalg.pinv(X_y)  # Y coordinate coefficients

        # Store separate coefficients for X and Y gaze components
        self.algorithm_state.x_coefficients = A_x.flatten()
        self.algorithm_state.y_coefficients = A_y.flatten()
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
            algorithm_name=f"interpolation_{polynomial_name}",
            processing_time=processing_time,
            intermediate_results=intermediate_results,
        )
