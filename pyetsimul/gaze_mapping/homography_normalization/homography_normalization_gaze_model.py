"""Homography normalization gaze estimation eye tracker.

Based on Hansen et al. (2010) "Homography Normalization for Robust
Gaze Estimation in Uncalibrated Setups" (ETRA 2010).
"""

import time

import numpy as np

from pyetsimul.core import Camera, EyeTracker, Light
from pyetsimul.log import info, warning
from pyetsimul.types.algorithms import GazePrediction
from pyetsimul.types.geometry import Point2D, Position3D
from pyetsimul.types.imaging import EyeMeasurement

from ...geometry.plane_detection import PlaneInfo, detect_calibration_plane, summarize_plane_detection
from .gaussian_process import GaussianProcessErrorCorrection
from .homography_normalization_state import HomographyNormalizationGazeModelState
from .homography_normalization_utils import apply_homography, compute_homography


class HomographyNormalizationGazeModel(EyeTracker):
    """Homography normalization gaze estimation eye tracker.

    Uses 4+ glints to normalize head pose effects through projective
    transformations. Does not require camera or geometric calibration.
    """

    def __init__(self, use_gp_correction: bool = False, ransac_threshold: float = 5.0, **kwargs: object) -> None:
        """Initialize homography gaze model tracker.

        Args:
            use_gp_correction: Whether to use Gaussian Process error correction
            ransac_threshold: Maximum reprojection error for a RANSAC inlier, in mm. Both homographies map
                            onto points given in mm: the light positions and the calibration targets.
                            Default: 5.0 mm
            **kwargs: Arguments passed to parent EyeTracker

        """
        super().__init__(**kwargs)
        self.algorithm_state = HomographyNormalizationGazeModelState(ransac_threshold=ransac_threshold)
        self.use_gp_correction = use_gp_correction
        self.ransac_threshold = ransac_threshold
        self.plane_info: PlaneInfo | None = None

    @property
    def algorithm_name(self) -> str:
        """Algorithm name identifier."""
        return "homography_normalization"

    @classmethod
    def create(
        cls,
        cameras: list[Camera],
        lights: list[Light],
        calib_points: list[Position3D],
        use_gp_correction: bool = False,
        ransac_threshold: float = 5.0,
        use_refraction: bool = True,
        pupil_center_method: str = "ellipse",
    ) -> "HomographyNormalizationGazeModel":
        """Create homography normalization gaze model eye tracker setup.

        Factory method for configuring complete homography normalization gaze model tracker.

        Args:
            cameras: List of Camera objects
            lights: List of Light objects (minimum 4 required)
            calib_points: List of 3D calibration points
            use_gp_correction: Whether to use Gaussian Process error correction
            ransac_threshold: RANSAC reprojection error threshold in mm
            use_refraction: Whether to use refraction in optical calculations
            pupil_center_method: Pupil center method: "ellipse", "convex_hull" or "center_of_mass"

        Returns:
            HomographyNormalizationGazeModel: Configured homography normalization gaze model eye tracker

        """
        return cls(
            use_gp_correction=use_gp_correction,
            ransac_threshold=ransac_threshold,
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
            pupil_center_method=pupil_center_method,
        )

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibrate homography gaze model.

        Implements Hansen et al. (2010) homography normalization calibration:

        1. Detect calibration plane for coordinate system.
        2. Define a canonical reference pattern for the N light sources.
        3. For each calibration point:

           - Pair each glint with the reference point of its light (an image holds its glints in the
             order of the lights).
           - Compute H^n_i: glints_image → normalized space using a best-fit method.
           - Normalize the pupil center: pc^n = H^n_i @ pc^i.

        4. Solve for H^s_n: normalized_pupil → screen_coordinates.
        5. (Optional) Train Gaussian Process on residual errors.
        """
        self.plane_info = detect_calibration_plane(self.calib_points)
        info(summarize_plane_detection(self.calib_points, self.plane_info))

        first_valid = next((m for m in calibration_measurements if m.camera_image.corneal_reflections), None)
        if first_valid is None or len(first_valid.camera_image.corneal_reflections) < 4:
            raise ValueError("Homography normalization requires at least 4 glints.")

        n_glints = len(first_valid.camera_image.corneal_reflections)
        info(f"Using {n_glints} glints for homography normalization.")

        # Define normalized reference pattern using 2D projection of light positions, in the order of the
        # lights. The homography will map current glints to this reference pattern
        light_positions_3d = [light.position for light in self.lights]
        light_positions_2d = np.array([self.plane_info.extract_2d_coords(p) for p in light_positions_3d])

        self.algorithm_state.reference_glints_normalized = [Point2D(x=p[0], y=p[1]) for p in light_positions_2d]

        normalized_pupils = []
        valid_calib_points = []

        for measurement, calib_point in zip(calibration_measurements, self.calib_points, strict=True):
            pc = measurement.pupil_data.center
            glints = measurement.camera_image.corneal_reflections

            if pc is None or glints is None or len(glints) < n_glints:
                continue

            glints_array = np.array([[g.x, g.y] for g in glints if g is not None], dtype=np.float32)
            if len(glints_array) != n_glints:
                continue

            reference_array = np.array(
                [[p.x, p.y] for p in self.algorithm_state.reference_glints_normalized], dtype=np.float32
            )

            H_n_i = compute_homography(glints_array, reference_array, self.ransac_threshold)  # noqa: N806
            pc_normalized = apply_homography(H_n_i, np.array([pc.x, pc.y], dtype=np.float32))

            normalized_pupils.append(pc_normalized)
            valid_calib_points.append(calib_point)

        if len(normalized_pupils) < 4:
            raise ValueError(f"Need at least 4 valid calibration points, got {len(normalized_pupils)}.")

        normalized_pupils_array = np.array(normalized_pupils, dtype=np.float32)
        calib_coords_2d = [self.plane_info.extract_2d_coords(pt) for pt in valid_calib_points]
        calib_points_array = np.array(calib_coords_2d, dtype=np.float32)

        H_s_n = compute_homography(normalized_pupils_array, calib_points_array, self.ransac_threshold)  # noqa: N806
        self.algorithm_state.H_s_n = H_s_n

        if self.use_gp_correction and len(normalized_pupils) > 4:
            predicted_points_2d = apply_homography(H_s_n, normalized_pupils_array)
            errors = calib_points_array - predicted_points_2d
            gp_model = GaussianProcessErrorCorrection()
            # Train GP on predicted positions, not targets (paper section 3.2)
            # The error field varies based on where we predict, not where the target is
            gp_model.fit(predicted_points_2d, errors)
            self.algorithm_state.gp_model = gp_model
            self.algorithm_state.calibration_errors = errors

            info("Gaussian Process error correction model trained.")
        elif self.use_gp_correction:
            # A homography fits four points exactly, so four valid points leave no residual to learn from.
            warning(
                f"Gaussian Process error correction was requested but not trained: it needs more than 4 valid "
                f"calibration points, got {len(normalized_pupils)}."
            )

        self.algorithm_state.is_calibrated = True
        info(f"Homography calibration complete with {len(normalized_pupils)} points.")

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction | None:
        """Predict gaze using homography normalization.

        Returns None when the pupil center or a corneal reflection is missing, and raises ValueError
        when the tracker has not been calibrated.
        """
        self._require_calibration()
        start_time = time.perf_counter()

        pc = measurement.pupil_data.center
        glints = measurement.camera_image.corneal_reflections
        if pc is None or glints is None:
            return None

        reference_glints = self.algorithm_state.reference_glints_normalized
        glints_array = np.array([[g.x, g.y] for g in glints if g is not None], dtype=np.float32)
        if len(glints_array) != len(reference_glints):
            return None

        intermediate_results = {"pc": pc, "glints": glints, "glints_array": glints_array}

        reference_array = np.array([[p.x, p.y] for p in reference_glints], dtype=np.float32)

        H_n_i = compute_homography(glints_array, reference_array, self.ransac_threshold)  # noqa: N806
        intermediate_results["H_n_i"] = H_n_i

        pc_normalized = apply_homography(H_n_i, np.array([pc.x, pc.y], dtype=np.float32))
        intermediate_results["pc_normalized"] = pc_normalized

        H_s_n = self.algorithm_state.H_s_n  # noqa: N806
        gaze_2d = apply_homography(H_s_n, pc_normalized.astype(np.float32))
        intermediate_results["gaze_2d"] = gaze_2d

        if self.algorithm_state.gp_model is not None:
            error_correction = self.algorithm_state.gp_model.predict(gaze_2d.reshape(1, -1))
            gaze_2d_corrected = gaze_2d + error_correction[0]
            intermediate_results["gaze_2d_uncorrected"] = gaze_2d
            intermediate_results["error_correction"] = error_correction[0]

            gaze_2d = gaze_2d_corrected

        gaze_point = self.plane_info.reconstruct_3d_point(float(gaze_2d[0]), float(gaze_2d[1]))

        return GazePrediction(
            gaze_point=gaze_point,
            confidence=1.0,
            algorithm_name=self.algorithm_name,
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
            "use_gp_correction": self.use_gp_correction,
            "ransac_threshold": self.ransac_threshold,
            "algorithm_state": self.algorithm_state.serialize(),
            "plane_info": self.plane_info.serialize() if self.plane_info else None,
            "cameras": [camera.serialize() for camera in self.cameras],
            "lights": [light.serialize() for light in self.lights],
            "calib_points": [point.serialize() for point in self.calib_points],
            "use_refraction": self.use_refraction,
            "pupil_center_method": self.pupil_center_method,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "HomographyNormalizationGazeModel":
        """Deserialize eye tracker from dictionary.

        Restores complete eye tracker state including calibration and hardware configuration.
        The restored tracker is ready to use without recalibration.

        Args:
            data: Dictionary from serialize() method

        Returns:
            HomographyNormalizationGazeModel: Fully configured and calibrated eye tracker

        """
        # Deserialize hardware components
        cameras = [Camera.deserialize(cam_data) for cam_data in data["cameras"]]
        lights = [Light.deserialize(light_data) for light_data in data["lights"]]
        calib_points = [Position3D.deserialize(pt_data) for pt_data in data["calib_points"]]

        # Create tracker instance
        tracker = cls(
            use_gp_correction=data["use_gp_correction"],
            ransac_threshold=data["ransac_threshold"],
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=data["use_refraction"],
            pupil_center_method=data["pupil_center_method"],
        )

        # Restore algorithm state and plane info
        tracker.algorithm_state = HomographyNormalizationGazeModelState.deserialize(data["algorithm_state"])
        if data["plane_info"]:
            tracker.plane_info = PlaneInfo.deserialize(data["plane_info"])

        return tracker
