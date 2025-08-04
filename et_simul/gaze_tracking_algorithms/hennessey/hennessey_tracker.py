"""Hennessey geometric eye tracker.

Implements geometric eye tracking method using corneal curvature estimation
and pupil-corneal reflection relationships for gaze estimation.
"""

import time
import numpy as np
from typing import List


from et_simul.core import EyeTracker
from et_simul.types import (
    HennesseyConfig,
    HennesseyState,
    GazePrediction,
    EyeMeasurement,
    Position3D,
    Point3D,
    Direction3D,
    Ray,
)
from et_simul.geometry.plane_detection import detect_calibration_plane


class HennesseyTracker(EyeTracker):
    """Geometric eye tracker using Hennessey et al.'s method.

    Implements corneal center estimation and geometric gaze computation
    based on pupil-corneal reflection relationships.

    Based on: Craig Hennessey, Borna Noureddin, Peter Lawrence.
    "A Single Camera Eye-Gaze Tracking System with Free Head Motion." ETRA 2006.
    """

    def __init__(self, config: HennesseyConfig = None, **kwargs):
        """Initialize Hennessey tracker with structured configuration.

        Args:
            config: HennesseyConfig instance with algorithm parameters
            **kwargs: Arguments passed to parent EyeTracker
        """
        super().__init__(**kwargs)
        self.config = config or HennesseyConfig()
        self.algorithm_state = HennesseyState()
        self.plane_info = None

    @property
    def algorithm_name(self) -> str:
        """Algorithm name identifier."""
        return "hennessey"

    @classmethod
    def create(
        cls,
        cameras: List,
        lights: List,
        calib_points: List[Position3D],
        config: HennesseyConfig = None,
        use_refraction: bool = True,
    ) -> "HennesseyTracker":
        """Create Hennessey eye tracker setup.

        Factory method for configuring complete Hennessey tracker with all components.

        Args:
            cameras: List of Camera objects
            lights: List of Light objects
            calib_points: List of calibration target positions
            config: HennesseyConfig instance (optional)
            use_refraction: Whether to use refraction model (default True)

        Returns:
            HennesseyTracker: Configured Hennessey eye tracker
        """
        return cls(
            config=config,
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
        )

    def calibrate(self, calib_data: List[EyeMeasurement]) -> None:
        """Calibration function for Hennessey method.

        Performs corneal center estimation and recalibration parameter computation
        using geometric relationships between pupil and corneal reflections.

        Args:
            calib_data: List of EyeMeasurement objects from calibration
        """
        self._calibrate_hennessey(calib_data)

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction:
        """Predict gaze from eye measurement using Hennessey method.

        Applies corneal center estimation and geometric gaze computation
        to predict screen gaze coordinates.

        Args:
            measurement: EyeMeasurement object containing pupil and corneal reflection data

        Returns:
            GazePrediction: Structured prediction result with confidence and timing
        """
        start_time = time.time()

        gaze_point, confidence, intermediate_results = self._predict_hennessey(measurement)

        processing_time = time.time() - start_time

        return GazePrediction(
            gaze_point=gaze_point,
            confidence=confidence,
            algorithm_name="hennessey",
            processing_time=processing_time,
            intermediate_results=intermediate_results,
        )

    def _calibrate_hennessey(self, calib_data: List[EyeMeasurement]) -> None:
        """Internal calibration implementation for Hennessey method.

        Estimates corneal center and computes recalibration parameters by comparing
        predicted gaze positions with known calibration targets.

        Args:
            calib_data: List of EyeMeasurement objects from calibration
        """
        # Detect calibration plane for coordinate system
        self.plane_info = detect_calibration_plane(self.calib_points)

        n_points = len(self.calib_points)
        gaze_measured = np.zeros((2, n_points))

        # For each calibration point, predict gaze and store results
        for i, measurement in enumerate(calib_data):
            # Use internal prediction to get measured gaze
            gaze_point, confidence, _ = self._predict_hennessey(measurement)

            # Skip failed predictions
            if confidence == 0.0:
                continue

            # Store measured gaze (convert Point3D to 2D screen coordinates)
            gaze_measured[0, i] = gaze_point.x
            gaze_measured[1, i] = gaze_point.z  # Use z for screen Y coordinate

        # Compute desired gaze positions from calibration points
        gaze_desired = np.zeros((2, n_points))
        for i, calib_point in enumerate(self.calib_points):
            gaze_desired[0, i] = calib_point.x
            gaze_desired[1, i] = calib_point.z  # Use z for screen Y coordinate

        # Store recalibration parameters
        self.algorithm_state.recalib_data = {"gaze_measured": gaze_measured, "offsets": gaze_desired - gaze_measured}

        self.algorithm_state.is_calibrated = True

    def _predict_hennessey(self, measurement: EyeMeasurement) -> tuple[Point3D, float, dict]:
        """Internal prediction implementation for Hennessey method.

        Converts the complete prediction logic to use structured types.
        Based on Craig Hennessey, Borna Noureddin, Peter Lawrence.
        "A Single Camera Eye-Gaze Tracking System with Free Head Motion." ETRA 2006.

        Args:
            measurement: EyeMeasurement object containing pupil and corneal reflection data

        Returns:
            Tuple of (gaze_point, confidence, intermediate_results)
        """
        from .estimate_cc import estimate_cc_hennessey
        from et_simul.geometry.intersections import intersect_ray_sphere, intersect_ray_plane
        from et_simul.optics.refractions import refract_ray_sphere
        from skimage.measure import EllipseModel

        # Initialize result containers
        intermediate_results = {}

        # Apply parameter errors to assumed values
        r_cornea_assumed = self.config.cornea_radius * (1 + self.config.parameter_error)
        rpc_assumed = self.config.pupil_cornea_distance * (1 + self.config.parameter_error)

        # Estimate corneal center using Hennessey's method with structured types
        cc_estim = estimate_cc_hennessey(
            self.cameras[0], self.lights, measurement.camera_image.corneal_reflections, r_cornea_assumed
        )

        if cc_estim is None:
            return Point3D(0.0, 0.0, 0.0), 0.0, {"error": "corneal_center_estimation_failed"}

        # Store corneal center as Position3D
        intermediate_results["corneal_center"] = cc_estim

        # Find position of pupil center using ray-sphere intersection
        pupil_alg = self.config.pupil_algorithm
        r_pupil = None

        if pupil_alg == "hennessey":
            # Full Hennessey algorithm with pupil contour

            # Transform corneal center to camera coordinates using structured types
            cc_estim_array = np.array([cc_estim.x, cc_estim.y, cc_estim.z, 1.0])
            cc_cam_array = np.linalg.solve(self.cameras[0].trans, cc_estim_array)
            d_cc = -cc_cam_array[2]  # Distance to corneal center

            # Fit ellipse to pupil image and compute radius
            pupil_contour = measurement.pupil_data.boundary_points
            points = np.column_stack((pupil_contour[0, :], pupil_contour[1, :]))
            ellipse = EllipseModel()

            if not ellipse.estimate(points):
                return Point3D(0.0, 0.0, 0.0), 0.0, {"error": "ellipse_fitting_failed"}

            # Get pupil radius from ellipse semi-axes
            r_pupil = max(ellipse.params[2:4])
            r_pupil = r_pupil / self.cameras[0].camera_matrix.focal_length * d_cc
            r_pupil = r_pupil / self.config.empirical_correction  # Apply empirical correction

            # Unproject pupil contour using structured types
            unprojected = self.cameras[0].unproject(pupil_contour, 1.0)  # Returns List[Position3D]

            # Compute rays from camera to unprojected points using structured types
            camera_position = self.cameras[0].position  # Point3D
            pupil_rays = []
            for pos in unprojected:
                ray_vec = pos - Position3D(camera_position.x, camera_position.y, camera_position.z)
                pupil_rays.append(Ray(origin=camera_position, direction=ray_vec.normalize()))

            # Initialize pupil points array
            pupil_points = np.zeros((4, 0))

            # Refract pupil contour rays
            for j, ray in enumerate(pupil_rays):
                # Refract ray at cornea
                intersection, refracted_ray = refract_ray_sphere(ray, cc_estim, r_cornea_assumed, 1.0, 1.376)

                if intersection is None or refracted_ray is None:
                    continue

                # Intersect refracted ray with sphere around corneal center
                sphere_radius = np.sqrt(rpc_assumed**2 + r_pupil**2)
                near_intersection, far_intersection = intersect_ray_sphere(refracted_ray, cc_estim, sphere_radius)

                if near_intersection is not None and near_intersection.point is not None:
                    pt = near_intersection.point.to_array()  # Convert Position3D to homogeneous array
                    if pupil_points.shape[1] <= j:
                        # Expand array as needed
                        new_points = np.zeros((4, j + 1))
                        new_points[:, : pupil_points.shape[1]] = pupil_points
                        pupil_points = new_points
                    # Convert 3D point to 4D homogeneous coordinates
                    pt_homogeneous = np.append(pt[:3], 1.0)  # Add homogeneous coordinate
                    pupil_points[:, j] = pt_homogeneous

            # Compute pupil center as mean of valid points
            valid_cols = np.any(pupil_points != 0, axis=0)
            if np.any(valid_cols):
                pc_estim = np.mean(pupil_points[:, valid_cols], axis=1)
            else:
                return Point3D(0.0, 0.0, 0.0), 0.0, {"error": "no_valid_pupil_points"}

        elif pupil_alg == "pupil_center":
            # Simplified pupil center method
            unprojected_pc = self.cameras[0].unproject(measurement.pupil_data.center, 1.0)
            dir_vec = unprojected_pc - Position3D(
                self.cameras[0].position.x, self.cameras[0].position.y, self.cameras[0].position.z
            )

            # Create ray and refract at cornea using structured types
            ray = Ray(origin=unprojected_pc, direction=dir_vec.normalize())
            intersection, refracted_ray = refract_ray_sphere(ray, cc_estim, r_cornea_assumed, 1.0, 1.376)

            if intersection is None or refracted_ray is None:
                return Point3D(0.0, 0.0, 0.0), 0.0, {"error": "ray_refraction_failed"}

            # Intersect with sphere around corneal center
            near_intersection, _ = intersect_ray_sphere(refracted_ray, cc_estim, rpc_assumed)
            if near_intersection is not None and near_intersection.point is not None:
                pc_estim = near_intersection.point.to_array()  # Convert to homogeneous array
            else:
                return Point3D(0.0, 0.0, 0.0), 0.0, {"error": "pupil_center_intersection_failed"}
        else:
            return Point3D(0.0, 0.0, 0.0), 0.0, {"error": f"unknown_pupil_algorithm: {pupil_alg}"}

        # Convert pc_estim to Position3D for structured operations
        pc_pos = Position3D(pc_estim[0], pc_estim[1], pc_estim[2])
        intermediate_results["pupil_center"] = pc_pos
        intermediate_results["pupil_radius"] = r_pupil

        # Compute 3D gaze direction using structured types
        gaze3d_vec = pc_pos - cc_estim  # Vector3D from structured type subtraction
        gaze3d_normalized = gaze3d_vec.normalize()  # Use Vector3D.normalize()
        intermediate_results["gaze_direction_3d"] = Direction3D(
            gaze3d_normalized.x, gaze3d_normalized.y, gaze3d_normalized.z
        )

        # Intersect gaze ray with screen plane (z=0, normal=[0,1,0,0])
        gaze_ray = Ray(
            origin=cc_estim, direction=Direction3D(gaze3d_normalized.x, gaze3d_normalized.y, gaze3d_normalized.z)
        )
        screen_intersection = intersect_ray_plane(
            gaze_ray,
            Position3D(0, 0, 0),  # Point on plane
            Direction3D(0, 1, 0),  # Normal vector
        )

        if screen_intersection is None:
            return Point3D(0.0, 0.0, 0.0), 0.0, {"error": "screen_plane_intersection_failed"}

        # Extract 2D gaze coordinates using structured type properties
        gaze_2d = np.array([screen_intersection.point.x, screen_intersection.point.z])

        # Apply recalibration if calibrated
        if self.algorithm_state.is_calibrated and self.algorithm_state.recalib_data:
            gaze_2d = self._apply_hennessey_recalibration(gaze_2d)

        # Convert to Point3D (using z for screen Y coordinate)
        gaze_point = Point3D(gaze_2d[0], 0.0, gaze_2d[1])
        confidence = 1.0  # Successful prediction

        return gaze_point, confidence, intermediate_results

    def _apply_hennessey_recalibration(self, gaze: np.ndarray) -> np.ndarray:
        """Apply Hennessey's recalibration procedure.

        Args:
            gaze: 2D gaze position to be corrected

        Returns:
            Corrected 2D gaze position
        """
        state = self.algorithm_state.recalib_data
        if not state or "gaze_measured" not in state or "offsets" not in state:
            return gaze

        # Compute distances to all calibration points
        gaze_expanded = np.tile(gaze.reshape(-1, 1), (1, state["gaze_measured"].shape[1]))
        d = state["gaze_measured"] - gaze_expanded
        d = np.sqrt(np.sum(d**2, axis=0))

        # Find exact matches (within numerical precision)
        I = np.where(d < 1e-8)[0]
        if len(I) > 0:
            # Use exact match
            weights = np.zeros(d.shape)
            weights[I[0]] = 1
        else:
            # Use inverse distance weighting
            weights = 1.0 / d
            weights = weights / np.sum(weights)

        # Apply weighted offset correction
        weights_expanded = np.tile(weights.reshape(1, -1), (2, 1))
        weighted_offsets = state["offsets"] * weights_expanded
        gaze = gaze + np.sum(weighted_offsets, axis=1)

        return gaze
