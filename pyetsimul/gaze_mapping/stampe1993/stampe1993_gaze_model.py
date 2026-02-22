"""Gaze model based on Stampe (1993): biquadratic polynomial + per-quadrant corner correction.

Polynomial:
    A biquadratic with 5 terms and no cross-term, fit via least-squares:
        gaze = a + b*x + c*y + d*x² + e*y²
    where (x, y) are the pupil-minus-corneal-reflection (P-CR) feature coordinates.

Corner correction:
    After the polynomial fit, a per-quadrant multiplicative correction removes residual
    nonlinearity at the screen corners. The screen is divided into 4 quadrants relative
    to the centroid (Xc, Yc) of the calibration grid. For each quadrant q:
        X_final = X_poly + cx[q] * (X_poly - Xc) * (Y_poly - Yc)
        Y_final = Y_poly + cy[q] * (X_poly - Xc) * (Y_poly - Yc)
    cx[q] and cy[q] are fit via least-squares over all calibration points in that quadrant.
    Quadrant assignment is based on the sign of (X_poly - Xc, Y_poly - Yc).

Reference: Stampe, D. M. (1993). Heuristic filtering and reliable calibration methods
for video-based pupil-tracking systems. Behavior Research Methods, 25(2), 137-142.
"""

import time

import numpy as np

from pyetsimul.gaze_mapping.polynomial import PolynomialGazeModel
from pyetsimul.gaze_mapping.polynomial.polynomial_descriptor import PolynomialDescriptor
from pyetsimul.gaze_mapping.polynomial.polynomials import register_polynomial
from pyetsimul.geometry.plane_detection import detect_calibration_plane, summarize_plane_detection
from pyetsimul.log import info
from pyetsimul.types import Position3D
from pyetsimul.types.algorithms import GazePrediction
from pyetsimul.types.geometry import Point3D
from pyetsimul.types.imaging import EyeMeasurement

# Stampe (1993) biquadratic: gaze = a + b*x + c*y + d*x² + e*y²
# 5 terms, no xy cross-term — each axis fit independently by least-squares.
STAMPE1993_BIQUADRATIC = PolynomialDescriptor(
    name="stampe1993_biquadratic",
    description="Stampe (1993) biquadratic: a + bx + cy + dx² + ey²",
    terms=["x", "y", "x", "y", "1"],
    orders=[2, 2, 1, 1, 0],
)
register_polynomial(STAMPE1993_BIQUADRATIC)


class Stampe1993GazeModel(PolynomialGazeModel):
    """Gaze model from Stampe (1993): biquadratic polynomial + per-quadrant corner correction.

    Two-stage calibration:
    1. Fit biquadratic polynomial via least-squares on P-CR features.
    2. Fit per-quadrant corner correction coefficients from calibration residuals.

    Corner correction (per quadrant q, relative to calibration grid centroid (Xc, Yc)):
        X_final = X_poly + cx[q] * (X_poly - Xc) * (Y_poly - Yc)
        Y_final = Y_poly + cy[q] * (X_poly - Xc) * (Y_poly - Yc)
    """

    def __init__(self, **kwargs: object) -> None:
        """Initialize with the Stampe (1993) biquadratic polynomial."""
        super().__init__(polynomial="stampe1993_biquadratic", **kwargs)
        self.corner_coefficients: np.ndarray | None = None
        self.screen_center_2d: tuple[float, float] = (0.0, 0.0)

    @property
    def algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        return "stampe1993_biquadratic_with_corner_correction"

    @classmethod
    def create(
        cls,
        cameras: list,
        lights: list,
        calib_points: list[Position3D],
        use_refraction: bool = True,
    ) -> "Stampe1993GazeModel":
        """Create a Stampe1993GazeModel with the given cameras, lights, calibration points, and refraction option."""
        return cls(
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
        )

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Two-stage calibration: biquadratic polynomial, then corner correction."""
        # Stage 1: Fit the biquadratic polynomial
        self.plane_info = detect_calibration_plane(self.calib_points)
        info(summarize_plane_detection(self.calib_points, self.plane_info))
        self._calibrate_same_xy(calibration_measurements)

        # Stage 2: Fit corner correction from residuals
        self._fit_corner_correction(calibration_measurements)

    def _fit_corner_correction(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Fit per-quadrant corner correction coefficients from polynomial residuals.

        The screen is divided into 4 quadrants relative to the calibration grid center.
        Corner correction: X_final = X_poly + cx[q] * (X_poly - Xc) * (Y_poly - Yc)
        """
        self.corner_coefficients = np.zeros((4, 2))  # [quadrant][cx, cy]

        calib_coords_2d = [self.plane_info.extract_2d_coords(pt) for pt in self.calib_points]

        # Screen center = center of calibration grid (mean of all calibration points)
        coords_array = np.array(calib_coords_2d)
        self.screen_center_2d = (float(np.mean(coords_array[:, 0])), float(np.mean(coords_array[:, 1])))
        cx_center, cy_center = self.screen_center_2d
        info(f"\nCalibration grid center (2D): ({cx_center * 1000:.1f}, {cy_center * 1000:.1f}) mm")

        # Group points by quadrant relative to calibration center
        quadrant_data: dict[int, list[tuple[float, float, float, float]]] = {0: [], 1: [], 2: [], 3: []}

        for i, measurement in enumerate(calibration_measurements):
            pc = measurement.pupil_data.center
            cr = (
                measurement.camera_image.corneal_reflections[0]
                if measurement.camera_image.corneal_reflections
                else None
            )
            if pc is None or cr is None:
                continue

            pcr = pc - cr
            poly_features = self.polynomial_func(pcr.x, pcr.y)
            coefficient_matrix = np.vstack([
                self.algorithm_state.x_coefficients,
                self.algorithm_state.y_coefficients,
            ])
            gaze_2d = coefficient_matrix @ poly_features.features
            poly_x, poly_y = gaze_2d[0], gaze_2d[1]

            target_x, target_y = calib_coords_2d[i]

            # Quadrant relative to calibration center
            dx = poly_x - cx_center
            dy = poly_y - cy_center
            q = 0 if dx < 0 else 1
            if dy >= 0:
                q += 2

            quadrant_data[q].append((poly_x, poly_y, target_x, target_y))

        # Fit cx, cy per quadrant via least-squares on centered coordinates:
        # target_x = poly_x + cx * (poly_x - Xc) * (poly_y - Yc)
        for q in range(4):
            points = quadrant_data[q]
            if not points:
                continue

            products = []
            residuals_x = []
            residuals_y = []

            for poly_x, poly_y, target_x, target_y in points:
                product = (poly_x - cx_center) * (poly_y - cy_center)
                if abs(product) < 1e-12:
                    continue
                products.append(product)
                residuals_x.append(target_x - poly_x)
                residuals_y.append(target_y - poly_y)

            if products:
                products_arr = np.array(products)
                dot_pp = np.dot(products_arr, products_arr)
                self.corner_coefficients[q, 0] = np.dot(residuals_x, products_arr) / dot_pp
                self.corner_coefficients[q, 1] = np.dot(residuals_y, products_arr) / dot_pp

        quadrant_names = ["Q0 (top-left)", "Q1 (top-right)", "Q2 (bottom-left)", "Q3 (bottom-right)"]
        info("\nCorner Correction Coefficients:")
        for q in range(4):
            cx, cy = self.corner_coefficients[q]
            n_points = len(quadrant_data[q])
            info(f"  {quadrant_names[q]}: cx={cx:.6e}, cy={cy:.6e}  ({n_points} points)")

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction | None:
        """Predict gaze: biquadratic polynomial + corner correction."""
        start_time = time.time()

        pc = measurement.pupil_data.center
        cr = measurement.camera_image.corneal_reflections[0] if measurement.camera_image.corneal_reflections else None

        intermediate_results = {"pc": pc, "cr": cr, "polynomial_name": self.polynomial_name}

        if pc is not None and cr is not None:
            pcr_vector = pc - cr
            intermediate_results["pcr_vector"] = pcr_vector

            # Stage 1: Biquadratic polynomial
            poly_features = self.polynomial_func(pcr_vector.x, pcr_vector.y)
            coefficient_matrix = np.vstack([
                self.algorithm_state.x_coefficients,
                self.algorithm_state.y_coefficients,
            ])
            gaze_2d = coefficient_matrix @ poly_features.features
            poly_x, poly_y = gaze_2d[0], gaze_2d[1]

            # Stage 2: Corner correction (quadrants relative to calibration center)
            if self.corner_coefficients is not None:
                cx_center, cy_center = self.screen_center_2d
                dx = poly_x - cx_center
                dy = poly_y - cy_center
                q = 0 if dx < 0 else 1
                if dy >= 0:
                    q += 2

                cc_x, cc_y = self.corner_coefficients[q]
                product = dx * dy
                final_x = poly_x + cc_x * product
                final_y = poly_y + cc_y * product
            else:
                final_x, final_y = poly_x, poly_y

            intermediate_results["poly_gaze"] = (poly_x, poly_y)
            intermediate_results["final_gaze"] = (final_x, final_y)

            gaze_point = self.plane_info.reconstruct_3d_point(final_x, final_y)
            confidence = 1.0
        else:
            gaze_point = Point3D(0.0, 0.0, 0.0)
            confidence = 0.0

        processing_time = time.time() - start_time

        return GazePrediction(
            gaze_point=gaze_point,
            confidence=confidence,
            algorithm_name=self.algorithm_name,
            processing_time=processing_time,
            intermediate_results=intermediate_results,
        )
