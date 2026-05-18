"""Gaze model from Stampe (1993): biquadratic polynomial + per-quadrant corner correction.

Thin wrapper around :class:`stampe1993_gaze_mapping.StampeModel`. Calibration
points are passed in HV9 paper order — first 5 inner (centre + cardinal edges),
last 4 outer (corners). Layouts with fewer than 9 points fit polynomial-only.

Reference: Stampe, D. M. (1993). Heuristic filtering and reliable calibration
methods for video-based pupil-tracking systems. Behavior Research Methods,
25(2), 137-142.
"""

import time

import numpy as np
from stampe1993_gaze_mapping import StampeModel

from pyetsimul.gaze_mapping.polynomial import PolynomialGazeModel
from pyetsimul.gaze_mapping.polynomial.polynomial_descriptor import PolynomialDescriptor
from pyetsimul.gaze_mapping.polynomial.polynomials import register_polynomial
from pyetsimul.geometry.plane_detection import detect_calibration_plane, summarize_plane_detection
from pyetsimul.log import info
from pyetsimul.types import Position3D
from pyetsimul.types.algorithms import GazePrediction
from pyetsimul.types.geometry import Point3D
from pyetsimul.types.imaging import EyeMeasurement

# Pyetsimul polynomial descriptor for the Stampe biquadratic shape.
# The polynomial math itself is delegated to StampeModel; this descriptor
# is registered so the parent class can be initialised by name.
STAMPE1993_BIQUADRATIC = PolynomialDescriptor(
    name="stampe1993_biquadratic",
    description="Stampe (1993) biquadratic: a + bx + cy + dx² + ey²",
    terms=["x", "y", "x", "y", "1"],
    orders=[2, 2, 1, 1, 0],
)
register_polynomial(STAMPE1993_BIQUADRATIC)


class Stampe1993GazeModel(PolynomialGazeModel):
    """Stampe (1993) gaze model — biquadratic polynomial with per-quadrant corner correction."""

    def __init__(self, **kwargs: object) -> None:
        """Initialise with the Stampe biquadratic polynomial."""
        super().__init__(polynomial="stampe1993_biquadratic", **kwargs)
        self._stampe: StampeModel | None = None

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
        """Create a Stampe1993GazeModel with cameras, lights, calibration points, and refraction option."""
        return cls(
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
        )

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Fit StampeModel on the calibration measurements.

        Inner = first 5 points (HV9 centre + cardinal edges).
        Outer = last 4 points (HV9 corners) when 9 points are supplied; otherwise
        the polynomial alone is fit and no corner correction is applied.
        """
        self.plane_info = detect_calibration_plane(self.calib_points)
        info(summarize_plane_detection(self.calib_points, self.plane_info))

        pcr = np.array([_pcr_xy(m) for m in calibration_measurements])
        targets = np.array([self.plane_info.extract_2d_coords(pt) for pt in self.calib_points])

        if len(pcr) >= 9:
            inner_pcr, inner_targets = pcr[:5], targets[:5]
            outer_pcr, outer_targets = pcr[5:9], targets[5:9]
        else:
            inner_pcr, inner_targets = pcr, targets
            outer_pcr, outer_targets = None, None

        self._stampe = StampeModel(degree=2)
        self._stampe.fit(inner_pcr, inner_targets, outer_pcr, outer_targets)
        self.algorithm_state.is_calibrated = True

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction:
        """Predict gaze for one EyeMeasurement using the fitted StampeModel."""
        start_time = time.time()

        pc = measurement.pupil_data.center
        cr = measurement.camera_image.corneal_reflections[0] if measurement.camera_image.corneal_reflections else None
        intermediate: dict = {"pc": pc, "cr": cr, "polynomial_name": self.polynomial_name}

        if pc is None or cr is None or self._stampe is None:
            return GazePrediction(
                gaze_point=Point3D(0.0, 0.0, 0.0),
                confidence=0.0,
                algorithm_name=self.algorithm_name,
                processing_time=time.time() - start_time,
                intermediate_results=intermediate,
            )

        pcr_vector = pc - cr
        intermediate["pcr_vector"] = pcr_vector

        gaze_xy = self._stampe.predict(np.array([pcr_vector.x, pcr_vector.y]))
        final_x, final_y = float(gaze_xy[0]), float(gaze_xy[1])
        intermediate["final_gaze"] = (final_x, final_y)

        gaze_point = self.plane_info.reconstruct_3d_point(final_x, final_y)
        return GazePrediction(
            gaze_point=gaze_point,
            confidence=1.0,
            algorithm_name=self.algorithm_name,
            processing_time=time.time() - start_time,
            intermediate_results=intermediate,
        )


def _pcr_xy(measurement: EyeMeasurement) -> tuple[float, float]:
    pc = measurement.pupil_data.center
    cr = measurement.camera_image.corneal_reflections[0]
    pcr = pc - cr
    return pcr.x, pcr.y
