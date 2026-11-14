"""Gaze model from Stampe (1993): biquadratic polynomial + per-quadrant corner correction.

Thin wrapper around :class:`stampe1993_gaze_mapping.StampeModel`. With the nine
HV9 calibration points, the polynomial is fitted on the 5 inner points (centre +
cardinal edges) and the corner correction on the 4 outer points (corners); the
points may be given in any order. Layouts with fewer than 9 points fit
polynomial-only.

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

HV9_POINTS = 9  # centre, four edge points and four corners
INNER_POINTS = 5  # the centre and the edge points, which the polynomial is fitted on


class Stampe1993GazeModel(PolynomialGazeModel):
    """Stampe (1993) gaze model — biquadratic polynomial with per-quadrant corner correction."""

    def __init__(self, **kwargs: object) -> None:
        """Initialise with the Stampe biquadratic polynomial."""
        super().__init__(polynomial="stampe1993_biquadratic", **kwargs)
        # The fitted mapping; None until the tracker is calibrated or loaded.
        self.stampe_model: StampeModel | None = None

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
        pupil_center_method: str = "ellipse",
    ) -> "Stampe1993GazeModel":
        """Create a Stampe1993GazeModel with cameras, lights, calibration points, refraction and pupil center options."""
        return cls(
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
            pupil_center_method=pupil_center_method,
        )

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Fit StampeModel on the calibration measurements.

        With 9 points the polynomial is fitted on the 5 inner points (HV9 centre + cardinal edges) and the
        corner correction on the 4 outer points (HV9 corners). Inner and outer are told apart by their
        distance from the centroid of the targets, so the points can be given in any order. With fewer
        points the polynomial alone is fit and no corner correction is applied.

        Raises:
            ValueError: If there are more than 9 calibration points, if a point has no pupil center or no
                corneal reflection, or if the 9 points do not separate into 5 inner and 4 outer ones.

        """
        if len(self.calib_points) > HV9_POINTS:
            raise ValueError(
                f"The Stampe (1993) model is calibrated on at most {HV9_POINTS} points (5 for the polynomial and 4 "
                f"corners); got {len(self.calib_points)}."
            )
        missing = [
            number
            for number, measurement in enumerate(calibration_measurements, start=1)
            if measurement.pupil_data.center is None
            or not measurement.camera_image.corneal_reflections
            or measurement.camera_image.corneal_reflections[0] is None
        ]
        if missing:
            raise ValueError(
                f"The Stampe (1993) model needs the pupil center and the corneal reflection at every calibration "
                f"point; they are missing at point(s) {missing}."
            )

        self.plane_info = detect_calibration_plane(self.calib_points)
        info(summarize_plane_detection(self.calib_points, self.plane_info))

        pcr = np.array([_pcr_xy(m) for m in calibration_measurements])
        targets = np.array([self.plane_info.extract_2d_coords(pt) for pt in self.calib_points])

        if len(pcr) == HV9_POINTS:
            distances = np.linalg.norm(targets - targets.mean(axis=0), axis=1)
            nearest_first = np.argsort(distances, kind="stable")
            if np.isclose(distances[nearest_first[INNER_POINTS - 1]], distances[nearest_first[INNER_POINTS]]):
                raise ValueError(
                    "The 9 calibration points do not separate into 5 inner and 4 outer points by their distance "
                    "from the centroid; the Stampe (1993) model needs the HV9 layout."
                )
            # Each group keeps the order the points were given in.
            inner, outer = np.sort(nearest_first[:INNER_POINTS]), np.sort(nearest_first[INNER_POINTS:])
            inner_pcr, inner_targets = pcr[inner], targets[inner]
            outer_pcr, outer_targets = pcr[outer], targets[outer]
        else:
            inner_pcr, inner_targets = pcr, targets
            outer_pcr, outer_targets = None, None

        self.stampe_model = StampeModel(degree=2)
        self.stampe_model.fit(inner_pcr, inner_targets, outer_pcr, outer_targets)
        self.algorithm_state.is_calibrated = True

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction | None:
        """Predict gaze for one EyeMeasurement using the fitted StampeModel.

        Returns None when the pupil center or the corneal reflection is missing, and raises ValueError
        when the tracker has not been calibrated.
        """
        self._require_calibration()
        start_time = time.perf_counter()

        pc = measurement.pupil_data.center
        cr = measurement.camera_image.corneal_reflections[0] if measurement.camera_image.corneal_reflections else None
        if pc is None or cr is None:
            return None

        pcr_vector = pc - cr
        intermediate: dict = {"pc": pc, "cr": cr, "polynomial_name": self.polynomial_name, "pcr_vector": pcr_vector}

        gaze_xy = self.stampe_model.predict(np.array([pcr_vector.x, pcr_vector.y]))
        final_x, final_y = float(gaze_xy[0]), float(gaze_xy[1])
        intermediate["final_gaze"] = (final_x, final_y)

        gaze_point = self.plane_info.reconstruct_3d_point(final_x, final_y)
        return GazePrediction(
            gaze_point=gaze_point,
            confidence=1.0,
            algorithm_name=self.algorithm_name,
            processing_time=time.perf_counter() - start_time,
            intermediate_results=intermediate,
        )

    def serialize(self) -> dict:
        """Serialize eye tracker to dictionary, with the fitted Stampe model."""
        data = super().serialize()
        stampe = self.stampe_model
        data["stampe_model"] = (
            None
            if stampe is None
            else {
                "degree": stampe.degree,
                "coef_x": stampe.coef_x.tolist(),
                "coef_y": stampe.coef_y.tolist(),
                "centroid": stampe.centroid.tolist(),
                "corner": stampe.corner.tolist(),
            }
        )
        return data

    @classmethod
    def _init_arguments(cls, data: dict) -> dict:
        """Constructor arguments of the saved tracker; the polynomial is fixed by this class."""
        arguments = super()._init_arguments(data)
        del arguments["polynomial"]
        return arguments

    @classmethod
    def deserialize(cls, data: dict) -> "Stampe1993GazeModel":
        """Deserialize eye tracker from dictionary, with the fitted Stampe model."""
        tracker = super().deserialize(data)
        saved = data["stampe_model"]
        if saved is not None:
            tracker.stampe_model = StampeModel(degree=saved["degree"])
            tracker.stampe_model.coef_x = np.array(saved["coef_x"])
            tracker.stampe_model.coef_y = np.array(saved["coef_y"])
            tracker.stampe_model.centroid = np.array(saved["centroid"])
            tracker.stampe_model.corner = np.array(saved["corner"])
        return tracker


def _pcr_xy(measurement: EyeMeasurement) -> tuple[float, float]:
    pc = measurement.pupil_data.center
    cr = measurement.camera_image.corneal_reflections[0]
    pcr = pc - cr
    return pcr.x, pcr.y
