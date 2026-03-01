"""EyeLink 1000 Plus gaze model with HREF preprocessing.

Simulates the EyeLink 1000 Plus desktop (stabilized head) gaze pipeline:
1. Convert P-CR vectors from camera pixels to HREF angular coordinates
2. Fit Stampe (1993) biquadratic polynomial + corner correction in HREF space

The HREF conversion uses four user-configurable parameters that mirror
the real EyeLink 1000 Plus configuration:
- camera_orientation_deg (elcl_camera_orientation): sensor rotation angle
- camera_to_eye_distance (camera_to_eye_distance): expected eye-camera distance in mm
- camera_lens_focal_length (camera_lens_focal_length): physical lens focal length in mm
- crd_scaler (crd_scaler): P-CR scaling correction (x, y)

All parameters are set by the user, nothing is auto-computed.
"""

import numpy as np

from pyetsimul.core import Camera, Light
from pyetsimul.gaze_mapping.polynomial.polynomial_state import PolynomialGazeModelState
from pyetsimul.gaze_mapping.stampe1993 import Stampe1993GazeModel
from pyetsimul.geometry.plane_detection import PlaneInfo
from pyetsimul.log import info
from pyetsimul.types import Position3D
from pyetsimul.types.algorithms import GazePrediction
from pyetsimul.types.geometry import Point2D
from pyetsimul.types.imaging import CameraImage, EyeMeasurement, PupilData

from .href import HrefConverter


class EyeLink1000PlusGazeModel(Stampe1993GazeModel):
    """EyeLink 1000 Plus gaze model: HREF preprocessing + Stampe (1993) polynomial.

    Wraps the Stampe1993 biquadratic polynomial with an HREF conversion layer.
    P-CR vectors are converted from camera pixel space to HREF angular coordinates
    before polynomial fitting and prediction.

    All HREF parameters are user-configured, matching the real EyeLink 1000 Plus.
    """

    def __init__(
        self,
        camera_orientation_deg: float = 142.0,
        camera_to_eye_distance: float = 600.0,
        camera_lens_focal_length: float = 38.0,
        crd_scaler: tuple[float, float] = (1.01, 1.01),
        **kwargs: object,
    ) -> None:
        """Initialize EyeLink 1000 Plus gaze model.

        Args:
            camera_orientation_deg: Camera sensor rotation angle in degrees (elcl_camera_orientation).
            camera_to_eye_distance: Expected eye-camera distance in mm (camera_to_eye_distance).
            camera_lens_focal_length: Physical lens focal length in mm (camera_lens_focal_length).
            crd_scaler: P-CR scaling correction (x, y) (crd_scaler).
            **kwargs: Arguments passed to Stampe1993GazeModel.

        """
        super().__init__(**kwargs)
        self.camera_orientation_deg = camera_orientation_deg
        self.camera_to_eye_distance = camera_to_eye_distance
        self.camera_lens_focal_length = camera_lens_focal_length
        self.crd_scaler = crd_scaler
        self.href_converter: HrefConverter | None = None

    @property
    def algorithm_name(self) -> str:
        """Return the name of the algorithm."""
        return "eyelink1000plus_href_stampe1993"

    @classmethod
    def create(
        cls,
        cameras: list,
        lights: list,
        calib_points: list[Position3D],
        use_refraction: bool = True,
        camera_orientation_deg: float = 142.0,
        camera_to_eye_distance: float = 600.0,
        camera_lens_focal_length: float = 38.0,
        crd_scaler: tuple[float, float] = (1.01, 1.01),
    ) -> "EyeLink1000PlusGazeModel":
        """Create an EyeLink 1000 Plus gaze model with all components.

        Args:
            cameras: List of Camera objects.
            lights: List of Light objects.
            calib_points: List of calibration target positions.
            use_refraction: Whether to use refraction model.
            camera_orientation_deg: Camera sensor rotation angle (degrees).
            camera_to_eye_distance: Expected eye-camera distance (mm).
            camera_lens_focal_length: Physical lens focal length (mm).
            crd_scaler: P-CR scaling correction (x, y).

        """
        return cls(
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=use_refraction,
            camera_orientation_deg=camera_orientation_deg,
            camera_to_eye_distance=camera_to_eye_distance,
            camera_lens_focal_length=camera_lens_focal_length,
            crd_scaler=crd_scaler,
        )

    def _init_href_converter(self) -> None:
        """Initialize the HREF converter using camera matrix and user-configured parameters."""
        focal_length_px = self.cameras[0].camera_matrix.focal_length
        self.href_converter = HrefConverter(
            camera_orientation_deg=self.camera_orientation_deg,
            camera_lens_focal_length_mm=self.camera_lens_focal_length,
            camera_to_eye_distance_mm=self.camera_to_eye_distance,
            crd_scaler=self.crd_scaler,
            focal_length_px=focal_length_px,
        )

    def _to_href_measurement(self, measurement: EyeMeasurement) -> EyeMeasurement:
        """Convert an EyeMeasurement from pixel P-CR space to HREF space.

        Computes the P-CR vector in pixel space, converts it to HREF via the
        HrefConverter, then creates a new measurement where PC=href_pcr and CR=(0,0)
        so the parent's `pc - cr` naturally yields the HREF vector.

        Args:
            measurement: Original measurement in camera pixel space.

        Returns:
            New EyeMeasurement in HREF angular coordinate space.

        """
        pc = measurement.pupil_data.center
        corneal_reflections = measurement.camera_image.corneal_reflections

        if pc is None or not corneal_reflections or corneal_reflections[0] is None:
            return measurement

        # Compute P-CR in pixel space and convert to HREF
        cr = corneal_reflections[0]
        pcr_pixels = Point2D(pc.x - cr.x, pc.y - cr.y)
        href_pcr = self.href_converter.pcr_to_href(pcr_pixels)

        # Create new CameraImage with CR=(0,0) and measurement with PC=href_pcr
        href_camera_image = CameraImage(
            corneal_reflections=[Point2D(0.0, 0.0)],
            pupil_boundary=measurement.camera_image.pupil_boundary,
            pupil_center=href_pcr,
            resolution=measurement.camera_image.resolution,
            glint_sizes_px=measurement.camera_image.glint_sizes_px,
        )
        href_pupil_data = PupilData(
            boundary_points=measurement.pupil_data.boundary_points,
            center=href_pcr,
        )
        return EyeMeasurement(
            camera_image=href_camera_image,
            pupil_data=href_pupil_data,
            gaze_direction=measurement.gaze_direction,
            timestamp=measurement.timestamp,
        )

    def calibrate(self, calibration_measurements: list[EyeMeasurement]) -> None:
        """Calibrate: convert P-CR to HREF, then fit Stampe (1993) polynomial.

        Args:
            calibration_measurements: List of EyeMeasurement objects from calibration.

        """
        self._init_href_converter()
        info(
            f"HREF converter initialized: orientation={self.camera_orientation_deg} deg, "
            f"distance={self.camera_to_eye_distance} mm, "
            f"focal_length={self.camera_lens_focal_length} mm, "
            f"crd_scaler={self.crd_scaler}, "
            f"focal_length_px={self.href_converter.focal_length_px:.1f} px"
        )

        # Convert all calibration measurements to HREF space
        href_measurements = [self._to_href_measurement(m) for m in calibration_measurements]

        # Delegate to Stampe1993 calibration (polynomial + corner correction)
        super().calibrate(href_measurements)

    def predict_gaze(self, measurement: EyeMeasurement) -> GazePrediction | None:
        """Predict gaze: convert P-CR to HREF, then apply calibrated polynomial.

        Args:
            measurement: EyeMeasurement in camera pixel space.

        Returns:
            GazePrediction with estimated gaze position.

        """
        href_measurement = self._to_href_measurement(measurement)
        prediction = super().predict_gaze(href_measurement)

        if prediction is not None:
            prediction.intermediate_results["href_config"] = {
                "camera_orientation_deg": self.camera_orientation_deg,
                "camera_to_eye_distance": self.camera_to_eye_distance,
                "camera_lens_focal_length": self.camera_lens_focal_length,
                "crd_scaler": self.crd_scaler,
            }

        return prediction

    def serialize(self) -> dict:
        """Serialize eye tracker to dictionary."""
        data = super().serialize()
        data["eyelink_config"] = {
            "camera_orientation_deg": self.camera_orientation_deg,
            "camera_to_eye_distance": self.camera_to_eye_distance,
            "camera_lens_focal_length": self.camera_lens_focal_length,
            "crd_scaler": list(self.crd_scaler),
        }
        if self.href_converter is not None:
            data["href_converter"] = self.href_converter.serialize()
        # Stampe1993-specific state not serialized by parent
        data["corner_coefficients"] = (
            self.corner_coefficients.tolist() if self.corner_coefficients is not None else None
        )
        data["screen_center_2d"] = list(self.screen_center_2d)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "EyeLink1000PlusGazeModel":
        """Deserialize eye tracker from dictionary."""
        eyelink_config = data["eyelink_config"]

        cameras = [Camera.deserialize(cam_data) for cam_data in data["cameras"]]
        lights = [Light.deserialize(light_data) for light_data in data["lights"]]
        calib_points = [Position3D.deserialize(pt_data) for pt_data in data["calib_points"]]

        tracker = cls(
            cameras=cameras,
            lights=lights,
            calib_points=calib_points,
            use_refraction=data["use_refraction"],
            camera_orientation_deg=eyelink_config["camera_orientation_deg"],
            camera_to_eye_distance=eyelink_config["camera_to_eye_distance"],
            camera_lens_focal_length=eyelink_config["camera_lens_focal_length"],
            crd_scaler=tuple(eyelink_config["crd_scaler"]),
        )

        tracker.algorithm_state = PolynomialGazeModelState.deserialize(data["algorithm_state"])
        if data["plane_info"]:
            tracker.plane_info = PlaneInfo.deserialize(data["plane_info"])
        if data.get("href_converter"):
            tracker.href_converter = HrefConverter.deserialize(data["href_converter"])

        # Restore corner correction state
        if "corner_coefficients" in data and data["corner_coefficients"] is not None:
            tracker.corner_coefficients = np.array(data["corner_coefficients"])
        if "screen_center_2d" in data:
            tracker.screen_center_2d = tuple(data["screen_center_2d"])

        tracker.use_legacy_look_at = data["use_legacy_look_at"]
        tracker.state = data["state"]

        return tracker
