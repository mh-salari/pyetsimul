"""HREF (Head-Referenced Eye Angle) coordinate converter.

Converts pupil-corneal reflection (P-CR) vectors from camera pixel space
to HREF angular coordinates used by the EyeLink 1000 Plus.

HREF places eye angles on a virtual plane at 15000 units from the eye
(~260 units/degree). The conversion pipeline:
1. Scale P-CR by crd_scaler (per-axis correction)
2. Rotate by camera_orientation_deg (sensor-to-world alignment)
3. Scale pixels to angular units: * (HREF_VIRTUAL_DISTANCE / focal_length_px)

All parameters are user-configured (matching the real EyeLink configuration),
nothing is auto-computed.

Reference: EyeLink CALIBR.INI default_eye_mapping = -15360, 80, -12800, 160
"""

from dataclasses import dataclass

import numpy as np

from pyetsimul.types.geometry import Point2D

HREF_VIRTUAL_DISTANCE = 15000.0


@dataclass
class HrefConverter:
    """Converts P-CR pixel vectors to HREF angular coordinates.

    All parameters mirror the EyeLink 1000 Plus configuration and
    must be set explicitly by the user.

    Attributes:
        camera_orientation_deg: Camera sensor rotation angle (elcl_camera_orientation).
        camera_lens_focal_length_mm: Physical lens focal length in mm (camera_lens_focal_length).
        camera_to_eye_distance_mm: Expected eye-camera distance in mm (camera_to_eye_distance).
        crd_scaler: P-CR scaling correction (x, y) (crd_scaler).
        focal_length_px: Camera focal length in pixels (from camera matrix).

    """

    camera_orientation_deg: float
    camera_lens_focal_length_mm: float
    camera_to_eye_distance_mm: float
    crd_scaler: tuple[float, float]
    focal_length_px: float

    def __post_init__(self) -> None:
        """Precompute rotation matrix from camera orientation angle."""
        theta = np.radians(self.camera_orientation_deg)
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        self._rotation_matrix = np.array([[cos_t, -sin_t], [sin_t, cos_t]])

    def pcr_to_href(self, pcr: Point2D) -> Point2D:
        """Convert a P-CR vector from pixel space to HREF angular coordinates.

        Args:
            pcr: Pupil minus corneal reflection vector in camera pixels.

        Returns:
            HREF-space P-CR vector (angular units on virtual plane at f=15000).

        """
        # Step 1: Apply P-CR scaling correction
        scaled = np.array([pcr.x * self.crd_scaler[0], pcr.y * self.crd_scaler[1]])

        # Step 2: Rotate from sensor axes to world-aligned axes
        rotated = self._rotation_matrix @ scaled

        # Step 3: Scale from pixels to HREF angular units
        href = rotated * (HREF_VIRTUAL_DISTANCE / self.focal_length_px)

        return Point2D(float(href[0]), float(href[1]))

    def serialize(self) -> dict:
        """Serialize to dictionary."""
        return {
            "camera_orientation_deg": self.camera_orientation_deg,
            "camera_lens_focal_length_mm": self.camera_lens_focal_length_mm,
            "camera_to_eye_distance_mm": self.camera_to_eye_distance_mm,
            "crd_scaler": list(self.crd_scaler),
            "focal_length_px": self.focal_length_px,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "HrefConverter":
        """Deserialize from dictionary."""
        return cls(
            camera_orientation_deg=data["camera_orientation_deg"],
            camera_lens_focal_length_mm=data["camera_lens_focal_length_mm"],
            camera_to_eye_distance_mm=data["camera_to_eye_distance_mm"],
            crd_scaler=tuple(data["crd_scaler"]),
            focal_length_px=data["focal_length_px"],
        )
