"""Dataclasses for camera images, pupil data, and imaging results.

Provides structured dataclasses for imaging operations to replace
the Dict[str, Any] pattern with type-safe alternatives.
"""

from dataclasses import dataclass

import numpy as np

from ..core.default_configs import CameraDefaults
from .geometry import Point2D, Point3D


@dataclass
class CameraImage:
    """Result of camera.take_image() operation."""

    corneal_reflections: list[Point2D | None]  # CR positions for each light
    pupil_boundary: list[Point2D] | None  # Pupil boundary points as structured types
    pupil_center: Point2D | None  # Pupil center position
    resolution: Point2D  # Camera resolution

    @classmethod
    def empty(cls, resolution: Point2D, num_lights: int) -> "CameraImage":
        """Create empty camera image with no detected features."""
        return cls(
            corneal_reflections=[None] * num_lights, pupil_boundary=None, pupil_center=None, resolution=resolution
        )


@dataclass
class PupilData:
    """Result of pupil detection/analysis operations."""

    boundary_points: np.ndarray | None = None  # 2xM matrix of boundary points
    center: Point2D | None = None  # Pupil center position
    ellipse_params: np.ndarray | None = None  # Ellipse fitting parameters
    area: float | None = None  # Pupil area in pixels

    @property
    def is_valid(self) -> bool:
        """Check if pupil data contains valid measurements."""
        return self.boundary_points is not None and self.center is not None

    @classmethod
    def empty(cls) -> "PupilData":
        """Create empty pupil data indicating no detection."""
        return cls()


@dataclass
class EyeMeasurement:
    """Complete eye measurement from camera."""

    camera_image: CameraImage
    pupil_data: PupilData
    gaze_direction: Point3D | None = None  # 3D gaze direction vector
    timestamp: float | None = None  # Measurement timestamp

    @property
    def is_valid(self) -> bool:
        """Check if measurement contains valid eye tracking data."""
        return self.pupil_data.is_valid and any(cr is not None for cr in self.camera_image.corneal_reflections)


@dataclass
class ProjectionResult:
    """Result of camera projection operation."""

    image_points: np.ndarray  # 2xn matrix of image coordinates (NaN for invalid points)
    distances: np.ndarray  # 1xn array of distances from camera along optical axis
    valid_mask: np.ndarray  # 1xn boolean array indicating points within image bounds

    @property
    def num_points(self) -> int:
        """Number of projected points."""
        return self.image_points.shape[1] if self.image_points.ndim > 1 else 1

    @property
    def valid_points(self) -> np.ndarray:
        """Get only the valid image points."""
        if self.image_points.ndim == 1:
            return self.image_points if self.valid_mask else np.array([np.nan, np.nan])
        return self.image_points[:, self.valid_mask]


class CameraMatrix:
    """3x3 camera matrix with convenient properties for focal_length and resolution."""

    def __init__(self, matrix: np.ndarray | None = None) -> None:
        """Initialize camera matrix.

        Args:
            matrix: Optional 3x3 camera matrix. If None, uses defaults.

        """
        if matrix is not None:
            self._matrix = np.asarray(matrix, dtype=np.float64)
        else:
            self._matrix = np.array(
                [
                    [CameraDefaults.FOCAL_LENGTH, 0.0, CameraDefaults.PRINCIPAL_POINT_X],
                    [0.0, CameraDefaults.FOCAL_LENGTH, CameraDefaults.PRINCIPAL_POINT_Y],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float64,
            )

    @property
    def focal_length(self) -> float:
        """Get the camera focal length in pixels."""
        return float(self._matrix[0, 0])

    @focal_length.setter
    def focal_length(self, value: float | list[float]) -> None:
        if isinstance(value, (int, float)):
            self._matrix[0, 0] = float(value)
            self._matrix[1, 1] = float(value)
        else:
            self._matrix[0, 0] = float(value[0])
            self._matrix[1, 1] = float(value[1])

    @property
    def resolution(self) -> Point2D:
        """Get the camera resolution in pixels."""
        cx, cy = self._matrix[0, 2], self._matrix[1, 2]
        return Point2D(x=int(2 * cx), y=int(2 * cy))

    @resolution.setter
    def resolution(self, value: Point2D) -> None:
        self._matrix[0, 2] = value.x / 2.0
        self._matrix[1, 2] = value.y / 2.0

    @property
    def matrix(self) -> np.ndarray:
        """Get the camera intrinsics matrix."""
        return self._matrix

    def __array__(self) -> np.ndarray:
        """Enable numpy array conversion for camera intrinsics."""
        return self._matrix
