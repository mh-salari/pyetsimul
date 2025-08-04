"""
This module provides dataclasses for camera images, pupil data, and related
imaging results to replace the Dict[str, Any] pattern.
"""

from dataclasses import dataclass
from typing import Optional, List
import numpy as np
from .geometry import Point2D, Point3D


@dataclass
class CameraImage:
    """Result of camera.take_image() operation."""

    corneal_reflections: List[Optional[Point2D]]  # CR positions for each light
    pupil_boundary: Optional[np.ndarray]  # 2×M matrix of pupil boundary points
    pupil_center: Optional[Point2D]  # Pupil center position
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

    boundary_points: Optional[np.ndarray] = None  # 2×M matrix of boundary points
    center: Optional[Point2D] = None  # Pupil center position
    ellipse_params: Optional[np.ndarray] = None  # Ellipse fitting parameters
    area: Optional[float] = None  # Pupil area in pixels

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
    gaze_direction: Optional[Point3D] = None  # 3D gaze direction vector
    timestamp: Optional[float] = None  # Measurement timestamp

    @property
    def is_valid(self) -> bool:
        """Check if measurement contains valid eye tracking data."""
        return self.pupil_data.is_valid and any(cr is not None for cr in self.camera_image.corneal_reflections)


@dataclass
class ProjectionResult:
    """Result of camera projection operation."""

    image_points: np.ndarray  # 2×n matrix of image coordinates (NaN for invalid points)
    distances: np.ndarray  # 1×n array of distances from camera along optical axis
    valid_mask: np.ndarray  # 1×n boolean array indicating points within image bounds

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
