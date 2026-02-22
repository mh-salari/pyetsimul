"""Structured data types and common type aliases for the project.

Re-exports the structured data types from submodules and defines
common type aliases used throughout PyEtSimul.
"""

from .algorithms import AlgorithmState, GazePrediction
from .geometry import (
    Direction3D,
    IntersectionResult,
    Point2D,
    Point3D,
    Position3D,
    Ray,
    RotationMatrix,
    ScreenGeometry,
    TransformationMatrix,
    Vector3D,
)
from .imaging import CameraImage, CameraMatrix, EyeMeasurement, ProjectionResult, PupilData

__all__ = [
    "AlgorithmState",
    "CameraImage",
    "CameraMatrix",
    "Direction3D",
    "EyeMeasurement",
    "GazePrediction",
    "IntersectionResult",
    "Point2D",
    "Point3D",
    "Position3D",
    "ProjectionResult",
    "PupilData",
    "Ray",
    "RotationMatrix",
    "ScreenGeometry",
    "TransformationMatrix",
    "Vector3D",
]
