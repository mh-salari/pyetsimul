"""
This module re-exports the structured data types from its submodules
and defines common type aliases for the project.
"""

from .algorithms import (
    AlgorithmState,
    GazePrediction,
    PolynomialGazeModelState,
    PolynomialFeatures,
)
from .geometry import (
    IntersectionResult,
    Point2D,
    Point3D,
    Ray,
    Vector3D,
    Position3D,
    Direction3D,
    TransformationMatrix,
    RotationMatrix,
)
from .imaging import CameraImage, CameraMatrix, EyeMeasurement, PupilData, ProjectionResult

__all__ = [
    "AlgorithmState",
    "GazePrediction",
    "PolynomialGazeModelState",
    "PolynomialFeatures",
    "IntersectionResult",
    "Point2D",
    "Point3D",
    "Ray",
    "Vector3D",
    "Position3D",
    "Direction3D",
    "CameraImage",
    "CameraMatrix",
    "EyeMeasurement",
    "PupilData",
    "ProjectionResult",
    "RotationMatrix",
    "TransformationMatrix",
]
