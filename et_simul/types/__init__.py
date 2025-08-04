"""
This module re-exports the structured data types from its submodules
and defines common type aliases for the project.
"""

from .algorithms import (
    AlgorithmState,
    GazePrediction,
    HennesseyConfig,
    HennesseyState,
    InterpolationConfig,
    InterpolationState,
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
from .imaging import CameraImage, EyeMeasurement, PupilData, ProjectionResult

__all__ = [
    "AlgorithmState",
    "GazePrediction",
    "HennesseyConfig",
    "HennesseyState",
    "InterpolationConfig",
    "InterpolationState",
    "PolynomialFeatures",
    "IntersectionResult",
    "Point2D",
    "Point3D",
    "Ray",
    "Vector3D",
    "Position3D",
    "Direction3D",
    "CameraImage",
    "EyeMeasurement",
    "PupilData",
    "ProjectionResult",
    "RotationMatrix",
    "TransformationMatrix",
]
