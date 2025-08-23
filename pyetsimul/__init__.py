"""
PyEtSimul - Python Eye Tracker Simulation Library

Main package API for the eye tracking simulation framework.
Exports core modules and common structured types for user convenience.
"""

__version__ = "1.0.0"

# Make core modules easily accessible
from . import core
from . import geometry
from . import optics
from . import types
from . import gaze_tracking_algorithms
from . import experimental_designs
from . import data_generation

# Also import commonly used structured types for convenience
from .types import (
    Position3D,
    Direction3D,
    Point3D,
    Vector3D,
    Point2D,
    CameraImage,
    PupilData,
    GazePrediction,
)

__all__ = [
    # Modules
    "core",
    "geometry",
    "optics",
    "types",
    "gaze_tracking_algorithms",
    "experimental_designs",
    "data_generation",
    # Common structured types for convenience
    "Position3D",
    "Direction3D",
    "Point3D",
    "Vector3D",
    "Point2D",
    "CameraImage",
    "PupilData",
    "GazePrediction",
]
