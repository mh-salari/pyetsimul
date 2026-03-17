"""PyEtSimul - Python Eye Tracker Simulation Library

Main package API for the eye tracking simulation framework.
Exports core modules and common structured types for user convenience.
"""

from importlib.metadata import version

__version__ = version("pyetsimul")

# Make core modules easily accessible
from . import core, gaze_mapping, geometry, optics, simulation, types

# Also import commonly used structured types for convenience
from .types import (
    CameraImage,
    Direction3D,
    GazePrediction,
    Point2D,
    Point3D,
    Position3D,
    PupilData,
    Vector3D,
)

__all__ = [
    "CameraImage",
    "Direction3D",
    "GazePrediction",
    "Point2D",
    "Point3D",
    "Position3D",
    "PupilData",
    "Vector3D",
    "core",
    "gaze_mapping",
    "geometry",
    "optics",
    "simulation",
    "types",
]
