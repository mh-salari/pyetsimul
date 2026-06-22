"""Core API for eye tracking simulation components.

Exports main classes for eye, camera, light, cornea, and eye tracker models.
"""

# Importing this subpackage registers the named eye models.
from . import eye_models  # noqa: F401
from .camera import Camera
from .cornea import ConicCornea, Cornea, SphericalCornea, ToricCornea, create_cornea
from .eye import Eye
from .eye_model import EyeModel, get_eye_model, list_eye_models, register_eye_model
from .eye_tracker import EyeTracker
from .eyelid import Eyelid, create_eyelid
from .light import Light

__all__ = [
    "Camera",
    "ConicCornea",
    "Cornea",
    "Eye",
    "EyeModel",
    "EyeTracker",
    "Eyelid",
    "Light",
    "SphericalCornea",
    "ToricCornea",
    "create_cornea",
    "create_eyelid",
    "get_eye_model",
    "list_eye_models",
    "register_eye_model",
]
