"""The default PyEtSimul eye: a bare ``EyeModel()`` that always reflects the current defaults."""

from ..eye_model import EyeModel, register_eye_model

register_eye_model("PyEtSimul", EyeModel())
