"""Optical calculations for eye tracking simulations.

Exports modules for light reflection and refraction physics, supporting ray tracing and pupil/corneal imaging.
"""

from . import reflections
from . import refractions

__all__ = ["reflections", "refractions"]
