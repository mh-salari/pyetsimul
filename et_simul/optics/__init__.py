"""Optical calculations for eye tracking simulations.

This module provides optical physics functions for:
- Light reflection calculations on curved surfaces
- Light refraction through optical media
- Ray tracing for corneal reflections and pupil imaging
"""

from . import reflections
from . import refractions

__all__ = ["reflections", "refractions"]
