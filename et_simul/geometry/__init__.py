"""Geometry utilities for 3D calculations in eye tracking.

This module provides geometric functions for:
- 3D coordinate conversions and transformations
- Ray-surface intersection calculations
- Vector and matrix utilities
"""

from . import conversions
from . import intersections
from . import utils

__all__ = ["conversions", "intersections", "utils"]
