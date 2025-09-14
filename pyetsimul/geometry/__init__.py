"""Geometry utilities for 3D calculations in eye tracking.

This module provides geometric functions for:
- 3D coordinate conversions and transformations
- Ray-surface intersection calculations
- Vector and matrix utilities
- Eye rotation calculations (Listing's law)
- Plane detection for calibration
"""

from . import conversions, intersections, listings_law, plane_detection, utils

__all__ = ["conversions", "intersections", "listings_law", "plane_detection", "utils"]
