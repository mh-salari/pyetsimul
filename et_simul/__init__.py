"""
et_simul - Python Eye Tracker Simulation Library

A comprehensive eye tracking simulation framework ported from MATLAB.
"""

__version__ = "1.0.0"

# Make core modules easily accessible
from . import core
from . import geometry
from . import optics
from . import visualization
from . import performance_analysis
from . import gaze_tracking_algorithms

__all__ = ["core", "geometry", "optics", "visualization", "performance_analysis", "gaze_tracking_algorithms"]
