"""Visualization tools for eye tracking setups and results.

This module provides visualization functions for:
- 3D eye tracking setup visualization with anatomical details
- Camera view rendering with pupil and corneal reflections
- Eye tracker setup diagrams and configuration displays
"""

from . import draw
from . import setup_visualization

__all__ = ['draw', 'setup_visualization']