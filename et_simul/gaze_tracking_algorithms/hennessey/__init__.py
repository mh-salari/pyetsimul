"""Hennessey et al. geometric eye tracking implementation.

This module implements the geometric eye tracking method described in
Hennessey et al. papers, which uses corneal curvature estimation and
geometric relationships between pupil center and corneal reflections
to estimate gaze direction.
"""

from .hennessey_tracker import HennesseyTracker

__all__ = ["HennesseyTracker"]
