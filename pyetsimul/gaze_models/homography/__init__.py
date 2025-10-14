"""Homography normalization gaze estimation module.

Based on Hansen et al. (2010) "Homography Normalization for Robust
Gaze Estimation in Uncalibrated Setups" (ETRA 2010).
"""

from .gaussian_process import GaussianProcessErrorCorrection
from .homography_gaze_model import HomographyGazeModel
from .homography_state import HomographyGazeModelState

__all__ = [
    "GaussianProcessErrorCorrection",
    "HomographyGazeModel",
    "HomographyGazeModelState",
]
