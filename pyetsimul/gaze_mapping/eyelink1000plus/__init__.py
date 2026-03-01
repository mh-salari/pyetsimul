"""EyeLink 1000 Plus gaze model with HREF preprocessing.

Provides EyeLink1000PlusGazeModel that converts P-CR from pixel space to
HREF angular coordinates before applying the Stampe (1993) biquadratic
polynomial + corner correction.
"""

from .eyelink1000plus_gaze_model import EyeLink1000PlusGazeModel
from .href import HrefConverter

__all__ = ["EyeLink1000PlusGazeModel", "HrefConverter"]
