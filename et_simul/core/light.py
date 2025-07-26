import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Light:
    """Light source for generating corneal reflections.

    Creates a light object positioned at the specified world coordinates.
    The position is stored internally as homogeneous coordinates [x, y, z, 1].

    Args:
        position: 3D position vector [x, y, z] in meters

    This class is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.
    """

    position: np.ndarray
    _pos_homogeneous: np.ndarray = None

    def __post_init__(self) -> None:
        """Initialize light with specified position."""
        pos = np.array(self.position, dtype=float)
        if len(pos) != 3:
            raise ValueError(f"Light position must be 3D coordinates, got {len(pos)}D")
        self._pos_homogeneous = np.array([pos[0], pos[1], pos[2], 1], dtype=float)

    def set_position(self, value: np.ndarray) -> None:
        """Set the light's position from 3D coordinates."""
        value = np.array(value, dtype=float)
        if len(value) != 3:
            raise ValueError(
                f"Light position must be 3D coordinates, got {len(value)}D"
            )
        self.position = value
        self._pos_homogeneous = np.array([value[0], value[1], value[2], 1], dtype=float)
