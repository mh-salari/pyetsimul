import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class Light:
    """Light source for generating corneal reflections.

    Creates a light object that is positioned at the world coordinate origin.
    The position is stored internally as homogeneous coordinates [x, y, z, 1].

    This class is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.
    """

    _pos_homogeneous: np.ndarray = None

    def __post_init__(self) -> None:
        """Initialize light positioned at the origin."""
        if self._pos_homogeneous is None:
            # Default position at origin
            self._pos_homogeneous = np.array([0, 0, 0, 1], dtype=float)

    @property
    def position(self) -> np.ndarray:
        """Get the light's position in world coordinates (3D vector)."""
        return self._pos_homogeneous[:3]

    @position.setter
    def position(self, value: np.ndarray) -> None:
        """Set the light's position from 3D coordinates."""
        value = np.array(value, dtype=float)
        if len(value) != 3:
            raise ValueError(
                f"Light position must be 3D coordinates, got {len(value)}D"
            )
        self._pos_homogeneous = np.array([value[0], value[1], value[2], 1], dtype=float)
