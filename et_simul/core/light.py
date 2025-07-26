import numpy as np
from dataclasses import dataclass, field
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

    _position: np.ndarray = field(init=False, repr=False)
    _pos_homogeneous: np.ndarray = field(init=False, repr=False)

    def __init__(self, position: np.ndarray):
        self.position = position  # Uses the setter

    @property
    def position(self) -> np.ndarray:
        return self._position

    @position.setter
    def position(self, value: np.ndarray) -> None:
        value = np.array(value, dtype=float)
        if len(value) != 3:
            raise ValueError(f"Must be 3D coordinates, got {len(value)}D")
        self._position = value
        self._pos_homogeneous = np.array([value[0], value[1], value[2], 1])

    @property
    def pos_homogeneous(self) -> np.ndarray:
        return self._pos_homogeneous

    @pos_homogeneous.setter
    def pos_homogeneous(self, value: np.ndarray) -> None:
        value = np.array(value, dtype=float)
        if len(value) != 4 or value[3] != 1:
            raise ValueError("Must be homogeneous coordinates [x,y,z,1]")
        self._pos_homogeneous = value
        self._position = value[:3]  # Sync back to position

