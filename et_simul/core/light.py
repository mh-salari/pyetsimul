import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class Light:
    """Light source for generating corneal reflections.
    
    This class is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.

    Creates a light object that is positioned at the world coordinate origin.
    The position is stored in world coordinates as homogeneous coordinates [x, y, z, 1].
    """
    position: Optional[np.ndarray] = None
    
    def __post_init__(self):
        """Initialize light positioned at the origin."""
        if self.position is None:
            # Default position at origin (matches original: l.pos=[0 0 0 1]')
            self.position = np.array([0, 0, 0, 1], dtype=float)
        else:
            # Ensure user input is numpy array with correct dtype and size
            self.position = np.array(self.position, dtype=float)
            if len(self.position) != 4:
                raise ValueError(f"Light position must be 4D homogeneous coordinates, got {len(self.position)}D")