import numpy as np
from dataclasses import dataclass, field
from ..types import Point3D


@dataclass
class Light:
    """Light source for generating corneal reflections.

    Creates a light object positioned at the specified world coordinates.
    Accepts 4D homogeneous coordinates [x, y, z, 1].

    Args:
        position: 4D homogeneous position vector [x, y, z, 1] in meters


    """

    position: Point3D = field(init=False)

    def __init__(self, position: Point3D):
        """Initialize light with 4D homogeneous coordinates.

        Args:
            position: [x, y, z, 1] homogeneous coordinates in meters
        """
        position = np.array(position, dtype=float)

        if len(position) != 4:
            raise ValueError(f"Position must be 4D homogeneous coordinates [x,y,z,1], got {len(position)}D")

        if position[3] != 1:
            raise ValueError(f"Homogeneous coordinates must have w=1, got w={position[3]}")

        self.position = position

    def __repr__(self) -> str:
        """String representation showing the light position."""
        return f"Light(position={self.position})"
