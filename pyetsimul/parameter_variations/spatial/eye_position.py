"""Eye 3D position variations."""

import numpy as np
from typing import List
from ...types import Position3D
from ...core import Eye
from ..core import ParameterVariation


class Eye3DPositionVariation(ParameterVariation):
    """Varies eye position in 3D space."""

    def __init__(self, center: Position3D, dx: List[float], dy: List[float], dz: List[float], grid_size: List[int]):
        super().__init__("eye_position")
        self.center = center
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.grid_size = grid_size

    def generate_values(self) -> List[Position3D]:
        """Generate all eye positions for the 3D grid."""
        positions = []

        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz
        nx, ny, nz = self.grid_size

        x_values = (
            [self.center.x + dx_min]
            if dx_min == dx_max
            else np.linspace(self.center.x + dx_min, self.center.x + dx_max, nx)
        )
        y_values = (
            [self.center.y + dy_min]
            if dy_min == dy_max
            else np.linspace(self.center.y + dy_min, self.center.y + dy_max, ny)
        )
        z_values = (
            [self.center.z + dz_min]
            if dz_min == dz_max
            else np.linspace(self.center.z + dz_min, self.center.z + dz_max, nz)
        )

        for x in x_values:
            for y in y_values:
                for z in z_values:
                    positions.append(Position3D(x, y, z))

        return positions

    def apply_to_eye(self, eye: Eye, value: Position3D) -> None:
        """Set eye position to the specified 3D location."""
        eye.position = value
