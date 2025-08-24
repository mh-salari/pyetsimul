"""Target 3D position variations."""

import numpy as np
from typing import List
from ...types import Position3D
from ..core import ParameterVariation


class TargetPositionVariation(ParameterVariation):
    """Varies gaze target positions in space."""

    def __init__(
        self, grid_center: Position3D, dx: List[float], dy: List[float], dz: List[float], grid_size: List[int]
    ):
        super().__init__("target_position")
        self.grid_center = grid_center
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.grid_size = grid_size

    def generate_values(self) -> List[Position3D]:
        """Generate all target positions for the parameter grid."""
        positions = []

        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz
        nx, ny, nz = self.grid_size

        x_values = (
            [self.grid_center.x + dx_min]
            if dx_min == dx_max
            else np.linspace(self.grid_center.x + dx_min, self.grid_center.x + dx_max, nx)
        )
        y_values = (
            [self.grid_center.y + dy_min]
            if dy_min == dy_max
            else np.linspace(self.grid_center.y + dy_min, self.grid_center.y + dy_max, ny)
        )
        z_values = (
            [self.grid_center.z + dz_min]
            if dz_min == dz_max
            else np.linspace(self.grid_center.z + dz_min, self.grid_center.z + dz_max, nz)
        )

        for z in z_values:
            for x in x_values:
                for y in y_values:
                    positions.append(Position3D(x, y, z))

        return positions

    def apply_to_eye(self, eye, value: Position3D) -> None:
        """Target position variations don't modify the eye object."""
        pass
