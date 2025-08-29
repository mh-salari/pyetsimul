"""Target 3D position variations."""

from typing import List
from ....types import Position3D
from ..core import TargetVariation
from .grid_base import RegularGrid


class TargetPositionVariation(TargetVariation):
    """Varies gaze target positions in space using grid generation."""

    def __init__(
        self, grid_center: Position3D, dx: List[float], dy: List[float], dz: List[float], grid_size: List[int]
    ):
        super().__init__("target_position")
        self.grid = RegularGrid(center=grid_center, dx=dx, dy=dy, dz=dz, grid_size=grid_size)

    @property
    def description(self):
        return f"{self.__class__.__name__} (grid_size={self.grid.grid_size})"

    def generate_values(self) -> List[Position3D]:
        """Generate all target positions using the grid system."""
        return self.grid.generate_positions()
