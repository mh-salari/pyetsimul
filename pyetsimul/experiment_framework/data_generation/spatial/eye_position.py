"""Eye 3D position variations."""

from typing import List
from ....types import Position3D
from ..core import Eye
from ..core import EyeParameterVariation
from .grid_base import RegularGrid


class EyePositionVariation(EyeParameterVariation):
    """Varies eye position in space using grid generation."""

    def __init__(self, center: Position3D, dx: List[float], dy: List[float], dz: List[float], grid_size: List[int]):
        super().__init__("eye_position")
        self.grid = RegularGrid(center=center, dx=dx, dy=dy, dz=dz, grid_size=grid_size)

    @property
    def description(self):
        return f"{self.__class__.__name__} (grid_size={self.grid.grid_size})"

    def generate_values(self) -> List[Position3D]:
        """Generate all eye positions using the grid system."""
        return self.grid.generate_positions()

    def apply_to_eye(self, eye: Eye, value: Position3D) -> None:
        """Set eye position to the specified location."""
        eye.position = value
