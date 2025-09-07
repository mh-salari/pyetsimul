"""Eye 3D position variations."""

import math
from typing import Iterable
from ..types import Position3D
from ..core import Eye
from .core import EyeParameterVariation
from .grid_base import RegularGrid


class EyePositionVariation(EyeParameterVariation):
    """Varies eye position in space using grid generation."""

    def __init__(self, center: Position3D, dx: list[float], dy: list[float], dz: list[float], grid_size: list[int]):
        super().__init__("eye_position")
        self.grid = RegularGrid(center=center, dx=dx, dy=dy, dz=dz, grid_size=grid_size)

    def __len__(self) -> int:
        return math.prod(self.grid.grid_size)

    def generate_values(self) -> Iterable[Position3D]:
        """Generate all eye positions using the grid system."""
        yield from self.grid.generate_positions()

    def describe(self) -> str:
        """Return a human-readable description of eye position variation."""
        total_positions = len(self)
        return f"eye position across {total_positions} spatial locations"

    def apply_to_eye(self, eye: Eye, value: Position3D) -> None:
        """Set eye position to the specified location."""
        eye.position = value
