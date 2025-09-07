"""Base grid generation system for spatial parameter variations."""

import numpy as np
from abc import ABC, abstractmethod
from typing import Iterable, Optional
from ..types import Position3D


def _generate_axis_values(center_coord: float, min_offset: float, max_offset: float, num_points: int) -> np.ndarray:
    """Generate values along a single axis."""
    if min_offset == max_offset or num_points == 1:
        return np.array([center_coord + min_offset])
    return np.linspace(center_coord + min_offset, center_coord + max_offset, num_points)


class GridGenerator(ABC):
    """Abstract base for 3D grid generation."""

    @abstractmethod
    def generate_positions(self) -> Iterable[Position3D]:
        """Generate list of 3D positions."""
        pass


class RegularGrid(GridGenerator):
    """Regular 3D grid generation with uniform spacing."""

    def __init__(self, center: Position3D, dx: list[float], dy: list[float], dz: list[float], grid_size: list[int]):
        """Initialize regular grid.

        Args:
            center: Grid center position
            dx: [min_offset, max_offset] in x direction
            dy: [min_offset, max_offset] in y direction
            dz: [min_offset, max_offset] in z direction
            grid_size: [nx, ny, nz] number of points per dimension
        """
        self.center = center
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.grid_size = grid_size

        self._validate_parameters()

    def _validate_parameters(self):
        """Validate grid parameters."""
        for name, param in [("dx", self.dx), ("dy", self.dy), ("dz", self.dz)]:
            if len(param) != 2:
                raise ValueError(f"{name} must have exactly 2 elements [min, max], got {len(param)}")

        if len(self.grid_size) != 3:
            raise ValueError(f"grid_size must have exactly 3 elements [nx, ny, nz], got {len(self.grid_size)}")

        if any(n < 1 for n in self.grid_size):
            raise ValueError(f"grid_size elements must be >= 1, got {self.grid_size}")

    def generate_positions(self) -> Iterable[Position3D]:
        """Generate regular grid positions."""
        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz
        nx, ny, nz = self.grid_size

        x_values = _generate_axis_values(self.center.x, dx_min, dx_max, nx)
        y_values = _generate_axis_values(self.center.y, dy_min, dy_max, ny)
        z_values = _generate_axis_values(self.center.z, dz_min, dz_max, nz)

        # For XZ plane: Z outer (slow), Y middle, X inner (fast)
        # For XY plane: Y outer (slow), Z middle, X inner (fast)
        for z in z_values:
            for y in y_values:
                for x in x_values:
                    yield Position3D(x, y, z)


class RandomGrid(GridGenerator):
    """Random 3D positions within bounds."""

    def __init__(
        self, center: Position3D, dx: list[float], dy: list[float], dz: list[float], num_points: int, seed: Optional[int] = None
    ):
        """Initialize random grid.

        Args:
            center: Grid center position
            dx: [min_offset, max_offset] in x direction
            dy: [min_offset, max_offset] in y direction
            dz: [min_offset, max_offset] in z direction
            num_points: Number of random points to generate
            seed: Random seed for reproducibility
        """
        self.center = center
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.num_points = num_points
        self.seed = seed

        self._validate_parameters()

    def _validate_parameters(self):
        """Validate grid parameters."""
        if len(self.dx) != 2:
            raise ValueError(f"dx must have exactly 2 elements [min, max], got {len(self.dx)} elements: {self.dx}")
        if len(self.dy) != 2:
            raise ValueError(f"dy must have exactly 2 elements [min, max], got {len(self.dy)} elements: {self.dy}")
        if len(self.dz) != 2:
            raise ValueError(f"dz must have exactly 2 elements [min, max], got {len(self.dz)} elements: {self.dz}")
        if self.num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {self.num_points}")

    def generate_positions(self) -> Iterable[Position3D]:
        """Generate random positions."""
        if self.seed is not None:
            np.random.seed(self.seed)

        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz

        for _ in range(self.num_points):
            x = self.center.x + np.random.uniform(dx_min, dx_max)
            y = self.center.y + np.random.uniform(dy_min, dy_max)
            z = self.center.z + np.random.uniform(dz_min, dz_max)
            yield Position3D(x, y, z)
