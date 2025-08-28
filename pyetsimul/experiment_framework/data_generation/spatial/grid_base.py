"""Base grid generation system for spatial parameter variations."""

import numpy as np
from abc import ABC, abstractmethod
from typing import List
from ....types import Position3D


class GridGenerator(ABC):
    """Abstract base for 3D grid generation."""

    @abstractmethod
    def generate_positions(self) -> List[Position3D]:
        """Generate list of 3D positions."""
        pass


class RegularGrid(GridGenerator):
    """Regular 3D grid generation with uniform spacing."""

    def __init__(self, center: Position3D, dx: List[float], dy: List[float], dz: List[float], grid_size: List[int]):
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
        if len(self.dx) != 2:
            raise ValueError("dx must have exactly 2 elements [min, max]")
        if len(self.dy) != 2:
            raise ValueError("dy must have exactly 2 elements [min, max]")
        if len(self.dz) != 2:
            raise ValueError("dz must have exactly 2 elements [min, max]")
        if len(self.grid_size) != 3:
            raise ValueError("grid_size must have exactly 3 elements [nx, ny, nz]")
        if any(n < 1 for n in self.grid_size):
            raise ValueError("grid_size elements must be >= 1")

    def generate_positions(self) -> List[Position3D]:
        """Generate regular grid positions."""
        positions = []

        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz
        nx, ny, nz = self.grid_size

        # Handle single point cases
        x_values = (
            [self.center.x + dx_min]
            if dx_min == dx_max or nx == 1
            else np.linspace(self.center.x + dx_min, self.center.x + dx_max, nx)
        )
        y_values = (
            [self.center.y + dy_min]
            if dy_min == dy_max or ny == 1
            else np.linspace(self.center.y + dy_min, self.center.y + dy_max, ny)
        )
        z_values = (
            [self.center.z + dz_min]
            if dz_min == dz_max or nz == 1
            else np.linspace(self.center.z + dz_min, self.center.z + dz_max, nz)
        )

        for x in x_values:
            for y in y_values:
                for z in z_values:
                    positions.append(Position3D(x, y, z))

        return positions


class RandomGrid(GridGenerator):
    """Random 3D positions within bounds."""

    def __init__(
        self, center: Position3D, dx: List[float], dy: List[float], dz: List[float], num_points: int, seed: int = None
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
            raise ValueError("dx must have exactly 2 elements [min, max]")
        if len(self.dy) != 2:
            raise ValueError("dy must have exactly 2 elements [min, max]")
        if len(self.dz) != 2:
            raise ValueError("dz must have exactly 2 elements [min, max]")
        if self.num_points < 1:
            raise ValueError("num_points must be >= 1")

    def generate_positions(self) -> List[Position3D]:
        """Generate random positions."""
        if self.seed is not None:
            np.random.seed(self.seed)

        positions = []
        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz

        for _ in range(self.num_points):
            x = self.center.x + np.random.uniform(dx_min, dx_max)
            y = self.center.y + np.random.uniform(dy_min, dy_max)
            z = self.center.z + np.random.uniform(dz_min, dz_max)
            positions.append(Position3D(x, y, z))

        return positions
