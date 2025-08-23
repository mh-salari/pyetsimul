"""Gaze movement patterns for eye tracking simulation."""

import numpy as np
import warnings
from typing import List, Dict, Any
from dataclasses import dataclass

from ..types import Position3D
from .base import ExperimentalDesignBase


@dataclass
class GazeMovement(ExperimentalDesignBase):
    """Gaze movement pattern in 3D space.

    Defines systematic gaze target arrangements for eye tracking evaluation
    or data generation. Supports 2D grids (fixed depth) and 3D grids (varying depth).

    Args:
        grid_center: Center point of the gaze target grid in 3D space
        dx: X range [start, end] relative to grid center in meters
        dy: Y range [start, end] relative to grid center in meters
        dz: Z range [start, end] relative to grid center in meters
        grid_size: List of grid points [x, y, z] for each dimension
    """

    grid_center: Position3D
    dx: List[float]
    dy: List[float]
    dz: List[float]
    grid_size: List[int]

    def generate_target_positions(self) -> List[Position3D]:
        """Generate list of 3D gaze target positions based on design.

        Returns:
            List of Position3D objects representing gaze targets
        """
        # Extract grid sizes for each dimension
        x_grid_size, y_grid_size, z_grid_size = self.grid_size

        # Create coordinate grids for all three dimensions
        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz

        x_values = (
            [self.grid_center.x + dx_min]
            if dx_min == dx_max
            else np.linspace(self.grid_center.x + dx_min, self.grid_center.x + dx_max, x_grid_size)
        )
        y_values = (
            [self.grid_center.y + dy_min]
            if dy_min == dy_max
            else np.linspace(self.grid_center.y + dy_min, self.grid_center.y + dy_max, y_grid_size)
        )
        z_values = (
            [self.grid_center.z + dz_min]
            if dz_min == dz_max
            else np.linspace(self.grid_center.z + dz_min, self.grid_center.z + dz_max, z_grid_size)
        )

        # Generate all combinations
        targets = []
        for z in z_values:
            for x in x_values:
                for y in y_values:
                    targets.append(Position3D(x, y, z))

        return targets

    def get_design_parameters(self) -> Dict[str, Any]:
        """Return the design parameters as a dictionary."""
        return {
            "grid_center": self.grid_center,
            "dx": self.dx,
            "dy": self.dy,
            "dz": self.dz,
            "grid_size": self.grid_size,
            "total_targets": self._calculate_total_targets(),
        }

    def validate_design(self) -> bool:
        """Validate that the design parameters are valid."""
        # Validate range formats
        for name, range_val in [("dx", self.dx), ("dy", self.dy), ("dz", self.dz)]:
            if len(range_val) != 2:
                raise ValueError(f"{name} must be a list of 2 floats [min, max]")
            if range_val[0] > range_val[1]:
                raise ValueError(f"{name} min value must be <= max value")

        # Validate grid size format
        if len(self.grid_size) != 3:
            raise ValueError("grid_size must be a list of 3 integers [x, y, z]")
        if any(size < 1 for size in self.grid_size):
            raise ValueError("All grid sizes must be at least 1")

        # Generate warnings for mismatched dimensions
        x_grid_size, y_grid_size, z_grid_size = self.grid_size

        if self.dx[0] == self.dx[1] and x_grid_size > 1:
            warnings.warn(f"dx range is fixed ({self.dx[0]}) but x grid size is {x_grid_size} > 1", UserWarning)
        if self.dy[0] == self.dy[1] and y_grid_size > 1:
            warnings.warn(f"dy range is fixed ({self.dy[0]}) but y grid size is {y_grid_size} > 1", UserWarning)
        if self.dz[0] == self.dz[1] and z_grid_size > 1:
            warnings.warn(f"dz range is fixed ({self.dz[0]}) but z grid size is {z_grid_size} > 1", UserWarning)

        return True

    def _calculate_total_targets(self) -> int:
        """Calculate total number of target positions."""
        x_size, y_size, z_size = self.grid_size
        return x_size * y_size * z_size
