"""Eye movement experimental design for eye tracking simulation.

Defines eye position patterns for systematic eye movement experiments.
The gaze target is fixed while the eye position varies across a 3D grid.
"""

from typing import List, Dict, Any
import numpy as np

from .base import ExperimentalDesignBase
from ..types import Position3D


class EyeMovement(ExperimentalDesignBase):
    """Eye movement experimental design.

    Generates systematic eye position variations around a central position
    while keeping the gaze target fixed. This tests eye tracker robustness
    to observer position changes.
    """

    def __init__(
        self,
        eye_center: Position3D,
        gaze_target: Position3D,
        dx: List[float],
        dy: List[float],
        dz: List[float],
        grid_size: List[int],
    ):
        """Initialize eye movement design.

        Args:
            eye_center: Central eye position [x, y, z] in meters
            gaze_target: Fixed gaze target position [x, y, z] in meters
            dx: Eye position variation range [min, max] in X direction (meters)
            dy: Eye position variation range [min, max] in Y direction (meters)
            dz: Eye position variation range [min, max] in Z direction (meters)
            grid_size: Number of positions [nx, ny, nz] along each axis
        """
        if len(dx) != 2 or len(dy) != 2 or len(dz) != 2:
            raise ValueError("dx, dy, dz must each contain exactly 2 values [min, max]")
        if len(grid_size) != 3 or any(n <= 0 for n in grid_size):
            raise ValueError("grid_size must contain 3 positive integers [nx, ny, nz]")

        self.eye_center = eye_center
        self.gaze_target = gaze_target
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.grid_size = grid_size

    def validate_design(self) -> None:
        """Validate the experimental design parameters."""
        if self.dx[0] > self.dx[1]:
            raise ValueError("dx[0] must be <= dx[1]")
        if self.dy[0] > self.dy[1]:
            raise ValueError("dy[0] must be <= dy[1]")
        if self.dz[0] > self.dz[1]:
            raise ValueError("dz[0] must be <= dz[1]")

    def generate_eye_positions(self) -> List[Position3D]:
        """Generate all eye positions for the experimental design.

        Returns:
            List of Position3D objects representing eye positions
        """
        positions = []

        # Generate coordinate arrays
        dx_min, dx_max = self.dx
        dy_min, dy_max = self.dy
        dz_min, dz_max = self.dz
        nx, ny, nz = self.grid_size

        x_values = (
            [self.eye_center.x + dx_min]
            if dx_min == dx_max
            else np.linspace(self.eye_center.x + dx_min, self.eye_center.x + dx_max, nx)
        )
        y_values = (
            [self.eye_center.y + dy_min]
            if dy_min == dy_max
            else np.linspace(self.eye_center.y + dy_min, self.eye_center.y + dy_max, ny)
        )
        z_values = (
            [self.eye_center.z + dz_min]
            if dz_min == dz_max
            else np.linspace(self.eye_center.z + dz_min, self.eye_center.z + dz_max, nz)
        )

        # Generate all combinations
        for i, x in enumerate(x_values):
            for j, y in enumerate(y_values):
                for k, z in enumerate(z_values):
                    positions.append(Position3D(x, y, z))

        return positions

    def get_design_parameters(self) -> Dict[str, Any]:
        """Get design parameters for summary reporting.

        Returns:
            Dictionary containing design parameters
        """
        varies_x = self.dx[0] != self.dx[1]
        varies_y = self.dy[0] != self.dy[1]
        varies_z = self.dz[0] != self.dz[1]

        return {
            "type": "eye_movement",
            "eye_center": [self.eye_center.x, self.eye_center.y, self.eye_center.z],
            "gaze_target": [self.gaze_target.x, self.gaze_target.y, self.gaze_target.z],
            "dx_range": self.dx,
            "dy_range": self.dy,
            "dz_range": self.dz,
            "grid_size": self.grid_size,
            "total_positions": np.prod(self.grid_size),
            "varying_dimensions": {
                "x": varies_x,
                "y": varies_y,
                "z": varies_z,
                "count": sum([varies_x, varies_y, varies_z]),
            },
        }
