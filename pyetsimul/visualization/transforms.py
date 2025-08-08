"""Coordinate transformation utilities for visualization.

Provides functions for transforming local surface coordinates to world coordinates for 3D plotting.
"""

import numpy as np
from typing import Tuple
from ..types import Vector3D, TransformationMatrix


def transform_surface(
    x_local: Vector3D, y_local: Vector3D, z_local: Vector3D, trans_matrix: TransformationMatrix
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    """Transform surface coordinates to world coordinates for 3D visualization.

    Applies homogeneous transformation to local surface coordinates for plotting.

    Args:
        x_local, y_local, z_local: Local surface coordinates
        trans_matrix: 4x4 transformation matrix

    Returns:
        Tuple of transformed (x, y, z) world coordinates
    """
    ones = np.ones_like(x_local)
    local_coords = np.stack([x_local, y_local, z_local, ones], axis=0)
    world_coords = np.einsum("ij,j...->i...", trans_matrix, local_coords)
    return world_coords[0], world_coords[1], world_coords[2]
