"""Coordinate transformation utilities for visualization.

Provides functions for transforming local surface coordinates to world coordinates for 3D plotting.
"""

import numpy as np
from typing import Tuple
from ..types import TransformationMatrix

def _assert_shape(x, shape:list):
    """ ex: assert_shape(conv_input_array, [8, 3, None, None]) """
    assert len(x.shape) == len(shape), (x.shape, shape)
    for _a, _b in zip(x.shape, shape):
        if isinstance(_b, int):
            assert _a == _b, (x.shape, shape)

def transform_surface(
    x_local: np.ndarray, y_local: np.ndarray, z_local: np.ndarray, trans_matrix: TransformationMatrix
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform surface coordinates to world coordinates for 3D visualization.

    Applies homogeneous transformation to local surface coordinates for plotting.

    Args:
        x_local, y_local, z_local: Local surface coordinates
        trans_matrix: 4x4 transformation matrix

    Returns:
        Tuple of transformed (x, y, z) world coordinates
    """
    _assert_shape(y_local, x_local.shape)
    _assert_shape(z_local, x_local.shape)
    ones = np.ones_like(x_local)
    local_coords = np.stack([x_local, y_local, z_local, ones], axis=0)
    world_coords = np.einsum("ij,j...->i...", trans_matrix, local_coords)
    return world_coords[0], world_coords[1], world_coords[2]
