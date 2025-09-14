"""Coordinate transformation utilities for visualization.

Provides functions for transforming local surface coordinates to world coordinates for 3D plotting.
"""

import numpy as np

from ..types import TransformationMatrix


def _assert_shape(x: np.ndarray, shape: list[int | None]) -> None:
    """ex: assert_shape(conv_input_array, [8, 3, None, None])"""
    if len(x.shape) != len(shape):
        raise ValueError(f"Expected shape of length {len(shape)}, got {x.shape}")

    for actual, expected in zip(x.shape, shape, strict=False):
        if isinstance(expected, int) and actual != expected:
            raise ValueError(f"Expected dimension {expected} but got {actual} for shape {x.shape}")


def transform_surface(
    x_local: np.ndarray, y_local: np.ndarray, z_local: np.ndarray, trans_matrix: TransformationMatrix
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Transform surface coordinates to world coordinates for 3D visualization.

    Applies homogeneous transformation to local surface coordinates for plotting.

    Args:
        x_local: Local surface X coordinates
        y_local: Local surface Y coordinates
        z_local: Local surface Z coordinates
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
