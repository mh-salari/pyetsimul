"""
Listing's law implementation for eye rotation calculations.

This module implements Listing's law, which describes how the eye rotates
when changing fixation direction. Extracted from the Eye class for better
modularity and testability.
"""

import numpy as np
from ..types import Vector3D


def calculate_eye_rotation(out_rest: Vector3D, out_new: Vector3D) -> np.ndarray:
    """Calculate eye rotation matrix using Listing's law.

    Computes rotation matrix for eye movement from rest position to new position
    using Listing's law coordinate system approach.

    Args:
        out_rest: Direction of optical axis in rest position (3D vector)
        out_new: Direction of optical axis in new position (3D vector)

    Returns:
        3x3 rotation matrix A representing the eye rotation
    """
    # Normalize input vectors
    out_rest_norm = out_rest.normalize()
    out_new_norm = out_new.normalize()

    # Calculate rotation axis as cross product
    axis = out_new_norm.cross(out_rest_norm)

    # Check if vectors are parallel (no rotation needed)
    if axis.magnitude() == 0:
        return np.eye(3)

    # Normalize the rotation axis
    axis = axis.normalize()

    # Calculate third vectors for coordinate system
    third_rest = out_rest_norm.cross(axis)
    third_new = out_new_norm.cross(axis)

    # Build rotation matrix using coordinate system approach
    left_matrix = np.column_stack([np.array(axis), np.array(out_new_norm), np.array(third_new)])
    right_matrix = np.vstack([np.array(axis), np.array(out_rest_norm), np.array(third_rest)])

    return left_matrix @ right_matrix
