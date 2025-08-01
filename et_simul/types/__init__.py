"""
Type definitions for et_simul framework.

This module provides type aliases and protocols for geometric operations,
optical calculations, and algorithm interfaces.
"""

from typing import Union

import numpy as np

# Core geometric types
Point2D = np.ndarray  # shape (2,) - 2D points
Point3D = np.ndarray  # shape (3,) - 3D points
Point4D = np.ndarray  # shape (4,) - homogeneous coordinates
Vector3D = np.ndarray  # shape (3,) - direction vectors
Matrix3x3 = np.ndarray  # shape (3, 3) - rotation matrices
Matrix4x4 = np.ndarray  # shape (4, 4) - transformation matrices


# Type aliases for common patterns
CoordinateType = Union[Point3D, Point4D]  # For functions that accept both 3D and 4D
RotationMatrix = Matrix3x3  # Alias for clarity
TransformationMatrix = Matrix4x4  # Alias for clarity
