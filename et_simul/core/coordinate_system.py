"""Coordinate system configuration for the et_simul framework.

This module provides global configuration for coordinate system handedness.
By default, the framework enforces a right-handed coordinate system, but this
can be disabled to allow left-handed systems for compatibility with legacy setups.
"""

import numpy as np

# Global flag to enforce right-handed coordinate systems
_ENFORCE_RIGHT_HANDED = True


def enforce_right_handed_coordinates(enforce: bool) -> None:
    """Set whether to enforce right-handed coordinate systems.

    Args:
        enforce: If True, enforce right-handed coordinate systems (default).
                If False, allow left-handed coordinate systems.
    """
    global _ENFORCE_RIGHT_HANDED
    _ENFORCE_RIGHT_HANDED = enforce


def is_right_handed_enforced() -> bool:
    """Check if right-handed coordinate systems are enforced.

    Returns:
        True if right-handed coordinate systems are enforced, False otherwise.
    """
    return _ENFORCE_RIGHT_HANDED


def validate_orientation_matrix(matrix: np.ndarray, object_name: str = "object") -> None:
    """Validate that an orientation matrix is a proper rotation matrix.

    Args:
        matrix: 3x3 rotation matrix to validate
        object_name: Name of the object for error messages (e.g., "Eye", "Camera")

    Raises:
        ValueError: If the matrix is not a proper rotation matrix, or if it's
                   left-handed and right-handed enforcement is enabled.
    """
    det = np.linalg.det(matrix)

    if abs(det - 1.0) > 1e-6:
        if abs(det + 1.0) < 1e-6:
            # Left-handed coordinate system (det = -1)
            if _ENFORCE_RIGHT_HANDED:
                raise ValueError(
                    f"Left-handed coordinate system detected. {object_name} orientation "
                    f"must be right-handed (det = +1). Use enforce_right_handed_coordinates(False) "
                    f"to allow left-handed coordinate systems."
                )
        else:
            # Invalid rotation matrix
            raise ValueError(
                f"Invalid rotation matrix (det = {det:.3f}). Determinant must be ±1 for a proper rotation matrix."
            )
