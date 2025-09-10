"""Helper utilities for working with rotation matrices."""

import numpy as np
import warnings
from ..types import RotationMatrix


def _calculate_optical_axis_direction(rotation_matrix: RotationMatrix) -> np.ndarray:
    """Calculate the world direction of the optical axis for a rotation matrix.

    Both eyes and cameras have their optical axis along -Z in local coordinates.

    Args:
        rotation_matrix: RotationMatrix object

    Returns:
        3D direction vector in world coordinates
    """
    optical_axis_local = np.array([0, 0, -1])
    return rotation_matrix @ optical_axis_local


def get_facing_direction(rotation_matrix: RotationMatrix) -> str:
    """Get the facing direction description for a rotation matrix.

    Both eyes and cameras have their optical axis along -Z in local coordinates.
    This function transforms that direction to world coordinates and provides
    a human-readable description.

    Args:
        rotation_matrix: RotationMatrix object

    Returns:
        Human-readable string like '+Y axis' or '33° rotated away from +Y axis toward +Z'

    Example:
        >>> eye_matrix = RotationMatrix([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        >>> desc = get_facing_direction(eye_matrix)
        >>> print(f"Eye is looking along {desc}")
        Eye is looking along -Y axis
    """
    # Transform optical axis to world coordinates
    direction = _calculate_optical_axis_direction(rotation_matrix)

    # Define all possible axis directions
    axis_vectors = {
        "+X": np.array([1, 0, 0]),
        "-X": np.array([-1, 0, 0]),
        "+Y": np.array([0, 1, 0]),
        "-Y": np.array([0, -1, 0]),
        "+Z": np.array([0, 0, 1]),
        "-Z": np.array([0, 0, -1]),
    }

    # Find closest axis and check for exact alignment
    best_axis = ""
    best_dot = -1
    for axis_name, axis_vec in axis_vectors.items():
        dot = np.dot(direction, axis_vec)
        if dot > best_dot:
            best_dot = dot
            best_axis = axis_name

        # Check for exact axis alignment
        if np.allclose(direction, axis_vec):
            return f"({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}) {axis_name} axis"

    # Calculate angle from the closest axis
    angle_rad = np.arccos(np.clip(best_dot, -1, 1))
    angle_deg = np.degrees(angle_rad)

    # If very close to an axis (< 5°), just return the axis name
    if angle_deg < 5:
        return f"({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}) {best_axis} axis"

    # Find the direction of rotation (which other axis contributes most)
    best_axis_vec = axis_vectors[best_axis]

    # Project direction onto the plane perpendicular to best_axis
    projection = direction - np.dot(direction, best_axis_vec) * best_axis_vec
    projection_norm = np.linalg.norm(projection)

    if projection_norm < 1e-10:  # Edge case: exactly aligned
        return f"({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}) {best_axis} axis"

    projection_unit = projection / projection_norm

    # Find which axis the projection is closest to (exclude the best_axis)
    best_toward = ""
    best_toward_dot = -1

    for axis_name, axis_vec in axis_vectors.items():
        if axis_name != best_axis:
            dot = np.dot(projection_unit, axis_vec)
            if dot > best_toward_dot:
                best_toward_dot = dot
                best_toward = axis_name

    return f"({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f}) {angle_deg:.0f}° rotated away from {best_axis} axis toward {best_toward}"


def validate_eye_camera_setup(eye_rest_orientation: RotationMatrix, camera_orientation: RotationMatrix) -> bool:
    """Validate that eye and camera orientations make sense for eye tracking.

    Checks if the orientations would allow the eye and camera to see each other.
    Note: point_at() automatically adjusts camera orientation, so this validates
    the intended setup geometry.

    Args:
        eye_rest_orientation: Eye rest orientation matrix
        camera_orientation: Camera orientation matrix

    Returns:
        True if setup is valid, False otherwise. Issues warnings for problems.
    """
    # Get facing directions
    eye_direction = _calculate_optical_axis_direction(eye_rest_orientation)
    camera_direction = _calculate_optical_axis_direction(camera_orientation)

    # Check if they're facing roughly opposite directions (dot product < 0)
    dot_product = np.dot(eye_direction, camera_direction)

    if dot_product > 0.1:  # Facing same direction (threshold for tolerance)
        warnings.warn(
            f"Suboptimal eye-camera orientation. The camera might not be able to capture the eye correctly as eye and camera are facing the same direction.\n"
            f"Eye:    {get_facing_direction(eye_rest_orientation)}\n"
            f"Camera: {get_facing_direction(camera_orientation)}",
            UserWarning,
        )
        return False
    elif abs(dot_product) < 0.1:  # Nearly perpendicular
        warnings.warn(
            f"Eye and camera are facing perpendicular directions.\n"
            f"Eye:    {get_facing_direction(eye_rest_orientation)}\n"
            f"Camera: {get_facing_direction(camera_orientation)}\n"
            f"Consider adjusting the setup for better eye tracking.",
            UserWarning,
        )
        return False

    return True
