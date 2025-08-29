"""Coordinate and gaze conversion utilities for eye tracking simulation.

Implements conversions between gaze direction, rotation angles, and observer/screen coordinates.
"""

import numpy as np
from typing import Optional
from ..types import Point2D, Point3D, Direction3D, Position3D, RotationMatrix


def gaze2angle(gaze: Direction3D, rest_pos: Optional[RotationMatrix] = None) -> Point2D:
    """Convert gaze direction to eye rotation angles.

    Calculates horizontal and vertical rotation angles from a 3D gaze direction
    relative to the eye's rest position. Uses Listing's law coordinate system.

    Args:
        gaze: 3D gaze direction vector
        rest_pos: Optional 3x3 rotation matrix for eye rest position.
                 Defaults to [[1,0,0], [0,0,1], [0,1,0]]

    Returns:
        Point2D containing [horizontal_angle, vertical_angle] in radians
    """
    if not isinstance(gaze, Direction3D):
        raise TypeError(f"gaze must be Direction3D, got {type(gaze)}")

    # Default rest position if not provided
    if rest_pos is None:
        rest_pos = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    # Transform gaze to rest position coordinate system
    gaze_transformed = np.linalg.solve(rest_pos, np.array(gaze)[:3])

    # Calculate rotation angles using arctan2 for proper quadrant handling
    horizontal_angle = np.arctan2(gaze_transformed[0], -gaze_transformed[2])
    vertical_angle = np.arctan2(gaze_transformed[1], np.linalg.norm([gaze_transformed[0], gaze_transformed[2]]))

    return Point2D(horizontal_angle, vertical_angle)


def angle2gaze(angles: Point2D, rest_pos: Optional[RotationMatrix] = None) -> Direction3D:
    """Convert eye rotation angles to gaze direction.

    Calculates gaze direction from horizontal and vertical rotation angles
    using Euler rotation matrices. Applies rotations in Listing's law order.

    Args:
        angles: 2D point containing rotation angles [horizontal, vertical] in radians
        rest_pos: Optional 3x3 rotation matrix for eye rest position.
                 Defaults to [[1,0,0], [0,0,1], [0,1,0]]

    Returns:
        Direction3D representing the gaze direction vector
    """
    if not isinstance(angles, Point2D):
        raise TypeError(f"angles must be Point2D, got {type(angles)}")

    # Default rest position if not provided
    if rest_pos is None:
        rest_pos = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])

    # Create rotation matrices for horizontal and vertical rotations
    angles_arr = np.array(angles)
    cos_h, sin_h = np.cos(angles_arr[0]), np.sin(angles_arr[0])
    cos_v, sin_v = np.cos(angles_arr[1]), np.sin(angles_arr[1])

    # Horizontal rotation (x-z plane)
    rot_h = np.array([[cos_h, 0, -sin_h, 0], [0, 1, 0, 0], [sin_h, 0, cos_h, 0], [0, 0, 0, 1]])
    # Vertical rotation (y-z plane)
    rot_v = np.array([[1, 0, 0, 0], [0, cos_v, -sin_v, 0], [0, sin_v, cos_v, 0], [0, 0, 0, 1]])

    # Convert rest position to 4x4 homogeneous matrix
    rest_pos_4x4 = np.eye(4)
    rest_pos_4x4[:3, :3] = rest_pos

    # Apply transformations: rest_pos * rot_h * rot_v * default_gaze
    default_gaze = np.array([0, 0, -1, 0])
    gaze_4d = rest_pos_4x4 @ rot_h @ rot_v @ default_gaze

    return Direction3D.from_array(gaze_4d[:3])


def calculate_angular_error_degrees(
    actual_point: Point3D, predicted_point: Point3D, observer_pos: Position3D
) -> float:
    """Calculate angular error between actual and predicted gaze points.

    Creates 3D gaze vectors from observer to each point and computes the angle
    between them using the dot product formula. Handles numerical precision issues.

    Args:
        actual_point: Actual target point [x, y, z] in meters
        predicted_point: Predicted gaze point [x, y, z] in meters
        observer_pos: Observer position [x, y, z] in meters

    Returns:
        Angular error in degrees
    """
    if not isinstance(actual_point, Point3D):
        raise TypeError(f"actual_point must be Point3D, got {type(actual_point)}")
    if not isinstance(predicted_point, Point3D):
        raise TypeError(f"predicted_point must be Point3D, got {type(predicted_point)}")
    if not isinstance(observer_pos, Position3D):
        raise TypeError(f"observer_pos must be Position3D, got {type(observer_pos)}")

    # Create 3D gaze vectors from observer to target points using structured types
    gaze_actual = actual_point - observer_pos
    gaze_predicted = predicted_point - observer_pos

    # Calculate vector magnitudes
    actual_magnitude = np.sqrt(gaze_actual.x**2 + gaze_actual.y**2 + gaze_actual.z**2)
    predicted_magnitude = np.sqrt(gaze_predicted.x**2 + gaze_predicted.y**2 + gaze_predicted.z**2)

    # Calculate dot product manually using structured type components
    dot_product = (
        gaze_actual.x * gaze_predicted.x + gaze_actual.y * gaze_predicted.y + gaze_actual.z * gaze_predicted.z
    )

    # Normalize dot product
    normalized_dot = dot_product / (actual_magnitude * predicted_magnitude)

    # Clip for numerical stability and calculate angle
    normalized_dot = np.clip(normalized_dot, -1, 1)
    return float(np.degrees(np.arccos(normalized_dot)))
