"""Eye operation functions extracted from the Eye class.

This module contains eye manipulation operations that were previously
part of the Eye class, extracted for better modularity and testability.
"""

from typing import TYPE_CHECKING

import numpy as np

from ..geometry.listings_law import calculate_eye_rotation
from ..types import Position3D, RotationMatrix, Vector3D

if TYPE_CHECKING:
    from .eye import Eye


def look_at_target(eye: "Eye", target_position: Position3D) -> None:
    """Rotates an eye to look at a given position in space.

    Uses Listing's law to compute eye rotation with proper torsion.
    Accounts for fovea displacement if enabled for realistic gaze alignment.

    Args:
        eye: Eye object to rotate
        target_position: Position in world coordinates to look at

    Raises:
        ValueError: If target_position is the same as eye position (zero-length direction vector)

    """
    # Get eye position as Position3D
    eye_position = eye.position

    # Calculate direction vector to target
    direction_vec = Vector3D(
        target_position.x - eye_position.x, target_position.y - eye_position.y, target_position.z - eye_position.z
    )

    # Check for zero-length direction vector
    if direction_vec.magnitude() == 0:
        raise ValueError(
            f"Cannot look at target: direction vector has zero length. "
            f"Target position {target_position} cannot be the same as eye position {eye_position}."
        )

    direction_vec = direction_vec.normalize()

    # Choose which local axis to align to target: visual axis if fovea displacement is enabled,
    # otherwise the optical axis (-Z).
    if eye.fovea_displacement:
        # Local visual axis direction (unit), pointing anteriorly (toward cornea)
        # Derived from fovea displacement angles (alpha: horizontal, beta: vertical)
        alpha = eye.fovea_alpha_deg * np.pi / 180.0
        beta = eye.fovea_beta_deg * np.pi / 180.0
        # Unit vector components (see Eye.point_rest_orientation_at_target)
        v_local = np.array([
            -np.sin(alpha) * np.cos(beta),
            -np.sin(beta),
            -np.cos(alpha) * np.cos(beta),
        ])
        rest_axis = Vector3D.from_array(eye.rest_orientation @ v_local)
    else:
        # Optical axis in local coordinates is -Z
        rest_axis = Vector3D.from_array(eye.rest_orientation @ np.array([0.0, 0.0, -1.0]))

    # Use Listing's law to compute eye rotation that aligns chosen axis with the target direction
    new_orientation = calculate_eye_rotation(rest_axis, direction_vec) @ eye.rest_orientation
    # Preserve handedness from rest_orientation (allow left-handed if rest is left-handed)
    eye.orientation = RotationMatrix(new_orientation, validate_handedness=False)


def look_at_target_optical_then_kappa(eye: "Eye", target_position: Position3D) -> None:
    """Rotate the eye using optical-axis alignment followed by kappa offsets.

    Duplicates Böhme et al. (2008) original MATLAB implementation of look_at.
    Aligns the optical axis to the target first, then applies foveal (kappa)
    offsets via post-rotations. The post step rotates the eye away so that
    neither the optical axis nor the visual axis ends up passing exactly
    through the target.

    Args:
        eye: Eye object to rotate
        target_position: Position in world coordinates to look at

    Raises:
        ValueError: If target_position coincides with eye position (zero-length vector)

    """
    # Eye position
    eye_position = eye.position

    # Direction to target (world)
    direction_vec = Vector3D(
        target_position.x - eye_position.x,
        target_position.y - eye_position.y,
        target_position.z - eye_position.z,
    )

    if direction_vec.magnitude() == 0:
        raise ValueError(
            f"Cannot look at target: direction vector has zero length. "
            f"Target position {target_position} cannot be the same as eye position {eye_position}."
        )

    direction_vec = direction_vec.normalize()

    # First align optical axis (-Z in local) to the target using Listing's law
    rest_optical_axis = Vector3D.from_array(eye.rest_orientation @ np.array([0.0, 0.0, -1.0]))
    orientation = calculate_eye_rotation(rest_optical_axis, direction_vec) @ eye.rest_orientation

    # Then apply post-rotations from foveal displacement (kappa) if enabled
    if eye.fovea_displacement:
        alpha = eye.fovea_alpha_deg * np.pi / 180.0
        beta = eye.fovea_beta_deg * np.pi / 180.0

        rotation_matrix_x = np.array([
            [np.cos(alpha), 0.0, -np.sin(alpha)],
            [0.0, 1.0, 0.0],
            [np.sin(alpha), 0.0, np.cos(alpha)],
        ])

        rotation_matrix_y = np.array([
            [1.0, 0.0, 0.0],
            [0.0, np.cos(beta), np.sin(beta)],
            [0.0, -np.sin(beta), np.cos(beta)],
        ])

        orientation = orientation @ rotation_matrix_y @ rotation_matrix_x

    eye.orientation = RotationMatrix(orientation, validate_handedness=False)
