"""
Eye operation functions extracted from the Eye class.

This module contains eye manipulation operations that were previously
part of the Eye class, extracted for better modularity and testability.
"""

import numpy as np
from typing import TYPE_CHECKING
from ..types import Position3D, Vector3D
from ..geometry.listings_law import calculate_eye_rotation

if TYPE_CHECKING:
    from .eye import Eye


def look_at_target(eye: "Eye", target_position: Position3D) -> None:
    """Rotates an eye to look at a given position in space.

    Uses Listing's law to compute eye rotation with proper torsion.
    Accounts for fovea displacement if enabled for realistic gaze alignment.

    Args:
        eye: Eye object to rotate
        target_position: Position in world coordinates to look at
    """
    # Get eye position as Position3D
    eye_position = eye.position

    # Calculate direction vector to target
    direction_vec = Vector3D(
        target_position.x - eye_position.x, target_position.y - eye_position.y, target_position.z - eye_position.z
    ).normalize()

    # Use Listing's law to compute eye rotation
    rest_optical_axis = Vector3D.from_array(eye._rest_orientation @ np.array([0, 0, -1]))
    eye.orientation = calculate_eye_rotation(rest_optical_axis, direction_vec) @ eye._rest_orientation

    # Compensate for fovea displacement if enabled
    if eye.fovea_displacement:
        # Use configurable fovea displacement angles
        alpha = eye.fovea_alpha_deg / 180 * np.pi  # Horizontal displacement
        beta = eye.fovea_beta_deg / 180 * np.pi  # Vertical displacement

        # Horizontal rotation matrix
        A = np.array(
            [
                [np.cos(alpha), 0, -np.sin(alpha)],
                [0, 1, 0],
                [np.sin(alpha), 0, np.cos(alpha)],
            ]
        )

        # Vertical rotation matrix
        B = np.array(
            [
                [1, 0, 0],
                [0, np.cos(beta), np.sin(beta)],
                [0, -np.sin(beta), np.cos(beta)],
            ]
        )

        # Apply fovea displacement compensation
        eye.orientation = eye.orientation @ B @ A
