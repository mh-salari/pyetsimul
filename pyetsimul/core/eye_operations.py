"""Eye operation functions extracted from the Eye class.

This module contains eye manipulation operations that were previously
part of the Eye class, extracted for better modularity and testability.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.listings_law import calculate_eye_rotation
from ..types import Position3D, RotationMatrix, Vector3D

if TYPE_CHECKING:
    from .eye import Eye

# Re-pivot solve (gaze-dependent rotation centre): the aiming direction depends on where the globe
# centre sits after pivoting, and the pivot depends on the gaze direction, so orientation and
# position are iterated to a fixed point. The shift is sub-millimetre, so this converges in a couple
# of steps.
_REPIVOT_MAX_ITERS = 8
_REPIVOT_TOL = 1e-9  # mm; stop once the globe-centre position stops moving


def look_at_target(eye: "Eye", target_position: Position3D) -> None:
    """Rotates an eye to look at a given position in space.

    Uses Listing's law to compute eye rotation with proper torsion.
    Accounts for fovea displacement if enabled for realistic gaze alignment.
    When ``eye.rotation_center`` is set, the eye also pivots about a gaze-direction-dependent
    rotation centre rather than the single fixed globe centre (see ``rotation_center.py``).

    Args:
        eye: Eye object to rotate
        target_position: Position in world coordinates to look at

    Raises:
        ValueError: If target_position is the same as eye position (zero-length direction vector)

    """
    # Choose which local axis to align to target: visual axis if fovea displacement is enabled,
    # otherwise the optical axis (-Z). This depends only on the rest orientation, not the eye
    # position, so it is computed once and reused across the re-pivot iterations.
    if eye.fovea_displacement:
        # Local visual axis direction (unit), pointing anteriorly (toward cornea), derived from the
        # fovea displacement angles (alpha: horizontal, beta: vertical).
        alpha = eye.fovea_alpha_deg * np.pi / 180.0
        beta = eye.fovea_beta_deg * np.pi / 180.0
        v_local = np.array([
            -np.sin(alpha) * np.cos(beta),
            -np.sin(beta),
            -np.cos(alpha) * np.cos(beta),
        ])
        rest_axis = Vector3D.from_array(eye.rest_orientation @ v_local)
    else:
        # Optical axis in local coordinates is -Z
        rest_axis = Vector3D.from_array(eye.rest_orientation @ np.array([0.0, 0.0, -1.0]))

    def orientation_from(eye_position: Position3D) -> np.ndarray:
        # Use Listing's law to compute the rotation that aligns the chosen axis with the direction
        # from this eye position to the target.
        direction = _direction_to_target(target_position, eye_position)
        return calculate_eye_rotation(rest_axis, direction) @ eye.rest_orientation

    _apply_orientation(eye, target_position, orientation_from)


def look_at_target_optical_then_kappa(eye: "Eye", target_position: Position3D) -> None:
    """Rotate the eye using optical-axis alignment followed by kappa offsets.

    Duplicates Böhme et al. (2008) original MATLAB implementation of look_at.
    Aligns the optical axis to the target first, then applies foveal (kappa)
    offsets via post-rotations. The post step rotates the eye away so that
    neither the optical axis nor the visual axis ends up passing exactly
    through the target. Honours ``eye.rotation_center`` when set.

    Args:
        eye: Eye object to rotate
        target_position: Position in world coordinates to look at

    Raises:
        ValueError: If target_position coincides with eye position (zero-length vector)

    """
    # Rest optical axis (-Z in local) in world coordinates; independent of the eye position.
    rest_optical_axis = Vector3D.from_array(eye.rest_orientation @ np.array([0.0, 0.0, -1.0]))

    def orientation_from(eye_position: Position3D) -> np.ndarray:
        # First align the optical axis to the target using Listing's law.
        direction = _direction_to_target(target_position, eye_position)
        orientation = calculate_eye_rotation(rest_optical_axis, direction) @ eye.rest_orientation

        # Then apply post-rotations from foveal displacement (kappa) if enabled.
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
        return orientation

    _apply_orientation(eye, target_position, orientation_from)


# ---------------------------------------------------------------------------
# Orientation application and the gaze-dependent rotation centre
# ---------------------------------------------------------------------------


def _direction_to_target(target_position: Position3D, eye_position: Position3D) -> Vector3D:
    """Unit direction from ``eye_position`` to ``target_position`` (raises on a zero-length vector)."""
    direction = Vector3D(
        target_position.x - eye_position.x,
        target_position.y - eye_position.y,
        target_position.z - eye_position.z,
    )
    if direction.magnitude() == 0:
        raise ValueError(
            f"Cannot look at target: direction vector has zero length. "
            f"Target position {target_position} cannot be the same as eye position {eye_position}."
        )
    return direction.normalize()


def _apply_orientation(
    eye: "Eye", target_position: Position3D, orientation_from: Callable[[Position3D], np.ndarray]
) -> None:
    """Set the eye orientation; with a rotation centre configured, also re-pivot the eye position.

    Without ``eye.rotation_center`` the orientation is computed once from the current eye position
    and the position is left untouched (the single fixed centre at the globe centre). With it, the
    eye additionally translates so it rotates about the gaze-direction-dependent centre.
    """
    if eye.rotation_center is None:
        new_orientation = orientation_from(eye.position)
        eye.orientation = RotationMatrix(new_orientation, validate_handedness=False)
        return
    new_orientation, position = _repivot(eye, target_position, orientation_from)
    eye.orientation = RotationMatrix(new_orientation, validate_handedness=False)
    # Write the translation directly so the rest placement (used to re-pivot) is not overwritten.
    eye.trans[:3, 3] = position


def _horizontal_fraction(eye: "Eye", target_position: Position3D) -> float:
    """Horizontal share of the gaze eccentricity in the eye's rest frame, in ``[0, 1]``.

    The gaze direction (rest placement to target) is expressed in the rest frame (x right, y up,
    -z forward); 1 is a purely horizontal target, 0 a purely vertical one.
    """
    rest = np.asarray(eye.rest_orientation, dtype=float)
    placement = eye.placement
    rel = rest.T @ np.array(
        [
            target_position.x - placement.x,
            target_position.y - placement.y,
            target_position.z - placement.z,
        ],
        dtype=float,
    )
    horizontal = rel[0] * rel[0]
    denom = horizontal + rel[1] * rel[1]
    return horizontal / denom if denom > 1e-12 else 0.0


def _repivot(
    eye: "Eye", target_position: Position3D, orientation_from: Callable[[Position3D], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Solve orientation and position together for the gaze-direction-dependent rotation centre.

    The rigid eye rotates about a point ``depth`` mm behind the corneal apex, where ``depth`` blends
    the horizontal and vertical rotation-centre depths by the horizontal fraction of the gaze. The
    pivot is anchored to the rest placement, so the globe centre translates as the eye turns to
    off-axis targets. Orientation and position depend on each other (aiming uses the translated globe
    centre), so they are iterated to a fixed point.

    Returns:
        (orientation matrix (3x3), eye-position xyz array (3,)).

    """
    placement = np.array([eye.placement.x, eye.placement.y, eye.placement.z], dtype=float)
    rest = np.asarray(eye.rest_orientation, dtype=float)
    apex_to_center = abs(eye.cornea.get_apex_position().z)  # current model centre depth (apex -> globe centre)
    depth = eye.rotation_center.depth_for(_horizontal_fraction(eye, target_position))
    # Pivot offset along the optical axis in eye-local coordinates (+z toward the retina, i.e. deeper).
    offset = np.array([0.0, 0.0, depth - apex_to_center], dtype=float)

    position = placement.copy()
    orientation = rest
    for _ in range(_REPIVOT_MAX_ITERS):
        orientation = np.asarray(orientation_from(Position3D(position[0], position[1], position[2])), dtype=float)
        updated = placement + (rest - orientation) @ offset
        if np.max(np.abs(updated - position)) < _REPIVOT_TOL:
            position = updated
            break
        position = updated
    return orientation, position
