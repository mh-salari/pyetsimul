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


def _roll_about_optical_axis(deg: float) -> np.ndarray:
    """Rotation about the eye-local optical axis (z), applied as a torsion deviation from Listing's law."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def look_at_target(eye: "Eye", target_position: Position3D) -> None:
    """Rotates an eye to look at a given position in space.

    Uses Listing's law to compute eye rotation with proper torsion.
    Accounts for fovea displacement if enabled for realistic gaze alignment.
    When ``eye.model.rotation_center`` is set, the eye also pivots about a gaze-direction-dependent
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
    if eye.model.fovea_displacement:
        # Local visual axis direction (unit), pointing anteriorly (toward cornea), derived from the
        # fovea displacement angles (alpha: horizontal, signed by eye side; beta: vertical).
        alpha = eye._signed_alpha_rad
        beta = eye.model.fovea_beta_deg * np.pi / 180.0
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
        rotation = calculate_eye_rotation(rest_axis, direction) @ eye.rest_orientation
        if not eye.model.torsion_deg:
            return rotation
        return rotation @ _roll_about_optical_axis(eye.model.torsion_deg)

    _apply_orientation(eye, target_position, orientation_from)


def look_at_target_optical_then_kappa(eye: "Eye", target_position: Position3D) -> None:
    """Rotate the eye using optical-axis alignment followed by kappa offsets.

    Duplicates Böhme et al. (2008) original MATLAB implementation of look_at.
    Aligns the optical axis to the target first, then applies foveal (kappa)
    offsets via post-rotations. The post step rotates the eye away so that
    neither the optical axis nor the visual axis ends up passing exactly
    through the target. Honours ``eye.model.rotation_center`` when set.

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
        if eye.model.fovea_displacement:
            alpha = eye._signed_alpha_rad  # signed by eye side: the fovea is temporal in both eyes
            beta = eye.model.fovea_beta_deg * np.pi / 180.0

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


def look_at_target_line_of_sight(eye: "Eye", target_position: Position3D) -> None:
    """Rotate the eye so the line of sight -- fovea through the pupil centre -- points at the target.

    ``look_at_target`` aligns the visual axis (fovea to the eye centre) to the target; this aligns the line
    from the fovea through the current pupil centre instead. Because the pupil centre carries the off-axis
    offset and the size-dependent decentration, this axis moves when the pupil decentres, so the eye re-aims
    as the pupil moves; the visual axis does not.

    Raises:
        ValueError: If target_position coincides with eye position (zero-length vector).

    """
    fovea = eye.fovea_position
    pupil = eye.pupil.pos_pupil
    los = np.array([pupil.x - fovea.x, pupil.y - fovea.y, pupil.z - fovea.z])
    rest_axis = Vector3D.from_array(eye.rest_orientation @ (los / np.linalg.norm(los)))

    def orientation_from(eye_position: Position3D) -> np.ndarray:
        direction = _direction_to_target(target_position, eye_position)
        return calculate_eye_rotation(rest_axis, direction) @ eye.rest_orientation

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

    Without ``eye.model.rotation_center`` the orientation is computed once from the current eye position
    and the position is left untouched (the single fixed centre at the globe centre). With it, the
    eye additionally translates so it rotates about the gaze-direction-dependent centre.
    """
    if eye.model.rotation_center is None:
        new_orientation = orientation_from(eye.position)
        eye.orientation = RotationMatrix(new_orientation, validate_handedness=False)
        return
    repivot = _repivot_fick if eye.model.rotation_center.fick else _repivot
    new_orientation, position = repivot(eye, target_position, orientation_from)
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
    horizontal_fraction = _horizontal_fraction(eye, target_position)
    gaze_rest = rest.T @ (np.array([target_position.x, target_position.y, target_position.z], dtype=float) - placement)
    depth = eye.model.rotation_center.depth_for(horizontal_fraction, gaze_rest[1])  # gaze_rest[1] >= 0 up, < 0 down
    lateral_x, lateral_y = eye.model.rotation_center.lateral_for(horizontal_fraction, eye.which_eye)
    # Pivot offset in eye-local coordinates: lateral (x nasal-signed, y superior) plus axial (+z deeper).
    offset = np.array([lateral_x, lateral_y, depth - apex_to_center], dtype=float)

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


def _repivot_fick(
    eye: "Eye", _target_position: Position3D, orientation_from: Callable[[Position3D], np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """Position via separate Fick rotation centres for azimuth and elevation (gkaModelEye / Aguirre 2019).

    Azimuth rotates about the horizontal (azimuth) centre and elevation about the vertical (elevation)
    centre, applied sequentially, rather than blending both into one pivot. The orientation is the same
    Listing-law aiming as the single-pivot path; only the translation differs. With both centres equal this
    reduces to ``_repivot`` exactly.
    """
    placement = np.array([eye.placement.x, eye.placement.y, eye.placement.z], dtype=float)
    rest = np.asarray(eye.rest_orientation, dtype=float)
    apex_to_center = abs(eye.cornea.get_apex_position().z)
    rc = eye.model.rotation_center
    nasal = -rc.horizontal_nasal_mm if eye.which_eye == "right" else rc.horizontal_nasal_mm
    c_azi = np.array([nasal, 0.0, rc.horizontal_depth_mm - apex_to_center], dtype=float)
    identity = np.eye(3)

    position = placement.copy()
    orientation = rest
    for _ in range(_REPIVOT_MAX_ITERS):
        orientation = np.asarray(orientation_from(Position3D(position[0], position[1], position[2])), dtype=float)
        fwd = (rest.T @ orientation) @ np.array([0.0, 0.0, -1.0])  # rotated optical axis in the rest frame
        elev = float(np.arcsin(np.clip(fwd[1], -1.0, 1.0)))  # Fick elevation (about the horizontal axis)
        azim = float(np.arctan2(-fwd[0], -fwd[2]))  # Fick azimuth (about the vertical axis)
        # Elevation centre depth follows the gaze-varying vertical centre (up vs down) at the current elevation.
        c_ele = np.array([0.0, rc.vertical_superior_mm, rc.vertical_depth(elev) - apex_to_center], dtype=float)
        ca, sa, ce, se = np.cos(azim), np.sin(azim), np.cos(elev), np.sin(elev)
        r_azi = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
        r_ele = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
        t_fick = r_azi @ (identity - r_ele) @ c_ele + (identity - r_azi) @ c_azi  # eye-local translation
        updated = placement + rest @ t_fick
        if np.max(np.abs(updated - position)) < _REPIVOT_TOL:
            position = updated
            break
        position = updated
    return orientation, position
