"""Eye operation functions extracted from the Eye class.

This module contains eye manipulation operations that were previously
part of the Eye class, extracted for better modularity and testability.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np

from ..geometry.listings_law import calculate_eye_rotation
from ..types import Position3D, RotationMatrix, Vector3D
from .rotation_center import EyeballCenter

if TYPE_CHECKING:
    from .eye import Eye

# Pose solve: the aim direction is measured from a point of the eye that moves with the pose. The aimed
# axis passes through a point that turns with the eye, and an eye with a rotation centre behind its
# origin shifts as it turns. The aim direction therefore depends on the orientation it produces, and the
# two are iterated to a fixed point. Each step shrinks the error by about the ratio of the point's
# distance from the rotation centre to its distance from the target. A target at a viewing distance
# settles in under ten steps; a target about as near to the eye as its rotation centre does not settle.
_POSE_MAX_ITERS = 100
_POSE_TOL = 1e-13  # largest change of an orientation-matrix entry between two steps


def _roll_about_optical_axis(deg: float) -> np.ndarray:
    """Rotation about the eye-local optical axis (z), applied as a torsion deviation from Listing's law."""
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


LOOK_AT_METHODS = ("visual_axis", "line_of_sight", "optical_then_kappa", "optical_axis_target_direction")


def look_at_target(eye: "Eye", target_position: Position3D, method: str = "visual_axis") -> None:
    """Rotate the eye to look at a target by one of several gaze constructions.

    Each method aligns a rest-frame axis to the target with Listing's law and pivots about the model's
    rotation centre (a gaze-dependent ``RotationCenter`` or a fixed ``EyeballCenter``, see
    ``rotation_center.py``). They differ only in which axis is aimed, where the aim direction is measured
    from, and the post-rotation applied:

    - ``visual_axis``: the visual axis (the optical axis when fovea displacement is off) to the target.
    - ``line_of_sight``: the fovea-through-pupil-centre axis to the target; the pupil centre carries the
      off-axis offset and the size-dependent decentration, so this axis re-aims as the pupil decentres.
    - ``optical_then_kappa``: the optical axis to the target, then a foveal (kappa) post-rotation so that
      neither the optical nor the visual axis ends up through the target (Böhme et al. 2008 et_simul).
    - ``optical_axis_target_direction``: the optical axis along the apex-to-target direction taken from the
      fixed rest apex, so the pose is the azimuth/elevation of that direction, independent of target distance,
      and the axis does not pass through a finite target (the eyePose convention of gkaModelEye).

    Args:
        eye: Eye object to rotate
        target_position: Position in world coordinates to look at
        method: Gaze construction to use; one of ``LOOK_AT_METHODS``.

    Raises:
        ValueError: If ``method`` is unrecognised, the target coincides with the eye (zero-length direction),
            or the target is too close to the eye for the pose to settle.

    """
    if method not in LOOK_AT_METHODS:
        raise ValueError(f"unknown look_at method {method!r}; expected one of {LOOK_AT_METHODS}")

    # Rest-frame axis aligned to the target, and the eye-local point that axis passes through. Both depend
    # only on the rest orientation and the eye's anatomy, so they are fixed while the pose is solved.
    if method == "line_of_sight":
        # The line of sight runs from the fovea through the current pupil centre.
        fovea, pupil = eye.fovea_position, eye.pupil.pos_pupil
        los = np.array([pupil.x - fovea.x, pupil.y - fovea.y, pupil.z - fovea.z])
        los /= np.linalg.norm(los)
        rest_axis = Vector3D.from_array(eye.rest_orientation @ los)
        axis_point = axis_point_nearest_origin(np.array([pupil.x, pupil.y, pupil.z], dtype=float), los)
    elif method == "visual_axis" and eye.model.fovea_displacement:
        # Visual axis from the foveal displacement angles (alpha horizontal, signed by eye side; beta vertical).
        # It runs from the fovea through the eyeball centre.
        alpha, beta = eye._signed_alpha_rad, np.radians(eye.model.fovea_beta_deg)
        v_local = np.array([-np.sin(alpha) * np.cos(beta), -np.sin(beta), -np.cos(alpha) * np.cos(beta)])
        rest_axis = Vector3D.from_array(eye.rest_orientation @ v_local)
        axis_point = axis_point_nearest_origin(np.array([0.0, 0.0, eye.eyeball_center_z], dtype=float), v_local)
    else:
        # Optical axis is -Z in local coordinates and passes through the eye-local origin.
        rest_axis = Vector3D.from_array(eye.rest_orientation @ np.array([0.0, 0.0, -1.0]))
        axis_point = np.zeros(3)

    rest = np.asarray(eye.rest_orientation, dtype=float)
    origin_for = _origin_for_orientation(eye, target_position)

    def orientation_facing(direction: Vector3D) -> np.ndarray:
        return calculate_eye_rotation(rest_axis, direction) @ rest

    def posed(aimed: np.ndarray) -> np.ndarray:
        """Eye orientation for an aimed orientation: the rotation the method applies after the aim."""
        if method == "optical_then_kappa" and eye.model.fovea_displacement:
            # Foveal (kappa) post-rotation lands the fovea on the target after the optical axis is aimed.
            alpha, beta = eye._signed_alpha_rad, np.radians(eye.model.fovea_beta_deg)
            rot_x = np.array([
                [np.cos(alpha), 0.0, -np.sin(alpha)],
                [0.0, 1.0, 0.0],
                [np.sin(alpha), 0.0, np.cos(alpha)],
            ])
            rot_y = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(beta), np.sin(beta)], [0.0, -np.sin(beta), np.cos(beta)]])
            return aimed @ rot_y @ rot_x
        if eye.model.torsion_deg and method in {"visual_axis", "optical_axis_target_direction"}:
            return aimed @ _roll_about_optical_axis(eye.model.torsion_deg)
        return aimed

    if method == "optical_axis_target_direction":
        # The direction is taken from the fixed rest apex, so the pose does not depend on the target
        # distance and the axis does not pass through a finite target.
        aimed = orientation_facing(_direction_to_target(target_position, eye.placement))
    else:
        # The direction is taken from the point of the posed eye that the aimed axis passes through, so
        # that the axis itself passes through the target.
        aimed = orientation_through_target(
            orientation_facing, rest, axis_point, lambda orientation: origin_for(posed(orientation)), target_position
        )
    orientation = posed(aimed)
    eye.orientation = RotationMatrix(orientation, validate_handedness=False)
    # Write the translation directly so the rest placement (which anchors the rotation centre) is kept.
    eye.trans[:3, 3] = origin_for(orientation)


# ---------------------------------------------------------------------------
# Pose solve and the rotation centre
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


def axis_point_nearest_origin(point: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Point nearest the eye-local origin on the line through ``point`` along the unit ``direction``.

    Any point of the aimed axis gives the same pose; the aim solve settles fastest with the nearest one.
    """
    return point - (point @ direction) * direction


def orientation_through_target(
    orientation_facing: Callable[[Vector3D], np.ndarray],
    start: np.ndarray,
    axis_point: np.ndarray,
    origin_for: Callable[[np.ndarray], np.ndarray],
    target_position: Position3D,
) -> np.ndarray:
    """Orientation for which the aimed axis passes through the target.

    ``orientation_facing`` gives the orientation that points the aimed axis along a world direction. The
    axis passes through ``axis_point`` (eye-local), and ``origin_for`` gives the world position of the
    eye-local origin for an orientation. The world position of the axis point therefore depends on the
    orientation, so the direction from that point to the target and the orientation are iterated to a
    fixed point, beginning at ``start``.

    Raises:
        ValueError: If the iteration does not settle, which happens for a target about as near to the eye
            as the point the eye turns about.

    """
    orientation = start
    for _ in range(_POSE_MAX_ITERS):
        point = origin_for(orientation) + orientation @ axis_point
        direction = _direction_to_target(target_position, Position3D(point[0], point[1], point[2]))
        updated = np.asarray(orientation_facing(direction), dtype=float)
        if np.max(np.abs(updated - orientation)) < _POSE_TOL:
            return updated
        orientation = updated
    raise ValueError(
        f"Cannot look at target {target_position}: the eye pose did not settle within {_POSE_MAX_ITERS} steps. "
        f"The target is too close to the eye."
    )


def _origin_for_orientation(eye: "Eye", target_position: Position3D) -> Callable[[np.ndarray], np.ndarray]:
    """World position of the eye-local origin as a function of the eye orientation.

    An ``EyeballCenter`` leaves the origin where it is. A ``RotationCenter`` turns the eye about a centre
    behind the corneal apex that is anchored to the rest placement, so the origin shifts as the eye turns.
    """
    rotation_center = eye.model.rotation_center
    if isinstance(rotation_center, EyeballCenter):
        origin = np.array([eye.position.x, eye.position.y, eye.position.z], dtype=float)
        return lambda _orientation: origin
    if rotation_center.fick:
        return _fick_origin(eye)
    return _blended_centre_origin(eye, target_position)


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


def _blended_centre_origin(eye: "Eye", target_position: Position3D) -> Callable[[np.ndarray], np.ndarray]:
    """Origin of an eye that turns about one rotation centre whose depth depends on the gaze direction.

    The centre lies ``depth`` mm behind the corneal apex, where ``depth`` blends the horizontal and
    vertical rotation-centre depths by the horizontal fraction of the gaze. The centre is anchored to the
    rest placement, so the eye-local origin translates as the eye turns to off-axis targets.
    """
    placement = np.array([eye.placement.x, eye.placement.y, eye.placement.z], dtype=float)
    rest = np.asarray(eye.rest_orientation, dtype=float)
    # Apex depth below the origin: ~axial_length/2 under "center", ~0 under "apex" (origin = apex);
    # offset.z = depth - apex_to_center keeps the pivot depth mm behind the apex under either convention.
    apex_to_center = abs(eye.cornea.get_apex_position().z)
    horizontal_fraction = _horizontal_fraction(eye, target_position)
    gaze_rest = rest.T @ (np.array([target_position.x, target_position.y, target_position.z], dtype=float) - placement)
    depth = eye.model.rotation_center.depth_for(horizontal_fraction, gaze_rest[1])  # gaze_rest[1] >= 0 up, < 0 down
    lateral_x, lateral_y = eye.model.rotation_center.lateral_for(horizontal_fraction, eye.which_eye)
    # Pivot offset in eye-local coordinates: lateral (x nasal-signed, y superior) plus axial (+z deeper).
    offset = np.array([lateral_x, lateral_y, depth - apex_to_center], dtype=float)

    def origin(orientation: np.ndarray) -> np.ndarray:
        return placement + (rest - orientation) @ offset

    return origin


def _fick_origin(eye: "Eye") -> Callable[[np.ndarray], np.ndarray]:
    """Origin of an eye with separate Fick rotation centres for azimuth and elevation (gkaModelEye / Aguirre 2021).

    Azimuth rotates about the horizontal (azimuth) centre and elevation about the vertical (elevation)
    centre, applied sequentially, rather than blending both into one pivot. With both centres equal this
    gives the same origin as one blended centre.
    """
    placement = np.array([eye.placement.x, eye.placement.y, eye.placement.z], dtype=float)
    rest = np.asarray(eye.rest_orientation, dtype=float)
    apex_to_center = abs(eye.cornea.get_apex_position().z)
    rc = eye.model.rotation_center
    c_azi = eye._azimuth_rotation_centre_local()
    identity = np.eye(3)

    def origin(orientation: np.ndarray) -> np.ndarray:
        fwd = (rest.T @ orientation) @ np.array([0.0, 0.0, -1.0])  # rotated optical axis in the rest frame
        elev = float(np.arcsin(np.clip(fwd[1], -1.0, 1.0)))  # Fick elevation (about the horizontal axis)
        azim = float(np.arctan2(-fwd[0], -fwd[2]))  # Fick azimuth (about the vertical axis)
        # Elevation centre depth follows the gaze-varying vertical centre (up vs down) at the current elevation.
        c_ele = np.array([0.0, rc.vertical_superior_mm, rc.vertical_depth(elev) - apex_to_center], dtype=float)
        ca, sa, ce, se = np.cos(azim), np.sin(azim), np.cos(elev), np.sin(elev)
        r_azi = np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])
        r_ele = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
        t_fick = r_azi @ (identity - r_ele) @ c_ele + (identity - r_azi) @ c_azi  # eye-local translation
        return placement + rest @ t_fick

    return origin
