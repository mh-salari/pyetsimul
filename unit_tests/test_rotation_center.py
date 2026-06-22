"""Unit tests for the gaze-direction-dependent eye rotation centre."""

import numpy as np
import numpy.testing as npt

from pyetsimul.core.eye import Eye
from pyetsimul.core.eye_model import EyeModel
from pyetsimul.core.rotation_center import RotationCenter
from pyetsimul.types import Position3D


def _xyz(position: Position3D) -> np.ndarray:
    return np.array([position.x, position.y, position.z], dtype=float)


def _apex_to_center(eye: Eye) -> float:
    """Current model rotation-centre depth: corneal apex to globe centre (mm)."""
    return abs(eye.cornea.get_apex_position().z)


def test_depth_for_blends_linearly() -> None:
    """depth_for interpolates between the vertical and horizontal depths by the horizontal fraction."""
    rc = RotationCenter(horizontal_depth_mm=15.0, vertical_depth_mm=12.5)
    npt.assert_allclose(rc.depth_for(0.0), 12.5)
    npt.assert_allclose(rc.depth_for(1.0), 15.0)
    npt.assert_allclose(rc.depth_for(0.5), 13.75)


def test_equal_depths_reproduce_single_center() -> None:
    """A rotation centre with both depths at the geometric apex-to-centre value == the default model."""
    target = Position3D(12000, -7000, -30000)
    placement = Position3D(5.0, 3.0, 40.0)

    baseline = Eye(model=EyeModel(fovea_displacement=False))
    baseline.position = placement
    baseline.look_at(target)

    d0 = _apex_to_center(Eye(model=EyeModel(fovea_displacement=False)))
    configured = Eye(model=EyeModel(fovea_displacement=False, rotation_center=RotationCenter(d0, d0)))
    configured.position = placement
    configured.look_at(target)

    npt.assert_allclose(configured.orientation, baseline.orientation, rtol=1e-12, atol=1e-12)
    npt.assert_allclose(_xyz(configured.position), _xyz(baseline.position), atol=1e-12)


def test_none_leaves_position_unchanged() -> None:
    """Without a rotation centre, look_at never moves the eye position (single fixed centre)."""
    eye = Eye(model=EyeModel(fovea_displacement=False))
    eye.position = Position3D(5.0, 3.0, 40.0)
    eye.look_at(Position3D(15.0, 10.0, 0.0))
    npt.assert_allclose(_xyz(eye.position), [5.0, 3.0, 40.0], atol=1e-12)


def test_horizontal_target_shifts_globe_center_toward_gaze() -> None:
    """A deeper horizontal centre swings the globe centre toward the (rightward) gaze, no vertical move."""
    d0 = _apex_to_center(Eye(model=EyeModel(fovea_displacement=False)))
    eye = Eye(
        model=EyeModel(
            fovea_displacement=False,
            rotation_center=RotationCenter(horizontal_depth_mm=d0 + 3.0, vertical_depth_mm=d0),
        )
    )
    eye.position = Position3D(0.0, 0.0, 40.0)

    eye.look_at(Position3D(20.0, 0.0, 0.0))  # purely horizontal, rightward

    assert eye.position.x > 1e-3  # globe centre moved toward the gaze
    assert abs(eye.position.y) < 1e-9  # no vertical component for a horizontal target


def test_vertical_target_at_geometric_depth_does_not_move() -> None:
    """A purely vertical target uses the vertical depth; set to the geometric value it is a no-op."""
    d0 = _apex_to_center(Eye(model=EyeModel(fovea_displacement=False)))
    eye = Eye(
        model=EyeModel(
            fovea_displacement=False,
            rotation_center=RotationCenter(horizontal_depth_mm=d0 + 3.0, vertical_depth_mm=d0),
        )
    )
    eye.position = Position3D(0.0, 0.0, 40.0)

    eye.look_at(Position3D(0.0, 20.0, 0.0))  # purely vertical

    npt.assert_allclose(_xyz(eye.position), [0.0, 0.0, 40.0], atol=1e-9)


def test_eye_still_aims_at_target_after_repivot() -> None:
    """After re-pivoting, the optical axis from the (moved) globe centre still passes through the target."""
    d0 = _apex_to_center(Eye(model=EyeModel(fovea_displacement=False)))  # optical axis == aimed axis
    eye = Eye(
        model=EyeModel(
            fovea_displacement=False,
            rotation_center=RotationCenter(horizontal_depth_mm=d0 + 3.0, vertical_depth_mm=d0 + 1.0),
        )
    )
    eye.position = Position3D(0.0, 0.0, 40.0)

    target = Position3D(15.0, 10.0, 0.0)
    eye.look_at(target)

    optical_axis = np.asarray(eye.orientation) @ np.array([0.0, 0.0, -1.0])
    to_target = _xyz(target) - _xyz(eye.position)
    to_target /= np.linalg.norm(to_target)
    npt.assert_allclose(optical_axis, to_target, atol=1e-6)
