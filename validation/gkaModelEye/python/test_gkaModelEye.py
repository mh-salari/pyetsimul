"""1:1 validation of the ``gkaModelEye`` named model against the gkaModelEye reference.

``../matlab/run_reference.m`` generates ``../reference.json`` with gkaModelEye; this test rebuilds the same
scene with ``Eye(model="gkaModelEye")`` and asserts the pupil centre and glint match per gaze target. The eye
is posed by the model's own eyePose convention (``optical_axis_target_direction``) and imaged through a Camera
in FickRotation mode that reproduces gkaModelEye's non-orthonormal projection, so the comparison isolates the
eye optics.
"""

import json
import warnings
from itertools import starmap
from pathlib import Path

import numpy as np

import pyetsimul.core.eye_models  # noqa: F401  (imported for its registration side effect)
from pyetsimul.core.camera import Camera
from pyetsimul.core.eye import Eye
from pyetsimul.core.eye_model import get_eye_model
from pyetsimul.core.light import Light
from pyetsimul.optics.pupil_imaging import calculate_pupil_center_from_boundary
from pyetsimul.types import CameraMatrix, Point2D, Position3D

warnings.filterwarnings("ignore")

REFERENCE = Path(__file__).resolve().parent.parent / "reference.json"
# gkaModelEye is an independent model, reproduced only to the iterative-solver floor (the fsolve glint
# reflection and the refraction batch), so the agreement is sub-pixel rather than the exact 1e-3 the
# et_simul reproduction reaches.
TOLERANCE_PX = 0.05


def build_eye(eye_pos: np.ndarray) -> Eye:
    """A gkaModelEye eye placed by its apex at ``eye_pos``, kappa zeroed so the optical axis aims at the target.

    Zeroing the displacement angles makes ``look_at`` point the optical axis straight at the target, matching
    a gaze driven by bare azimuth/elevation. The cornea, indices and rotation centres stay as the model spec.
    """
    model = get_eye_model("gkaModelEye").copy(fovea_alpha_deg=0.0, fovea_beta_deg=0.0)
    eye = Eye(model=model, which_eye="right")
    eye.position = Position3D(*eye_pos)
    # The lab frame is +Z up (Y is depth), so the rest orientation must be told its up axis.
    eye.set_rest_orientation_at_target(
        Position3D(eye_pos[0], eye_pos[1] - 100.0, eye_pos[2]), up=Position3D(0.0, 0.0, 1.0)
    )
    return eye


def build_camera(scene: dict, eye: Eye) -> Camera:
    """Camera at the scene's position, given the FickRotation projection aimed at the eye.

    The rotation and reference-frame translation are computed from the camera position and the eye; no
    precomputed angles are taken from reference.json, which supplies only the camera's physical placement
    and intrinsic and otherwise serves as the answer key for the pixels.
    """
    camera = Camera(camera_matrix=CameraMatrix(np.array(scene["intrinsic_matrix"], float)))
    camera.position = Position3D(*np.array(scene["camera_position_mm"], float))
    camera.point_at(eye, mode="fick")
    return camera


def apparent_pupil_rim(eye: Eye, camera: Camera) -> np.ndarray:
    """The pupil boundary refracted toward the camera, as lab-frame 3D points on the visible cornea."""
    boundary = np.asarray(eye.get_pupil().boundary_points)[:3, :].T
    refracted, valid = eye.cornea.find_refraction_batch(
        camera.position, boundary, 1.0, eye.cornea.refractive_index, eye.trans, eye.model.n_aqueous_humor
    )
    keep = valid & eye.point_on_visible_cornea_batch(refracted)
    return refracted[keep]


def pupil_centre(rim_px: np.ndarray) -> np.ndarray:
    """Least-squares ellipse-fit centre of the projected pupil rim, in pixels (gkaModelEye reports the same)."""
    pts = [Point2D(x=float(x), y=float(y)) for x, y in rim_px]
    c = calculate_pupil_center_from_boundary(pts, Point2D(x=1.0, y=1.0), center_method="ellipse")
    return np.array([c.x, c.y])


def render_target(eye: Eye, camera: Camera, light: Light, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(pupil centre, glint) in image pixels for one gaze target.

    look_at uses the gkaModelEye model's own default method (optical_axis_target_direction), so the eye is
    posed by gka's eyePose convention and the comparison is the eye optics alone.
    """
    eye.look_at(Position3D(*target))
    rim = apparent_pupil_rim(eye, camera)
    rim_px = camera.project(list(starmap(Position3D, rim))).image_points.T
    glint_3d = np.asarray(eye.find_cr(light, camera), float).ravel()[:3]
    glint_px = camera.project(Position3D(*glint_3d)).image_points[:, 0]
    return pupil_centre(rim_px), glint_px


def test_gkaModelEye_matches_reference() -> None:  # noqa: N802  (named after the gkaModelEye model)
    """Eye(model='gkaModelEye') reproduces the gkaModelEye pupil centre and glint at every target."""
    reference = json.loads(REFERENCE.read_text())
    scene = reference["scene"]
    eye_pos = np.array(scene["eye_position_mm"], float)
    light = Light(position=Position3D(*np.array(scene["light_position_mm"], float)))
    eye = build_eye(eye_pos)
    eye.set_pupil_diameter(scene["stop_radius_mm"] * 2.0)
    camera = build_camera(scene, eye)

    for result in reference["results"]:
        pupil_center, glint = render_target(eye, camera, light, np.array(result["target_mm"], float))
        np.testing.assert_allclose(
            pupil_center,
            result["pupil_center_px"],
            atol=TOLERANCE_PX,
            err_msg=f"pupil centre mismatch at target {result['target_mm']}",
        )
        np.testing.assert_allclose(
            glint,
            result["glint_px"],
            atol=TOLERANCE_PX,
            err_msg=f"glint mismatch at target {result['target_mm']}",
        )
