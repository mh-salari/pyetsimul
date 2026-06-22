"""1:1 validation of the ``et_simul`` named model against the et_simul MATLAB reference.

``../matlab/run_reference.m`` generates ``../reference.json``; this test rebuilds the same scene with
``Eye(model="et_simul")`` and asserts the pupil centre and glint match per gaze target.
"""

import json
from pathlib import Path

import numpy as np

from pyetsimul.core import Camera, Eye, Light
from pyetsimul.types import Position3D

REFERENCE = Path(__file__).resolve().parent.parent / "reference.json"
TOLERANCE_PX = 1e-3


def test_et_simul_matches_matlab_reference() -> None:
    """Eye(model='et_simul') reproduces the et_simul MATLAB pupil centre and glint at every target."""
    reference = json.loads(REFERENCE.read_text())
    scene = reference["scene"]
    eye_position = Position3D(*scene["eye_position_mm"])
    camera_position = Position3D(*scene["camera_position_mm"])
    light_position = Position3D(*scene["light_position_mm"])

    light = Light(position=light_position)
    camera = Camera()
    camera.position = camera_position
    camera.point_at(eye_position)

    for result in reference["results"]:
        target = Position3D(*result["target_mm"])

        eye = Eye(model="et_simul")
        eye.position = eye_position
        eye.look_at(target)

        _, pupil_center = eye.get_pupil_in_camera_image(camera)
        cr = eye.find_cr(light, camera)
        assert pupil_center is not None, f"pupil not visible at target {result['target_mm']}"
        assert cr is not None, f"glint not visible at target {result['target_mm']}"
        glint = camera.project(cr).image_points[:, 0]

        np.testing.assert_allclose(
            [pupil_center.x, pupil_center.y],
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
