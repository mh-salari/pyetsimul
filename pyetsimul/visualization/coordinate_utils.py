"""Data preparation utilities for visualization.

Provides coordinate transformations and data preparation for eye tracking visualization.
Handles eye anatomy, camera imaging, and corneal reflection calculations.
Support for multiple eyes, cameras, and lights.
"""

import copy
from typing import Any

import numpy as np

from pyetsimul.core import Camera, ConicCornea, Eye, Light

from ..types import Position3D

CORNEA_SURFACE_STEPS = 20  # grid points of the drawn corneal cap, radially and around the axis


def prepare_eye_data_for_plots(
    eyes: list[Eye] | Eye,
    look_at_targets: list[Position3D] | Position3D,
    lights: list[Light] | Light | None = None,
    cameras: list[Camera] | Camera | None = None,
    use_refraction: bool = True,
    center_method: str = "ellipse",
) -> dict[str, Any]:
    """Prepare eye visualization data for plotting.

    Transforms eye anatomies to world coordinates and generates camera images.
    Calculates corneal reflections and optical axes for 3D visualization. The eyes are drawn looking at
    their targets; this is done on copies, so the eyes passed in keep their pose.

    Args:
        eyes: Eye object or list of Eye objects
        look_at_targets: Target point or list of target points, one per eye
        lights: Optional Light object or list of Light objects with positions
        cameras: Optional Camera object or list of Camera objects
        use_refraction: Whether the camera images use corneal refraction, as for ``Camera.take_image``; give
            a tracker's setting to see what the tracker sees
        center_method: Pupil center method of the camera images, as for ``Camera.take_image``

    Returns:
        dict: Contains eyes_data list, camera_images list, and cr_3d_lists for plotting. There is one camera
            image per camera: the image of the first eye, with the pupils and glints of every eye attached
            as ``pupil_boundaries``, ``pupil_centers`` and ``corneal_reflections_per_eye``. ``cr_3d_lists``
            holds, per eye, the points on the cornea that reflect each light toward the first camera, for the
            3D view.

    """
    # Convert single objects to lists
    if not isinstance(eyes, list):
        eyes = [eyes]
    if not isinstance(look_at_targets, list):
        look_at_targets = [look_at_targets]
    if lights is not None and not isinstance(lights, list):
        lights = [lights]
    if cameras is not None and not isinstance(cameras, list):
        cameras = [cameras]

    if len(eyes) != len(look_at_targets):
        raise ValueError("Number of eyes must match number of look_at_targets")

    if not eyes:
        raise ValueError("At least one eye must be provided")

    # Copies are turned to the targets, so that plotting does not change the pose of the caller's eyes.
    eyes = [copy.deepcopy(eye) for eye in eyes]

    eyes_data = []
    camera_images = []
    cr_3d_lists = []

    for eye, target in zip(eyes, look_at_targets, strict=False):
        eye_data = _prepare_single_eye_data(eye, target)
        eyes_data.append(eye_data)

        # Find corneal reflections for this eye (only if cameras available)
        cr_3d_list = []
        if lights is not None and cameras:
            for light in lights:
                cr_result = eye.find_cr(light, cameras[0])
                cr_3d_list.append(cr_result)
        cr_3d_lists.append(cr_3d_list)

    if cameras:
        for camera in cameras:
            eye_images = [
                camera.take_image(eye, lights, use_refraction=use_refraction, center_method=center_method)
                for eye in eyes
            ]
            first_eye_image = eye_images[0]
            first_eye_image.pupil_boundaries = [image.pupil_boundary for image in eye_images]
            first_eye_image.pupil_centers = [image.pupil_center for image in eye_images]
            first_eye_image.corneal_reflections_per_eye = [image.corneal_reflections for image in eye_images]
            camera_images.append(first_eye_image)
    else:
        camera_images = [None]

    return {"eyes_data": eyes_data, "camera_images": camera_images, "cr_3d_lists": cr_3d_lists}


def _anterior_cap_world(eye: Eye) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid of points on the anterior corneal surface, from the apex to the corneal depth, in world coordinates.

    The surface is the conic rho^2 + (1 + k) z^2 - 2 R z = 0 about the eye's axis, with z measured from the apex
    into the eye and rho the distance from the axis; k = 0 gives the sphere. A toric cornea is drawn as the
    conic it is built on, and a tilt of the cornea is not drawn; both are too small to show at this scale.
    """
    radius = eye.cornea.anterior_radius
    k = eye.cornea.anterior_k if isinstance(eye.cornea, ConicCornea) else 0.0
    depth = eye.cornea.get_corneal_depth()
    apex = eye.cornea.get_apex_position()

    rho = np.linspace(0.0, np.sqrt(2.0 * radius * depth - (1.0 + k) * depth**2), CORNEA_SURFACE_STEPS)
    phi = np.linspace(0.0, 2.0 * np.pi, CORNEA_SURFACE_STEPS)
    z = rho**2 / (radius + np.sqrt(radius**2 - (1.0 + k) * rho**2))
    local = np.stack([
        apex.x + np.outer(np.cos(phi), rho),
        apex.y + np.outer(np.sin(phi), rho),
        apex.z + np.outer(np.ones_like(phi), z),
        np.ones((CORNEA_SURFACE_STEPS, CORNEA_SURFACE_STEPS)),
    ])
    world = np.einsum("ij,jkl->ikl", np.asarray(eye.trans, dtype=float), local)
    return world[0], world[1], world[2]


def _prepare_single_eye_data(eye: Eye, look_at_target: Position3D) -> dict[str, Any]:
    """Helper function to prepare single eye data for visualization."""

    # Calculate all values once
    def transform_point(point: Position3D) -> Position3D:
        p = eye.trans @ point
        return Position3D.from_array(p) if not isinstance(p, Position3D) else p

    # Rotate the eye toward the target
    eye.look_at(look_at_target)
    # Get eye anatomy points
    cornea_center = eye.cornea.center
    pupil_center = eye.pupil.pos_pupil

    # Transform anatomical points to world coordinates
    cornea_center_world = transform_point(cornea_center)
    pupil_world = transform_point(pupil_center)
    # The visual axis runs from the fovea through the eyeball centre.
    fovea_world = transform_point(eye.fovea_position)
    eyeball_center_world = transform_point(Position3D(0.0, 0.0, eye.eyeball_center_z))

    cornea_surface_x, cornea_surface_y, cornea_surface_z = _anterior_cap_world(eye)

    # Calculate optical axis
    optical_axis_direction_local = np.array([0, 0, -1, 0])  # negative z in homogeneous coordinates
    optical_axis_direction_world = transform_point(optical_axis_direction_local)
    optical_axis_end = Position3D(
        cornea_center_world.x + optical_axis_direction_world.x * 20,
        cornea_center_world.y + optical_axis_direction_world.y * 20,
        cornea_center_world.z + optical_axis_direction_world.z * 20,
    )

    return {
        "cornea_surface_x": cornea_surface_x,
        "cornea_surface_y": cornea_surface_y,
        "cornea_surface_z": cornea_surface_z,
        "cornea_center_world": cornea_center_world,
        "pupil_world": pupil_world,
        "fovea_world": fovea_world,
        "eyeball_center_world": eyeball_center_world,
        "optical_axis_end": optical_axis_end,
    }
