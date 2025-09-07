"""Data preparation utilities for visualization.

Provides coordinate transformations and data preparation for eye tracking visualization.
Handles eye anatomy, camera imaging, and corneal reflection calculations.
Support for multiple eyes, cameras, and lights.
"""

import numpy as np
from typing import Any

from pyetsimul.core import Eye
from ..types import Position3D


def prepare_eye_data_for_plots(
    eyes,
    look_at_targets,
    lights=None,
    cameras=None,
    use_legacy_lookat=False
) -> dict[str, Any]:
    """Prepare eye visualization data for plotting.

    Transforms eye anatomies to world coordinates and generates camera images.
    Calculates corneal reflections and optical axes for 3D visualization.

    Args:
        eyes: Eye object or list of Eye objects
        look_at_targets: Target point or list of target points, one per eye
        lights: Optional Light object or list of Light objects with positions
        cameras: Optional Camera object or list of Camera objects

    Returns:
        dict: Contains eyes_data list, camera_images list, and cr_3d_lists for plotting
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

    eyes_data = []
    camera_images = []
    cr_3d_lists = []

    for i, (eye, target) in enumerate(zip(eyes, look_at_targets)):
        eye_data = _prepare_single_eye_data(eye, target, use_legacy_lookat)
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
            combined_pupil_boundaries = []
            combined_pupil_centers = []

            for eye in eyes:
                eye_image = camera.take_image(eye, lights)
                combined_pupil_boundaries.append(eye_image.pupil_boundary)
                combined_pupil_centers.append(eye_image.pupil_center)

            if eyes:
                first_eye_image = camera.take_image(eyes[0], lights)
                first_eye_image.pupil_boundaries = combined_pupil_boundaries
                first_eye_image.pupil_centers = combined_pupil_centers
                camera_images.append(first_eye_image)
            else:
                camera_images.append(None)
    else:
        camera_images = [None]

    return {"eyes_data": eyes_data, "camera_images": camera_images, "cr_3d_lists": cr_3d_lists}


def _prepare_single_eye_data(eye: Eye, look_at_target: Position3D, use_legacy_lookat) -> dict[str, Any]:
    """Helper function to prepare single eye data for visualization."""

    # Calculate all values once
    def transform_point(point) -> Position3D:
        p = eye.trans @ point
        return Position3D.from_array(p) if not isinstance(p, Position3D) else p

    # Rotate the eye toward the target
    eye.look_at(look_at_target, legacy=use_legacy_lookat)
    # Get eye anatomy points
    cornea_center = eye.cornea.center
    pupil_center = eye.pupil.pos_pupil
    r_cornea = eye.cornea.anterior_radius
    depth_cornea = eye.cornea.get_corneal_depth()

    # Transform anatomical points to world coordinates
    cornea_center_world = transform_point(cornea_center)
    pupil_world = transform_point(pupil_center)

    # Draw corneal surface
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    X_cornea = r_cornea * np.outer(np.cos(u), np.sin(v))
    Y_cornea = r_cornea * np.outer(np.sin(u), np.sin(v))
    Z_cornea = r_cornea * np.outer(np.ones(np.size(u)), np.cos(v))

    # Only show anterior surface (cap)
    mask = Z_cornea > -r_cornea + depth_cornea
    X_cornea[mask] = np.nan
    Y_cornea[mask] = np.nan
    Z_cornea[mask] = np.nan

    # Transform cornea surface points to world coordinates
    for i in range(X_cornea.shape[0]):
        for j in range(X_cornea.shape[1]):
            if not np.isnan(X_cornea[i, j]):
                point = np.array([cornea_center.x, cornea_center.y, cornea_center.z]) + np.array(
                    [X_cornea[i, j], Y_cornea[i, j], Z_cornea[i, j]]
                )
                point_world = transform_point(Position3D.from_array(point))
                X_cornea[i, j] = point_world.x
                Y_cornea[i, j] = point_world.y
                Z_cornea[i, j] = point_world.z

    # Calculate optical axis
    optical_axis_direction_local = np.array([0, 0, -1, 0])  # negative z in homogeneous coordinates
    optical_axis_direction_world = transform_point(optical_axis_direction_local)
    optical_axis_end = Position3D(
        cornea_center_world.x + optical_axis_direction_world.x * 0.02,
        cornea_center_world.y + optical_axis_direction_world.y * 0.02,
        cornea_center_world.z + optical_axis_direction_world.z * 0.02,
    )

    return {
        "X_cornea": X_cornea,
        "Y_cornea": Y_cornea,
        "Z_cornea": Z_cornea,
        "cornea_center_world": cornea_center_world,
        "pupil_world": pupil_world,
        "optical_axis_end": optical_axis_end,
    }
