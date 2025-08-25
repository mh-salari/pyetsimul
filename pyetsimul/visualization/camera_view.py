"""Camera view visualization module.

Provides functions for visualizing the camera's view of the eye.
Shows pupil detection, corneal reflections, and camera image coordinates.
"""

from typing import Optional, List

import matplotlib.pyplot as plt
import numpy as np

from pyetsimul.core import Camera
from ..types import Position3D, CameraImage


def plot_camera_view_of_eye(
    camera_images: List[CameraImage],
    cameras: List[Camera],
    cr_3d_lists: Optional[List[List[Optional[Position3D]]]] = None,
    ax=None,
    eye_colors: Optional[List[str]] = None,
    camera_colors: Optional[List[str]] = None,
) -> None:
    """Plot camera views of eyes.

    Shows what each camera sees - all eyes visible in that camera's field of view.

    Args:
        camera_images: List of CameraImage objects (each contains all eyes seen by that camera)
        cameras: List of Camera objects
        cr_3d_lists: List of lists of corneal reflection 3D positions for each eye
        ax: Optional matplotlib axis
        eye_colors: Optional list of colors for the eyes.
        camera_colors: Optional list of colors for the cameras.
    """
    if len(camera_images) != len(cameras):
        raise ValueError("Number of camera images must match number of cameras")

    ax2 = ax
    if ax2 is None:
        _, ax2 = plt.subplots()
    else:
        ax2.cla()

    if eye_colors is None:
        eye_colors = ["blue", "red", "green", "purple", "orange", "brown"]
    if camera_colors is None:
        camera_colors = ["black", "gray", "darkgreen", "darkblue", "purple", "brown"]

    all_resolutions = []
    has_valid_data = False

    for cam_idx, (camera_image, camera) in enumerate(zip(camera_images, cameras)):
        if camera_image is None:
            continue

        cam_color = camera_colors[cam_idx % len(camera_colors)]
        all_resolutions.append(camera.camera_matrix.resolution)

        if camera_image.pupil_boundaries:
            for eye_idx, boundary in enumerate(camera_image.pupil_boundaries):
                if boundary is not None and len(boundary) > 2:
                    has_valid_data = True
                    eye_color = eye_colors[eye_idx % len(eye_colors)]
                    pupil_x = [p.x for p in boundary] + [boundary[0].x]
                    pupil_y = [p.y for p in boundary] + [boundary[0].y]

                    center_x = np.mean([p.x for p in boundary])
                    center_y = np.mean([p.y for p in boundary])

                    scale_factor = 1.05
                    border_x = [center_x + (p.x - center_x) * scale_factor for p in boundary] + [
                        center_x + (boundary[0].x - center_x) * scale_factor
                    ]
                    border_y = [center_y + (p.y - center_y) * scale_factor for p in boundary] + [
                        center_y + (boundary[0].y - center_y) * scale_factor
                    ]

                    ax2.plot(
                        border_x,
                        border_y,
                        color=cam_color,
                        linewidth=2,
                        alpha=0.8,
                        linestyle="-",
                    )
                    ax2.plot(
                        pupil_x,
                        pupil_y,
                        color=eye_color,
                        linewidth=1,
                        alpha=0.9,
                        label=f"Camera {cam_idx + 1} - Pupil {eye_idx + 1}",
                        linestyle="-",
                    )

        if camera_image.pupil_centers:
            for eye_idx, pupil_center in enumerate(camera_image.pupil_centers):
                if pupil_center is not None:
                    has_valid_data = True
                    eye_color = eye_colors[eye_idx % len(eye_colors)]
                    pupil_center_img = pupil_center.to_array()

                    ax2.scatter(
                        pupil_center_img[0],
                        pupil_center_img[1],
                        color=eye_color,
                        s=25,
                        marker="*",
                        linewidth=2,
                        label=f"Camera {cam_idx + 1} - Eye {eye_idx + 1} Center",
                        edgecolors=cam_color,
                    )

        if cr_3d_lists:
            for eye_idx, cr_3d_list in enumerate(cr_3d_lists):
                for cr_idx, cr_3d in enumerate(cr_3d_list):
                    if cr_3d is not None:
                        has_valid_data = True
                        projection_result = camera.project(cr_3d)
                        cr_img = projection_result.image_points

                        ax2.scatter(
                            cr_img[0, 0],
                            cr_img[1, 0],
                            color="gold",
                            s=40,
                            marker="o",
                            edgecolor=cam_color,
                            linewidth=1.5,
                            label=f"Camera {cam_idx + 1} - Eye {eye_idx + 1} CR {cr_idx + 1}",
                        )

    if not has_valid_data:
        ax2.text(0, 0, "No camera data to display", ha="center", va="center", fontsize=12)
        return

    if all_resolutions:
        max_res_x = max(res.x for res in all_resolutions)
        max_res_y = max(res.y for res in all_resolutions)

        ax2.set_xlim(-max_res_x / 2, max_res_x / 2)
        ax2.set_ylim(-max_res_y / 2, max_res_y / 2)
        ax2.invert_yaxis()
        ax2.invert_xaxis()

    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_title("Camera View")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")

    handles, _ = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend(fontsize=8, bbox_to_anchor=(1.05, 1), loc="upper left")
