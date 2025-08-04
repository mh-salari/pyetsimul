"""Camera view visualization module.

Provides functions for visualizing the camera's view of the eye.
Shows pupil detection, corneal reflections, and camera image coordinates.
"""

import numpy as np
from typing import Optional, List

from et_simul.core import Camera
from ..types import Position3D, CameraImage


def plot_camera_view_of_eye(
    ax2,
    camera_image: CameraImage,
    camera: Camera,
    cr_3d_list: Optional[List[Optional[Position3D]]] = None,
) -> None:
    """Plot the camera view of the eye.

    Visualizes the eye as seen by the camera, including pupil boundary, center, and corneal reflections.
    Useful for debugging and evaluating image-based gaze tracking algorithms.

    Args:
        ax2: 2D matplotlib axis
        camera_image: Dict with camera image data
        camera: Camera object with transformation and parameters
        cr_3d_list: List of corneal reflection 3D positions
    """
    ax2.cla()

    # Check if we have valid data to plot
    pupil_valid = camera_image.pupil_boundary is not None and camera_image.pupil_boundary.shape[1] > 2
    pc_valid = camera_image.pupil_center is not None
    cr_valid = camera_image.corneal_reflections and any(cr is not None for cr in camera_image.corneal_reflections)

    if not (pupil_valid or pc_valid or cr_valid):
        print("No eye elements to plot - skipping camera view")
        return

    # Draw pupil in camera image
    if camera_image.pupil_boundary is not None and camera_image.pupil_boundary.shape[1] > 2:
        pupil_points_img = camera_image.pupil_boundary
        closed_pupil_points = np.hstack((pupil_points_img, pupil_points_img[:, 0:1]))
        ax2.plot(
            closed_pupil_points[0, :],
            closed_pupil_points[1, :],
            color="cornflowerblue",
            linewidth=3,
            label="Pupil",
        )

    # Draw pupil center in camera image
    if camera_image.pupil_center is not None:
        pupil_center_img = camera_image.pupil_center.to_array()
        ax2.scatter(
            pupil_center_img[0],
            pupil_center_img[1],
            color="cornflowerblue",
            s=100,
            marker="+",
            linewidth=3,
            label="Pupil Center",
        )

    # Draw corneal reflections in camera image
    if camera_image.corneal_reflections:
        cr_colors = [
            "#FFE171",
            "#F9F871",
            "#FFD67C",
            "#C9AF41",
        ]  # Same colors as 3D view
        for i, cr_3d in enumerate(cr_3d_list or []):
            if (
                cr_3d is not None
                and i < len(camera_image.corneal_reflections)
                and camera_image.corneal_reflections[i] is not None
            ):
                projection_result = camera.project(cr_3d)
                cr_img = projection_result.image_points

                color = cr_colors[i % len(cr_colors)]
                ax2.scatter(
                    cr_img[0, 0],
                    cr_img[1, 0],
                    color=color,
                    s=80,
                    marker="o",
                    edgecolor="black",
                    linewidth=1,
                    label=f"CR {i + 1}",
                )

    # Set camera image limits
    resolution = camera.camera_matrix.resolution

    ax2.set_xlim(-resolution.x / 2, resolution.x / 2)
    ax2.set_ylim(-resolution.y / 2, resolution.y / 2)

    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_title("Camera View of Eye")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")
    # Only show legend if there are labeled elements
    handles, labels = ax2.get_legend_handles_labels()
    if handles:
        ax2.legend()

    # Add measurement annotations for multiple CRs
    if camera_image.pupil_center is not None and camera_image.corneal_reflections:
        cr_colors = ["#FFE171", "#F9F871", "#FFD67C", "#C9AF41"]
        for i, cr_3d in enumerate(cr_3d_list or []):
            if (
                cr_3d is not None
                and i < len(camera_image.corneal_reflections)
                and camera_image.corneal_reflections[i] is not None
            ):
                projection_result = camera.project(cr_3d)
                cr_img = projection_result.image_points
                pupil_center_img = camera_image.pupil_center.to_array()

                pupil_cr_vector = cr_img.flatten() - pupil_center_img
                pupil_cr_distance_pixels = np.linalg.norm(pupil_cr_vector)

                color = cr_colors[i % len(cr_colors)]
                ax2.plot(
                    [pupil_center_img[0], cr_img[0, 0]],
                    [pupil_center_img[1], cr_img[1, 0]],
                    color=color,
                    alpha=0.7,
                    linewidth=2,
                )

                mid_point = [
                    (pupil_center_img[0] + cr_img[0, 0]) / 2,
                    (pupil_center_img[1] + cr_img[1, 0]) / 2,
                ]
                ax2.annotate(
                    f"{pupil_cr_distance_pixels:.1f} px",
                    xy=mid_point,
                    xytext=(10, 10 + i * 15),
                    textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                    fontsize=9,
                )
