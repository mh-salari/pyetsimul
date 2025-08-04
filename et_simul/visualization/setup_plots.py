"""3D eye tracking setup visualization module.

Provides comprehensive 3D visualization functions for eye tracking setups.
Handles coordinate transformations and anatomical structure plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any, Tuple

from et_simul.core import Camera, Light
from ..types import Position3D, Point3D, Vector3D, TransformationMatrix


def transform_surface(
    x_local: Vector3D, y_local: Vector3D, z_local: Vector3D, trans_matrix: TransformationMatrix
) -> Tuple[Vector3D, Vector3D, Vector3D]:
    """Transform surface coordinates to world coordinates"""
    ones = np.ones_like(x_local)
    local_coords = np.stack([x_local, y_local, z_local, ones], axis=0)
    world_coords = np.einsum("ij,j...->i...", trans_matrix, local_coords)
    return world_coords[0], world_coords[1], world_coords[2]


def plot_axis(
    ax,
    center: Point3D,
    trans_matrix: TransformationMatrix,
    axis_idx: int,
    label: str,
    color: str,
    length: float = 0.003,
) -> None:
    """Plot a single axis with arrow and label"""
    axis_local = np.zeros(4)
    axis_local[axis_idx] = 1
    axis_world = trans_matrix @ axis_local
    axis_end = center + length * axis_world[:3]

    ax.quiver(
        center[0],
        center[1],
        center[2],
        axis_world[0] * length,
        axis_world[1] * length,
        axis_world[2] * length,
        color=color,
        arrow_length_ratio=0.2,
        linewidth=2,
        alpha=0.8,
    )
    ax.text(
        axis_end[0],
        axis_end[1],
        axis_end[2],
        label,
        fontsize=10,
        color=color,
        weight="bold",
    )


def plot_setup(
    ax1,
    eye_data: Dict[str, Any],
    look_at_target: Position3D,
    lights: Optional[List[Light]],
    camera: Camera,
    cr_3d_list: Optional[List[Optional[Position3D]]] = None,
    ref_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    calib_points: Optional[Position3D] = None,
) -> None:
    """Plot the 3D eye tracking setup visualization.

    Creates comprehensive 3D visualization of eye tracking system components.
    Shows eye anatomy, camera, lights, and corneal reflections in world coordinates.

    Args:
        ax1: 3D matplotlib axis
        eye_data: Dict with transformed eye anatomy data
        look_at_target: Target point [x, y, z, 1] or [x, y, z]
        lights: List of Light objects with positions
        camera: Camera object with transformation and parameters
        cr_3d_list: List of corneal reflection 3D positions
        ref_bounds: Optional reference bounds dict with 'x', 'y', 'z' keys
        calib_points: Optional calibration points array to plot as black x markers
    """
    ax1.cla()

    # Plot corneal surface
    ax1.plot_surface(
        eye_data["X_cornea"],
        eye_data["Y_cornea"],
        eye_data["Z_cornea"],
        alpha=0.8,
        color="lightblue",
        edgecolor="blue",
        linewidth=0.1,
    )

    # Mark key anatomical points
    cornea_center = eye_data["cornea_center_world"]
    pupil_center = eye_data["pupil_world"]
    ax1.scatter(
        cornea_center.x,
        cornea_center.y,
        cornea_center.z,
        color="green",
        s=100,
        label="Cornea Center",
    )
    ax1.scatter(
        pupil_center.x,
        pupil_center.y,
        pupil_center.z,
        color="cornflowerblue",
        s=100,
        label="Pupil Center",
    )

    # Draw optical axis
    optical_axis_end = eye_data["optical_axis_end"]
    ax1.plot(
        [cornea_center.x, optical_axis_end.x],
        [cornea_center.y, optical_axis_end.y],
        [cornea_center.z, optical_axis_end.z],
        "g--",
        linewidth=3,
        label="Optical Axis",
    )

    # Draw visual axis to target
    target_world = look_at_target

    ax1.plot(
        [pupil_center.x, target_world.x],
        [pupil_center.y, target_world.y],
        [pupil_center.z, target_world.z],
        "r--",
        linewidth=3,
        label="Visual Axis",
    )

    # Add scene elements - multiple lights
    if lights is not None:
        light_colors = ["yellow", "orange", "gold", "khaki"]  # Colors for different lights
        for i, light in enumerate(lights):
            light_pos = light.position
            color = light_colors[i % len(light_colors)]
            ax1.scatter(
                light_pos.x, light_pos.y, light_pos.z, color=color, s=200, marker="*", label=f"Light Source {i + 1}"
            )

    camera_pos = camera.position
    ax1.scatter(camera_pos.x, camera_pos.y, camera_pos.z, color="black", s=200, marker="s", label="Camera")

    # Add line from camera to where it's pointing
    if camera.pointing_at is not None:
        pointing_pos = camera.pointing_at
        ax1.plot(
            [camera_pos.x, pointing_pos.x],
            [camera_pos.y, pointing_pos.y],
            [camera_pos.z, pointing_pos.z],
            color="black",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

    ax1.scatter(
        target_world.x, target_world.y, target_world.z, color="magenta", s=150, marker="D", label="Gaze Target"
    )

    # Add corneal reflections if provided
    if cr_3d_list is not None:
        cr_colors = [
            "#FFE171",
            "#F9F871",
            "#FFD67C",
            "#C9AF41",
        ]  # Custom colors for different CRs
        for i, cr_3d in enumerate(cr_3d_list):
            if cr_3d is not None:
                color = cr_colors[i % len(cr_colors)]
                ax1.scatter(
                    cr_3d.x,
                    cr_3d.y,
                    cr_3d.z,
                    color=color,
                    s=80,
                    marker="o",
                    edgecolor="black",
                    linewidth=1,
                    label=f"CR {i + 1}",
                )

                # Get corresponding light position
                light = lights[i]
                light_pos = light.position

                # Draw light ray paths (only from light to CR, not to camera)
                ax1.plot(
                    [light_pos.x, cr_3d.x],
                    [light_pos.y, cr_3d.y],
                    [light_pos.z, cr_3d.z],
                    color=color,
                    linestyle="-",
                    linewidth=2,
                    alpha=0.7,
                )

    # 3D plot formatting
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("3D Eye Tracking Setup")
    ax1.legend(loc="upper left")

    # Convert axes to mm for better readability
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1000:.0f}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1000:.0f}"))
    ax1.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x * 1000:.0f}"))

    # Apply reference bounds if provided, otherwise calculate bounds to include all objects
    if ref_bounds:
        ax1.set_xlim(ref_bounds["x"])
        ax1.set_ylim(ref_bounds["y"])
        ax1.set_zlim(ref_bounds["z"])
    else:
        # Calculate bounds that include eye, camera, and target with equal aspect ratio
        eye_center = eye_data["cornea_center_world"]
        camera_pos = camera.trans[:, 3]
        target_world = look_at_target

        # Find min/max of all objects
        all_points = [eye_center, camera_pos, target_world]

        # Include calibration points in bounds calculation if provided
        if calib_points is not None:
            calib_array = np.array(calib_points)
            if calib_array.ndim == 2 and calib_array.shape[1] == 3:
                all_points.extend(calib_array.tolist())

        all_points = np.array(all_points)
        min_coords = np.min(all_points[:, :3], axis=0)
        max_coords = np.max(all_points[:, :3], axis=0)

        # Add padding and make all axes have same range
        padding = 0.02  # 20mm padding
        ranges = max_coords - min_coords + 2 * padding
        max_range = np.max(ranges)

        # Center each axis and use the same range for all
        centers = (max_coords + min_coords) / 2
        half_range = max_range / 2

        ax1.set_xlim(centers[0] - half_range, centers[0] + half_range)
        ax1.set_ylim(centers[1] - half_range, centers[1] + half_range)
        ax1.set_zlim(centers[2] - half_range, centers[2] + half_range)

    # Set equal aspect ratio for proper sphere appearance (must be after setting limits)
    ax1.set_box_aspect([1, 1, 1])

    # Plot calibration points if provided
    if calib_points is not None:
        calib_points = np.array(calib_points).T
        # Assume 2D points are in the X-Z plane (y=0)
        calib_x = calib_points[0, :]
        calib_y = np.zeros_like(calib_points[0, :])
        calib_z = calib_points[1, :]

        ax1.scatter(
            calib_x,
            calib_y,
            calib_z,
            color="black",
            marker="x",
            s=50,
            linewidth=2,
            label="Calibration Points",
        )
