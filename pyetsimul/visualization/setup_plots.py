"""3D eye tracking setup visualization module.

Provides comprehensive 3D visualization functions for eye tracking setups.
Handles coordinate transformations and anatomical structure plotting.
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from matplotlib import ticker
from matplotlib.path import Path

if TYPE_CHECKING:
    from matplotlib.axes import Axes

from ..core import Camera, Light
from ..types import Point2D, Point3D, Position3D, ScreenGeometry, TransformationMatrix, Vector3D
from .plot_config import PlotConfig, create_plot_config


def _camera_marker() -> Path:
    """Create a camera-shaped marker: rectangle with a circle (lens) inside."""
    # Rectangle (outer body)
    rect = [
        (Path.MOVETO, (-1.0, -0.7)),
        (Path.LINETO, (1.0, -0.7)),
        (Path.LINETO, (1.0, 0.7)),
        (Path.LINETO, (-1.0, 0.7)),
        (Path.CLOSEPOLY, (-1.0, -0.7)),
    ]

    # Circle (lens) — approximate with cubic Bezier curves
    r = 0.45
    k = r * 0.5522848  # control point offset for circular Bezier
    circle = [
        (Path.MOVETO, (r, 0)),
        (Path.CURVE4, (r, k)),
        (Path.CURVE4, (k, r)),
        (Path.CURVE4, (0, r)),
        (Path.CURVE4, (-k, r)),
        (Path.CURVE4, (-r, k)),
        (Path.CURVE4, (-r, 0)),
        (Path.CURVE4, (-r, -k)),
        (Path.CURVE4, (-k, -r)),
        (Path.CURVE4, (0, -r)),
        (Path.CURVE4, (k, -r)),
        (Path.CURVE4, (r, -k)),
        (Path.CURVE4, (r, 0)),
    ]

    codes, verts = zip(*(rect + circle), strict=True)
    return Path(verts, codes)


CAMERA_MARKER = _camera_marker()


def plot_setup(
    ax1: "Axes",
    eyes_data: list[dict[str, Any]] | dict[str, Any],
    look_at_targets: list[Position3D] | Position3D,
    lights: list[Light] | Light | None = None,
    cameras: list[Camera] | Camera | None = None,
    cr_3d_lists: list[list[Any]] | None = None,
    ref_bounds: dict[str, tuple[float, float]] | None = None,
    calib_points: list[Point2D] | None = None,
    screen: ScreenGeometry | None = None,
) -> tuple[list[str], list[str]]:
    """Plot 3D eye tracking setup visualization.

    Shows eye anatomy, cameras, lights, and corneal reflections in world coordinates.

    Args:
        ax1: 3D matplotlib axis
        eyes_data: Eye data dict or list of eye data dicts with transformed eye anatomy data
        look_at_targets: Target point or list of target points for each eye
        lights: Optional Light object or list of Light objects with positions
        cameras: Optional Camera object or list of Camera objects
        cr_3d_lists: List of lists of corneal reflection 3D positions for each eye
        ref_bounds: Optional reference bounds dict with 'x', 'y', 'z' keys
        calib_points: Optional calibration points array to plot as black x markers
        screen: Optional screen geometry to draw the screen border

    Returns:
        Tuple[List[str], List[str]]: Lists of eye and camera colors used in the plot.

    """
    # Convert single objects to lists
    if not isinstance(eyes_data, list):
        eyes_data = [eyes_data]
    if not isinstance(look_at_targets, list):
        look_at_targets = [look_at_targets]
    if lights is not None and not isinstance(lights, list):
        lights = [lights]
    if cameras is not None and not isinstance(cameras, list):
        cameras = [cameras]
    if cr_3d_lists is not None and not isinstance(cr_3d_lists, list):
        cr_3d_lists = [cr_3d_lists]

    ax1.cla()

    if not eyes_data:
        config = create_plot_config()
        ax1.text(0, 0, 0, "No eyes to display", fontsize=config.fonts.title, ha="center")
        return [], []

    config = create_plot_config()

    # Use centralized color palettes
    eye_colors = config.colors.eyes
    light_eye_colors = config.colors.eyes_light
    camera_colors = config.colors.cameras

    # Plot all eyes
    for eye_idx, (eye_data, target) in enumerate(zip(eyes_data, look_at_targets, strict=False)):
        eye_color = eye_colors[eye_idx % len(eye_colors)]
        light_eye_color = light_eye_colors[eye_idx % len(light_eye_colors)]

        # Plot corneal surface
        ax1.plot_surface(
            eye_data["cornea_surface_x"],
            eye_data["cornea_surface_y"],
            eye_data["cornea_surface_z"],
            alpha=config.lines.secondary_alpha,
            color=light_eye_color,
            edgecolor=eye_color,
            linewidth=config.lines.thin_lines,
        )

        # Mark key anatomical points
        cornea_center = eye_data["cornea_center_world"]
        pupil_center = eye_data["pupil_world"]
        ax1.scatter(
            cornea_center.x,
            cornea_center.y,
            cornea_center.z,
            color=config.colors.cornea_outer,
            s=config.markers.landmarks,
            label=f"Eye {eye_idx + 1} Cornea Center",
        )
        ax1.scatter(
            pupil_center.x,
            pupil_center.y,
            pupil_center.z,
            color=config.colors.pupil,
            s=config.markers.landmarks,
            label=f"Eye {eye_idx + 1} Pupil Center",
        )

        # Draw optical axis
        optical_axis_end = eye_data["optical_axis_end"]
        ax1.plot(
            [cornea_center.x, optical_axis_end.x],
            [cornea_center.y, optical_axis_end.y],
            [cornea_center.z, optical_axis_end.z],
            color=config.colors.optical_axis,
            linestyle=config.lines.dashed,
            linewidth=config.lines.thick_lines,
            label=f"Eye {eye_idx + 1} Optical Axis",
        )

        # Draw visual axis to target
        ax1.plot(
            [pupil_center.x, target.x],
            [pupil_center.y, target.y],
            [pupil_center.z, target.z],
            color=config.colors.visual_axis,
            linestyle=config.lines.dashed,
            linewidth=config.lines.thick_lines,
            alpha=0.5,
            label=f"Eye {eye_idx + 1} Visual Axis",
        )

        # Mark gaze target
        ax1.scatter(
            target.x,
            target.y,
            target.z,
            color=config.colors.target,
            s=config.markers.key_landmarks,
            marker="+",
            label=f"Eye {eye_idx + 1} Target",
        )

    # Add scene elements - multiple lights
    if lights is not None:
        for i, light in enumerate(lights):
            light_pos = light.position
            color = config.colors.lights[i % len(config.colors.lights)]
            ax1.scatter(
                light_pos.x,
                light_pos.y,
                light_pos.z,
                color=color,
                edgecolors="black",
                linewidths=0.5,
                s=config.markers.scene_elements,
                marker="*",
                label=f"Light Source {i + 1}",
            )

    # Add cameras
    if cameras is not None:
        for cam_idx, camera in enumerate(cameras):
            camera_pos = camera.position
            color = camera_colors[cam_idx % len(camera_colors)]
            ax1.scatter(
                camera_pos.x,
                camera_pos.y,
                camera_pos.z,
                facecolors="none",
                edgecolors=color,
                linewidths=1.5,
                s=config.markers.scene_elements,
                marker=CAMERA_MARKER,
                label=f"Camera {cam_idx + 1}",
            )

            # Add line from camera to where it's pointing
            if camera.pointing_at is not None:
                pointing_pos = camera.pointing_at
                ax1.plot(
                    [camera_pos.x, pointing_pos.x],
                    [camera_pos.y, pointing_pos.y],
                    [camera_pos.z, pointing_pos.z],
                    color=color,
                    linestyle=config.lines.dashed,
                    alpha=config.lines.background_alpha,
                    linewidth=config.lines.standard_lines,
                )

    # Add corneal reflections if provided
    if cr_3d_lists is not None:
        for eye_idx, cr_3d_list in enumerate(cr_3d_lists):
            if cr_3d_list is not None:
                _plot_corneal_reflections_for_eye(ax1, config, cr_3d_list, eye_idx, lights)

    # 3D plot formatting
    ax1.set_xlabel("X (mm)")
    ax1.set_ylabel("Y (mm)")
    ax1.set_zlabel("Z (mm)")
    ax1.set_title("Eye Tracking Setup")
    ax1.legend(**config.layout.legend_upper_left, fontsize=config.fonts.annotation)

    # Convert axes to mm for better readability
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x * 1000:.0f}"))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x * 1000:.0f}"))
    ax1.zaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x * 1000:.0f}"))

    # Calculate bounds that include all eyes, cameras, and targets
    all_points = []

    # Add all eye centers and targets
    for eye_data, target in zip(eyes_data, look_at_targets, strict=False):
        all_points.extend([eye_data["cornea_center_world"], target])

    # Add camera positions
    if cameras:
        for camera in cameras:
            all_points.append(camera.position)

    # Include calibration points in bounds calculation if provided
    if calib_points is not None:
        calib_array = np.array([[cp.x, cp.y] for cp in calib_points])
        if calib_array.ndim == 2 and calib_array.shape[1] == 3:
            all_points.extend(calib_array.tolist())

    if ref_bounds:
        ax1.set_xlim(ref_bounds["x"])
        ax1.set_ylim(ref_bounds["y"])
        ax1.set_zlim(ref_bounds["z"])
    elif all_points:
        all_points_array = np.array([[p.x, p.y, p.z] for p in all_points])
        min_coords = np.min(all_points_array, axis=0)
        max_coords = np.max(all_points_array, axis=0)

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

    # Set equal aspect ratio for proper sphere appearance
    ax1.set_box_aspect([1, 1, 1])

    # Plot calibration points if provided
    if calib_points is not None:
        cps = np.array([[cp.x, cp.y] for cp in calib_points]).T
        calib_x = cps[0, :]
        calib_y = np.zeros_like(cps[0, :])
        calib_z = cps[1, :]

        ax1.scatter(
            calib_x,
            calib_y,
            calib_z,
            color=config.colors.calibration_points,
            marker="x",
            s=config.markers.calibration_points,
            linewidth=config.lines.thick_lines,
            label="Calibration Points",
        )

    # Draw screen border if screen geometry provided
    if screen is not None:
        hw = screen.width / 2
        hh = screen.height / 2
        if screen.plane == "xz":
            sx = [-hw, hw, hw, -hw, -hw]
            sy = [0.0, 0.0, 0.0, 0.0, 0.0]
            sz = [-hh, -hh, hh, hh, -hh]
        elif screen.plane == "xy":
            sx = [-hw, hw, hw, -hw, -hw]
            sy = [-hh, -hh, hh, hh, -hh]
            sz = [0.0, 0.0, 0.0, 0.0, 0.0]
        else:  # yz
            sx = [0.0, 0.0, 0.0, 0.0, 0.0]
            sy = [-hw, hw, hw, -hw, -hw]
            sz = [-hh, -hh, hh, hh, -hh]
        ax1.plot(sx, sy, sz, color="black", linewidth=config.lines.standard_lines, linestyle=config.lines.dashed, label="Screen")

    return eye_colors, camera_colors


def transform_surface(
    x_local: Vector3D, y_local: Vector3D, z_local: Vector3D, trans_matrix: TransformationMatrix
) -> tuple[Vector3D, Vector3D, Vector3D]:
    """Transform surface coordinates to world coordinates"""
    ones = np.ones_like(x_local)
    local_coords = np.stack([x_local, y_local, z_local, ones], axis=0)
    world_coords = np.einsum("ij,j...->i...", trans_matrix, local_coords)
    return world_coords[0], world_coords[1], world_coords[2]


def plot_axis(
    ax: "Axes",
    center: Point3D,
    trans_matrix: TransformationMatrix,
    axis_idx: int,
    label: str,
    color: str,
    length: float = 0.003,
) -> None:
    """Plot a single axis with arrow and label"""
    config = create_plot_config()

    axis_local = np.zeros(4)
    axis_local[axis_idx] = 1
    axis_world = trans_matrix @ axis_local
    axis_end = center + length * axis_world[:3]

    ax.quiver(
        center.x,
        center.y,
        center.z,
        axis_world[0] * length,
        axis_world[1] * length,
        axis_world[2] * length,
        color=color,
        arrow_length_ratio=0.2,
        linewidth=config.lines.thick_lines,
        alpha=config.lines.secondary_alpha,
    )
    ax.text(
        axis_end.x,
        axis_end.y,
        axis_end.z,
        label,
        fontsize=config.fonts.legend,
        color=color,
        weight=config.fonts.bold_weight,
    )


def _plot_corneal_reflections_for_eye(
    ax: "Axes", config: PlotConfig, cr_3d_list: list[Position3D], eye_idx: int, lights: list[Light] | Light | None
) -> None:
    """Plot corneal reflections and light rays for a single eye."""
    for cr_idx, cr_3d in enumerate(cr_3d_list):
        if cr_3d is not None:
            color = config.colors.lights[cr_idx % len(config.colors.lights)]
            ax.scatter(
                cr_3d.x,
                cr_3d.y,
                cr_3d.z,
                color=color,
                s=config.markers.corneal_reflections,
                marker="o",
                edgecolor=config.colors.rotation_center,
                linewidth=config.lines.standard_lines,
                label=f"Eye {eye_idx + 1} CR {cr_idx + 1}",
            )

            # Draw light ray paths if lights provided
            if lights and cr_idx < len(lights):
                light = lights[cr_idx]
                light_pos = light.position
                ax.plot(
                    [light_pos.x, cr_3d.x],
                    [light_pos.y, cr_3d.y],
                    [light_pos.z, cr_3d.z],
                    color=color,
                    linestyle=config.lines.dashed,
                    linewidth=config.lines.standard_lines,
                    alpha=config.lines.background_alpha,
                )
