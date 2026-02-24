"""Eye anatomy visualization using structured types and vector arithmetic.

Provides 3D visualization functions for eye anatomy using structured types.
Supports anatomical accuracy and vector-based transformations.
"""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
import numpy as np

from ..core import Eye
from ..geometry.intersections import intersect_ray_conic, intersect_ray_sphere
from ..types import Direction3D, Position3D
from ..utils.eye_surface_points import generate_corneal_surface_points, get_transformed_corneal_landmarks
from ..utils.eyelid_surface_points import generate_eyelid_opening_edge_local, transform_eyelid_points_to_world
from .plot_config import create_plot_config
from .transforms import transform_surface


def _filter_points_by_eyelid_occlusion(points_world: np.ndarray, eye: Eye) -> np.ndarray:
    """Return only points not occluded by eyelid opening."""
    if eye.eyelid is None or len(points_world) == 0:
        return points_world

    eyelid_trans_inv = np.linalg.inv(eye.eyelid_trans)
    points_world_h = np.column_stack([points_world, np.ones(len(points_world))])
    points_eyelid_local_h = (eyelid_trans_inv @ points_world_h.T).T
    points_eyelid_local = points_eyelid_local_h[:, :3]

    visible_mask = []
    for point in points_eyelid_local:
        is_visible = not eye.eyelid.point_within_eyelid(Position3D(point[0], point[1], point[2]))
        visible_mask.append(is_visible)
    visible_mask = np.array(visible_mask)
    return points_world[visible_mask]


def plot_eye_anatomy(eye: Eye, ax: "Axes | None" = None) -> "Axes":
    """Plot 3D eye anatomy using structured types and vector arithmetic.

    Visualizes anatomical structures of the eye in 3D using vector-based transformations.
    Useful for understanding eye geometry and verifying anatomical accuracy.
    Assumes the eye is already oriented as desired.

    Args:
        eye: Eye object to plot (required) - should already be oriented as desired
        ax: Matplotlib 3D axis (optional, creates new if None)

    Raises:
        ValueError: If eye has no current target point (call eye.look_at() first)

    """
    # Get target point from eye
    target_point = eye.current_target_point
    if target_point is None:
        raise ValueError(
            "Eye has no current target point. Call eye.look_at(target_position) first "
            "to orient the eye and set a target for visualization."
        )

    # Calculate all key points in WORLD coordinates using structured types
    eye_rotation_center = eye.position
    cornea_center = eye.cornea.center
    cornea_inner_center = eye.cornea.get_posterior_center()
    pupil_position = eye.pupil.pos_pupil
    fovea_position = eye.fovea_position

    # Transform positions to world coordinates using vector arithmetic
    cornea_center_world = Position3D.from_array(eye.trans @ np.array(cornea_center))
    cornea_inner_center_world = Position3D.from_array(eye.trans @ np.array(cornea_inner_center))
    fovea_world = Position3D.from_array(eye.trans @ np.array(fovea_position))

    # Eye sphere parameters
    main_eye_radius = eye.axial_length / 2
    apex_pos = eye.cornea.get_apex_position()
    limbus_z_local = apex_pos.z + eye.cornea.get_corneal_depth()

    # Generate corneal surface points using proper transformation handling
    # Choose intersection function based on cornea type
    if eye.cornea.cornea_type == "conic":
        intersection_func = intersect_ray_conic
    elif eye.cornea.cornea_type == "spherical":
        intersection_func = intersect_ray_sphere
    else:
        raise ValueError(f"Unknown cornea type: {eye.cornea.cornea_type}")

    anterior_points = generate_corneal_surface_points(eye, intersection_func, "anterior", n_points=50)
    posterior_points = generate_corneal_surface_points(eye, intersection_func, "posterior", n_points=50)

    # Get transformed corneal landmarks
    corneal_landmarks = get_transformed_corneal_landmarks(eye)
    cornea_center_world = corneal_landmarks["anterior_center"]
    cornea_inner_center_world = corneal_landmarks["posterior_center"]

    # Filter surfaces based on corneal depth limits (same as spherical cornea visualization)
    anterior_mask = np.array([eye.point_within_cornea(Position3D(p[0], p[1], p[2])) for p in anterior_points])
    posterior_mask = np.array([eye.point_within_cornea(Position3D(p[0], p[1], p[2])) for p in posterior_points])
    anterior_limited = anterior_points[anterior_mask]
    posterior_limited = posterior_points[posterior_mask]

    # Apply eyelid occlusion filtering to cornea points
    anterior_limited = _filter_points_by_eyelid_occlusion(anterior_limited, eye)
    posterior_limited = _filter_points_by_eyelid_occlusion(posterior_limited, eye)

    # Create eye sphere coordinates using structured types
    phi_eye = np.linspace(0, np.pi, 30)
    theta_eye = np.linspace(0, 2 * np.pi, 50)
    phi_eye_grid, theta_eye_grid = np.meshgrid(phi_eye, theta_eye)

    x_eye_local = main_eye_radius * np.sin(phi_eye_grid) * np.cos(theta_eye_grid)
    y_eye_local = main_eye_radius * np.sin(phi_eye_grid) * np.sin(theta_eye_grid)
    z_eye_local = main_eye_radius * np.cos(phi_eye_grid)

    x_eye_world, y_eye_world, z_eye_world = transform_surface(x_eye_local, y_eye_local, z_eye_local, eye.trans)

    # Mask out the front part where cornea is using vector arithmetic
    optical_axis_world = eye.trans @ np.array([0, 0, -1, 0])
    optical_axis_unit = optical_axis_world / np.linalg.norm(optical_axis_world)

    limbus_point_world = eye.trans @ np.array([0, 0, limbus_z_local, 1])
    limbus_projection = np.dot(limbus_point_world[:3] - np.array(eye_rotation_center)[:3], optical_axis_unit[:3])

    eye_vectors = np.stack([x_eye_world, y_eye_world, z_eye_world]) - np.array(eye_rotation_center)[:3].reshape(
        3, 1, 1
    )
    projections = np.einsum("i,ijk->jk", optical_axis_unit[:3], eye_vectors)
    mask = projections <= limbus_projection

    x_eye_world[~mask] = np.nan
    y_eye_world[~mask] = np.nan
    z_eye_world[~mask] = np.nan

    # Calculate axes using vector arithmetic
    axis_length = 20  # 20mm axis length

    # Optical axis using structured types
    optical_axis_vec = Direction3D.from_array(optical_axis_unit[:3])
    optical_axis_end = eye_rotation_center + optical_axis_vec * axis_length

    # Visual axis using structured types and vector arithmetic
    visual_axis_direction = (eye_rotation_center - fovea_world).normalize()
    visual_axis_end = eye_rotation_center + visual_axis_direction * axis_length

    config = create_plot_config()

    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=config.layout.anatomy_detail)
        ax = fig.add_subplot(111, projection="3d")

    # Plot eye components using structured type coordinates
    ax.plot_surface(
        x_eye_world,
        y_eye_world,
        z_eye_world,
        alpha=config.lines.grid_alpha,
        color=config.colors.eye_globe,
        label="Eye Globe",
    )

    # Plot corneal surfaces using filtered surface points
    if len(anterior_limited) > 0:
        ax.scatter(
            anterior_limited[:, 0],
            anterior_limited[:, 1],
            anterior_limited[:, 2],
            alpha=config.lines.primary_alpha,
            color=config.colors.cornea_outer,
            s=config.markers.cornea_surface_anterior,
            label="Cornea outer surface",
        )

    if len(posterior_limited) > 0:
        ax.scatter(
            posterior_limited[:, 0],
            posterior_limited[:, 1],
            posterior_limited[:, 2],
            alpha=config.lines.primary_alpha,
            color=config.colors.cornea_inner,
            s=config.markers.cornea_surface_posterior,
            label="Cornea inner surface",
        )

    # Plot key points using structured types
    ax.scatter(
        eye_rotation_center.x,
        eye_rotation_center.y,
        eye_rotation_center.z,
        color=config.colors.rotation_center,
        s=config.markers.small_details,
        marker="o",
        label="Rotation Center",
    )
    ax.scatter(
        cornea_center_world.x,
        cornea_center_world.y,
        cornea_center_world.z,
        color=config.colors.cornea_outer,
        s=config.markers.cornea_center_outer,
        marker="^",
        label="Cornea center (outer)",
    )
    ax.scatter(
        cornea_inner_center_world.x,
        cornea_inner_center_world.y,
        cornea_inner_center_world.z,
        color=config.colors.cornea_inner,
        s=config.markers.cornea_center_inner,
        marker="^",
        label="Cornea center (inner)",
    )
    ax.scatter(
        fovea_world.x,
        fovea_world.y,
        fovea_world.z,
        color=config.colors.fovea,
        s=config.markers.small_details + 30,
        marker="*",
        label="Fovea",
    )

    # Plot pupil as filled dark circle using structured types
    n_pupil_points = 120
    t = np.linspace(0, 2 * np.pi, n_pupil_points)

    # Create filled pupil with radial points from center to boundary
    n_radial = 10
    radial_factors = np.linspace(0, 1, n_radial)

    pupil_points_local = []
    for r_factor in radial_factors:
        for theta in t:
            cos_theta = np.cos(theta)
            sin_theta = np.sin(theta)
            # Point on pupil surface at radius factor r_factor
            point_local = np.array(pupil_position)[:3].reshape(-1, 1) + r_factor * (
                np.array(eye.pupil.x_pupil)[:3].reshape(-1, 1) * cos_theta
                + np.array(eye.pupil.y_pupil)[:3].reshape(-1, 1) * sin_theta
            )
            pupil_points_local.append(point_local.flatten())

    # Transform all pupil points to world coordinates
    pupil_points_local = np.array(pupil_points_local)  # Nx3 format
    # Convert to homogeneous coordinates for transformation
    pupil_points_local_h = np.column_stack([pupil_points_local, np.ones(len(pupil_points_local))])
    pupil_points_world_h = (eye.trans @ pupil_points_local_h.T).T  # Transform and back to Nx4
    pupil_points_world = pupil_points_world_h[:, :3]  # Extract Nx3 world coordinates

    # Apply eyelid occlusion filtering to pupil points
    pupil_points_world_filtered = _filter_points_by_eyelid_occlusion(pupil_points_world, eye)

    # Plot filtered pupil as scatter points
    if len(pupil_points_world_filtered) > 0:
        ax.scatter(
            pupil_points_world_filtered[:, 0],
            pupil_points_world_filtered[:, 1],
            pupil_points_world_filtered[:, 2],
            c=config.colors.pupil,
            s=config.markers.surface_points,
            alpha=config.lines.primary_alpha,
            label="Pupil Opening",
        )

    # Plot axes using structured types
    ax.plot(
        [eye_rotation_center.x, optical_axis_end.x],
        [eye_rotation_center.y, optical_axis_end.y],
        [eye_rotation_center.z, optical_axis_end.z],
        color=config.colors.optical_axis,
        linestyle=config.lines.dashed,
        linewidth=config.lines.standard_lines,
        label="Optical Axis",
    )

    ax.plot(
        [fovea_world.x, visual_axis_end.x],
        [fovea_world.y, visual_axis_end.y],
        [fovea_world.z, visual_axis_end.z],
        color=config.colors.visual_axis,
        linestyle=config.lines.dashed,
        linewidth=config.lines.standard_lines,
        label="Visual Axis",
    )

    # Plot eyelid opening edge if enabled
    if eye.eyelid is not None:
        opening_edge_local = generate_eyelid_opening_edge_local(eye.eyelid, n_edge_points=160)
        if len(opening_edge_local) > 0:
            edge_local_closed = np.vstack([opening_edge_local, opening_edge_local[0]])
            edge_world = transform_eyelid_points_to_world(edge_local_closed, eye.eyelid_trans)
            ax.plot(
                edge_world[:, 0],
                edge_world[:, 1],
                edge_world[:, 2],
                color=config.colors.eyelid,
                linewidth=config.elements.eyelid_width,
                label="Eyelid Opening",
            )

    # Plot target point
    ax.scatter(
        target_point.x,
        target_point.y,
        target_point.z,
        color=config.colors.target,
        s=config.markers.landmarks,
        marker="x",
        label="Target",
    )

    # Set labels and title
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Eye Anatomy")
    # Overlay eye openness percentage (top-left)
    if eye.eyelid is not None:
        openness_pct = 100.0 * float(eye.eyelid.openness)
        ax.text2D(0.02, 0.96, f"Eye openness: {openness_pct:.0f}%", transform=ax.transAxes)
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return ax
