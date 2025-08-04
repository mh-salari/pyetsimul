"""Eye anatomy visualization using structured types and vector arithmetic.

Provides 3D visualization functions for eye anatomy using structured types.
Supports anatomical accuracy and vector-based transformations.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..core import Eye
from ..types import Position3D, Direction3D


def plot_eye_anatomy(eye: Eye = Eye(), target_point: Position3D = Position3D(15e-3, 15e-3, 0), ax=None):
    """Plot 3D eye anatomy using structured types and vector arithmetic.

    Visualizes anatomical structures of the eye in 3D using vector-based transformations.
    Useful for understanding eye geometry and verifying anatomical accuracy.

    Args:
        eye: Eye object to plot
        target_point: Target position for eye to look at
        ax: Matplotlib 3D axis (optional)
    """
    # Make eye look at target
    eye.look_at(target_point)

    # Calculate all key points in WORLD coordinates using structured types
    eye_rotation_center = eye.position
    cornea_center = eye.cornea.center
    cornea_inner_center = eye.cornea.get_posterior_center()
    pupil_position = eye.pupil.pos_pupil
    fovea_position = eye.fovea_position

    # Transform positions to world coordinates using vector arithmetic
    cornea_center_world = Position3D.from_array(eye.trans @ np.array(cornea_center))
    cornea_inner_center_world = Position3D.from_array(eye.trans @ np.array(cornea_inner_center))
    pupil_center_world = Position3D.from_array(eye.trans @ np.array(pupil_position))
    fovea_world = Position3D.from_array(eye.trans @ np.array(fovea_position))

    # Eye sphere parameters
    main_eye_radius = eye.axial_length / 2
    apex_pos = eye.cornea.get_apex_position()
    limbus_z_local = apex_pos.z + eye.cornea.get_corneal_depth()

    # Create corneal surface coordinates using structured types
    cornea_radius = eye.cornea.anterior_radius
    depth = eye.cornea.get_corneal_depth()
    cap_angle = np.arccos((cornea_radius - depth) / cornea_radius)
    phi_cap = np.linspace(0, cap_angle, 20)
    theta_full = np.linspace(0, 2 * np.pi, 50)
    phi_grid, theta_grid = np.meshgrid(phi_cap, theta_full)

    # Outer corneal surface using vector arithmetic
    x_outer_local = cornea_center.x + cornea_radius * np.sin(phi_grid) * np.cos(theta_grid)
    y_outer_local = cornea_center.y + cornea_radius * np.sin(phi_grid) * np.sin(theta_grid)
    z_outer_local = cornea_center.z - cornea_radius * np.cos(phi_grid)

    # Inner corneal surface using vector arithmetic
    cornea_inner_radius = eye.cornea.posterior_radius
    x_inner_local = cornea_inner_center.x + cornea_inner_radius * np.sin(phi_grid) * np.cos(theta_grid)
    y_inner_local = cornea_inner_center.y + cornea_inner_radius * np.sin(phi_grid) * np.sin(theta_grid)
    z_inner_local = cornea_inner_center.z - cornea_inner_radius * np.cos(phi_grid)

    # Transform surfaces to world coordinates
    def transform_surface(x_local, y_local, z_local, trans_matrix):
        """Transform surface coordinates using homogeneous coordinates."""
        local_coords = np.array(
            [x_local.flatten(), y_local.flatten(), z_local.flatten(), np.ones_like(x_local.flatten())]
        )
        world_coords = np.einsum("ij,j...->i...", trans_matrix, local_coords)
        return (
            world_coords[0].reshape(x_local.shape),
            world_coords[1].reshape(y_local.shape),
            world_coords[2].reshape(z_local.shape),
        )

    x_outer_world, y_outer_world, z_outer_world = transform_surface(
        x_outer_local, y_outer_local, z_outer_local, eye.trans
    )
    x_inner_world, y_inner_world, z_inner_world = transform_surface(
        x_inner_local, y_inner_local, z_inner_local, eye.trans
    )

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
    axis_length = 0.02  # 20mm axis length

    # Optical axis using structured types
    optical_axis_vec = Direction3D.from_array(optical_axis_unit[:3])
    optical_axis_end = eye_rotation_center + optical_axis_vec * axis_length

    # Visual axis using structured types and vector arithmetic
    visual_axis_direction = (eye_rotation_center - fovea_world).normalize()
    visual_axis_end = eye_rotation_center + visual_axis_direction * axis_length

    # Create figure if not provided
    if ax is None:
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection="3d")

    # Plot eye components using structured type coordinates
    ax.plot_surface(x_eye_world, y_eye_world, z_eye_world, alpha=0.3, color="lightgray", label="Eye Globe")
    ax.plot_surface(x_outer_world, y_outer_world, z_outer_world, alpha=0.8, color="lightblue", label="Cornea (Outer)")
    ax.plot_surface(x_inner_world, y_inner_world, z_inner_world, alpha=0.8, color="lightcyan", label="Cornea (Inner)")

    # Plot key points using structured types
    ax.scatter(
        eye_rotation_center.x,
        eye_rotation_center.y,
        eye_rotation_center.z,
        color="black",
        s=50,
        marker="o",
        label="Rotation Center",
    )
    ax.scatter(
        cornea_center_world.x,
        cornea_center_world.y,
        cornea_center_world.z,
        color="blue",
        s=30,
        marker="s",
        label="Cornea Center",
    )
    ax.scatter(
        cornea_inner_center_world.x,
        cornea_inner_center_world.y,
        cornea_inner_center_world.z,
        color="cyan",
        s=20,
        marker="^",
        label="Inner Cornea Center",
    )
    ax.scatter(
        pupil_center_world.x,
        pupil_center_world.y,
        pupil_center_world.z,
        color="red",
        s=20,
        marker="o",
        label="Pupil Center",
    )
    ax.scatter(fovea_world.x, fovea_world.y, fovea_world.z, color="orange", s=80, marker="*", label="Fovea")

    # Plot pupil boundary using structured types
    n_pupil_points = 50
    t = np.linspace(0, 2 * np.pi, n_pupil_points)
    cos_t = np.cos(t)
    sin_t = np.sin(t)

    # Use vector arithmetic for pupil boundary
    pupil_boundary_local = (
        np.array(pupil_position).reshape(-1, 1)
        + np.array(eye.pupil.x_pupil).reshape(-1, 1) @ cos_t.reshape(1, -1)
        + np.array(eye.pupil.y_pupil).reshape(-1, 1) @ sin_t.reshape(1, -1)
    )
    pupil_boundary_world = eye.trans @ pupil_boundary_local

    ax.plot(
        pupil_boundary_world[0],
        pupil_boundary_world[1],
        pupil_boundary_world[2],
        "k-",
        linewidth=3,
        label="Pupil Opening",
    )

    # Plot axes using structured types
    ax.plot(
        [eye_rotation_center.x, optical_axis_end.x],
        [eye_rotation_center.y, optical_axis_end.y],
        [eye_rotation_center.z, optical_axis_end.z],
        "b--",
        linewidth=1,
        label="Optical Axis",
    )

    ax.plot(
        [fovea_world.x, visual_axis_end.x],
        [fovea_world.y, visual_axis_end.y],
        [fovea_world.z, visual_axis_end.z],
        "r--",
        linewidth=1,
        label="Visual Axis",
    )

    # Plot target point
    ax.scatter(target_point.x, target_point.y, target_point.z, color="green", s=100, marker="x", label="Target")

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Eye Anatomy")
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return ax
