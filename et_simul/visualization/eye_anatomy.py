"""Eye anatomy visualization using structured types and vector arithmetic.

Provides 3D visualization functions for eye anatomy using structured types.
Supports anatomical accuracy and vector-based transformations.
"""

import numpy as np
import matplotlib.pyplot as plt

from ..core import Eye
from ..types import Position3D, Direction3D
from ..utils.eye_surface_points import generate_corneal_surface_points, get_transformed_corneal_landmarks
from ..geometry.intersections import intersect_ray_sphere, intersect_ray_conic
from .transforms import transform_surface


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

    # Generate corneal surface points using proper transformation handling
    # Choose intersection function based on cornea type
    if eye.cornea.cornea_type == "conic":
        intersection_func = intersect_ray_conic
    elif eye.cornea.cornea_type == "spherical":
        intersection_func = intersect_ray_sphere
    else:
        raise ValueError(f"Unknown cornea type: {eye.cornea.cornea_type}")
        
    anterior_points = generate_corneal_surface_points(eye, intersection_func, "anterior", n_points=30)
    posterior_points = generate_corneal_surface_points(eye, intersection_func, "posterior", n_points=30)

    # Get transformed corneal landmarks
    corneal_landmarks = get_transformed_corneal_landmarks(eye)
    cornea_center_world = corneal_landmarks["anterior_center"]
    cornea_inner_center_world = corneal_landmarks["posterior_center"]

    # Filter surfaces based on corneal depth limits (same as spherical cornea visualization)
    anterior_mask = np.array([eye.point_within_cornea(Position3D(p[0], p[1], p[2])) for p in anterior_points])
    posterior_mask = np.array([eye.point_within_cornea(Position3D(p[0], p[1], p[2])) for p in posterior_points])
    anterior_limited = anterior_points[anterior_mask]
    posterior_limited = posterior_points[posterior_mask]

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

    # Plot corneal surfaces using filtered surface points
    if len(anterior_limited) > 0:
        ax.scatter(
            anterior_limited[:, 0],
            anterior_limited[:, 1],
            anterior_limited[:, 2],
            alpha=0.9,
            color="steelblue",
            s=2,
            label="Cornea (Outer)",
        )

    if len(posterior_limited) > 0:
        ax.scatter(
            posterior_limited[:, 0],
            posterior_limited[:, 1],
            posterior_limited[:, 2],
            alpha=0.9,
            color="darkturquoise",
            s=1.5,
            label="Cornea (Inner)",
        )

    # Plot key points using structured types
    ax.scatter(
        eye_rotation_center.x,
        eye_rotation_center.y,
        eye_rotation_center.z,
        color="navy",
        s=50,
        marker="o",
        label="Rotation Center",
    )
    ax.scatter(
        cornea_center_world.x,
        cornea_center_world.y,
        cornea_center_world.z,
        color="steelblue",
        s=30,
        marker="^",
        label="Cornea Center",
    )
    ax.scatter(
        cornea_inner_center_world.x,
        cornea_inner_center_world.y,
        cornea_inner_center_world.z,
        color="darkturquoise",
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

    # Plot pupil as filled dark circle using structured types
    n_pupil_points = 50
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
            point_local = (
                np.array(pupil_position).reshape(-1, 1) +
                r_factor * (np.array(eye.pupil.x_pupil).reshape(-1, 1) * cos_theta +
                           np.array(eye.pupil.y_pupil).reshape(-1, 1) * sin_theta)
            )
            pupil_points_local.append(point_local.flatten())
    
    # Transform all pupil points to world coordinates
    pupil_points_local = np.array(pupil_points_local).T
    pupil_points_world = eye.trans @ pupil_points_local
    
    # Plot filled pupil as scatter points
    ax.scatter(
        pupil_points_world[0],
        pupil_points_world[1], 
        pupil_points_world[2],
        c="black",
        s=3,
        alpha=0.9,
        label="Pupil Opening"
    )

    # Plot axes using structured types
    ax.plot(
        [eye_rotation_center.x, optical_axis_end.x],
        [eye_rotation_center.y, optical_axis_end.y],
        [eye_rotation_center.z, optical_axis_end.z],
        "g--",
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
    ax.scatter(target_point.x, target_point.y, target_point.z, color="hotpink", s=100, marker="x", label="Target")

    # Set labels and title
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Eye Anatomy")
    ax.legend()

    # Set equal aspect ratio
    ax.set_box_aspect([1, 1, 1])

    return ax
