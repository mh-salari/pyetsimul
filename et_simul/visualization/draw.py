"""Visualization functions for ET Simul.

This module provides 3D visualization and plotting functions for eye tracking
simulation components including eyes, cameras, lights, and complete scenes.
"""

import numpy as np
import matplotlib.pyplot as plt


from et_simul.core import Eye


def transform_surface(x_local, y_local, z_local, trans_matrix):
    """Transform surface coordinates to world coordinates"""
    ones = np.ones_like(x_local)
    local_coords = np.stack([x_local, y_local, z_local, ones], axis=0)
    world_coords = np.einsum("ij,j...->i...", trans_matrix, local_coords)
    return world_coords[0], world_coords[1], world_coords[2]


def plot_axis(ax, center, trans_matrix, axis_idx, label, color, length=0.003):
    """Plot a single axis with arrow and label"""
    axis_local = np.zeros(4)
    axis_local[axis_idx] = 1
    axis_world = (trans_matrix @ axis_local)[:3]
    axis_end = center + length * axis_world

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


def plot_eye_anatomy(eye=Eye(), target_point=(15e-3, 15e-3, 0)):
    """Plot 3D eye anatomy with transformed coordinates"""
    eye.look_at(target_point)

    # Calculate all key points in WORLD coordinates using eye.trans
    eye_rotation_center = eye.position
    cornea_center_world = (eye.trans @ eye.pos_cornea)[:3]
    cornea_inner_center_world = (eye.trans @ eye.cornea_inner_center)[:3]
    pupil_center_world = (eye.trans @ eye.pos_pupil)[:3]
    fovea_world = (eye.trans @ np.append(eye.fovea_position, 1))[:3]

    # Eye sphere parameters
    main_eye_radius = eye.axial_length / 2
    limbus_z_local = eye.pos_apex[2] + eye.depth_cornea
    limbus_point_world = (eye.trans @ np.array([0, 0, limbus_z_local, 1]))[:3]

    # Transform corneal surfaces to world coordinates
    # Create local corneal surface coordinates first
    cornea_radius = eye.r_cornea
    depth = eye.depth_cornea
    cap_angle = np.arccos((cornea_radius - depth) / cornea_radius)
    phi_cap = np.linspace(0, cap_angle, 20)
    theta_full = np.linspace(0, 2 * np.pi, 50)
    phi_grid, theta_grid = np.meshgrid(phi_cap, theta_full)

    # Outer corneal surface in local coordinates
    x_outer_local = eye.pos_cornea[0] + cornea_radius * np.sin(phi_grid) * np.cos(
        theta_grid
    )
    y_outer_local = eye.pos_cornea[1] + cornea_radius * np.sin(phi_grid) * np.sin(
        theta_grid
    )
    z_outer_local = eye.pos_cornea[2] - cornea_radius * np.cos(phi_grid)

    # Inner corneal surface in local coordinates
    cornea_inner_radius = eye.r_cornea_inner
    x_inner_local = eye.cornea_inner_center[0] + cornea_inner_radius * np.sin(
        phi_grid
    ) * np.cos(theta_grid)
    y_inner_local = eye.cornea_inner_center[1] + cornea_inner_radius * np.sin(
        phi_grid
    ) * np.sin(theta_grid)
    z_inner_local = eye.cornea_inner_center[2] - cornea_inner_radius * np.cos(phi_grid)

    # Transform corneal surfaces
    x_outer_world, y_outer_world, z_outer_world = transform_surface(
        x_outer_local, y_outer_local, z_outer_local, eye.trans
    )
    x_inner_world, y_inner_world, z_inner_world = transform_surface(
        x_inner_local, y_inner_local, z_inner_local, eye.trans
    )

    # Create eye sphere in world coordinates
    phi_eye = np.linspace(0, np.pi, 30)
    theta_eye = np.linspace(0, 2 * np.pi, 50)
    phi_grid_eye, theta_grid_eye = np.meshgrid(phi_eye, theta_eye)

    # Generate sphere coordinates in local space
    x_eye_local = main_eye_radius * np.sin(phi_grid_eye) * np.cos(theta_grid_eye)
    y_eye_local = main_eye_radius * np.sin(phi_grid_eye) * np.sin(theta_grid_eye)
    z_eye_local = main_eye_radius * np.cos(phi_grid_eye)

    # Transform to world coordinates
    x_eye_world, y_eye_world, z_eye_world = transform_surface(
        x_eye_local, y_eye_local, z_eye_local, eye.trans
    )

    # Mask out the front part where cornea is
    limbus_z_local = eye.pos_apex[2] + eye.depth_cornea
    optical_axis_world = (eye.trans @ np.array([0, 0, -1, 0]))[:3]
    optical_axis_unit = optical_axis_world / np.linalg.norm(optical_axis_world)

    # Calculate limbus position projection
    limbus_point_world = (eye.trans @ np.array([0, 0, limbus_z_local, 1]))[:3]
    limbus_projection = np.dot(
        limbus_point_world - eye_rotation_center, optical_axis_unit
    )

    # Calculate projections for all eye points
    eye_vectors = np.stack([x_eye_world, y_eye_world, z_eye_world]) - np.array(
        eye_rotation_center
    ).reshape(3, 1, 1)
    projections = np.einsum("i,ijk->jk", optical_axis_unit, eye_vectors)

    # Apply mask to keep only back part of eye
    mask = projections <= limbus_projection
    x_eye_masked, y_eye_masked, z_eye_masked = np.where(
        mask, [x_eye_world, y_eye_world, z_eye_world], np.nan
    )

    # Calculate axes in world coordinates
    axis_length = 0.02  # 20mm axis length

    # Optical axis: from rotation center in transformed -Z direction
    optical_axis_end = eye_rotation_center + axis_length * optical_axis_unit

    # Visual axis: from fovea through rotation center, extending outward
    visual_axis_direction = (eye_rotation_center - fovea_world) / np.linalg.norm(
        eye_rotation_center - fovea_world
    )
    visual_axis_end = eye_rotation_center + axis_length * visual_axis_direction

    # Create pupil ellipse in world coordinates
    pupil_radius_x = np.linalg.norm(eye.x_pupil[:3])
    pupil_radius_y = np.linalg.norm(eye.y_pupil[:3])

    # Transform pupil vectors to world coordinates
    x_pupil_world = (eye.trans @ eye.x_pupil)[:3]
    y_pupil_world = (eye.trans @ eye.y_pupil)[:3]

    # Plot everything
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")

    # Plot the eye sphere
    ax.plot_surface(
        x_eye_masked,
        y_eye_masked,
        z_eye_masked,
        alpha=0.3,
        color="lightgray",
        label="Eye Sclera",
    )

    # Plot corneal surfaces
    ax.plot_surface(
        x_outer_world,
        y_outer_world,
        z_outer_world,
        alpha=0.7,
        color="lightblue",
        label="Cornea",
    )
    ax.plot_surface(
        x_inner_world, y_inner_world, z_inner_world, alpha=0.7, color="lightblue"
    )

    # Plot reference points
    ax.scatter(
        *eye_rotation_center,
        color="black",
        s=120,
        marker="x",
        label="Eye Rotation Center",
    )
    ax.scatter(
        *cornea_center_world,
        color="darkblue",
        s=20,
        marker="o",
        label="Outer Cornea Center",
    )
    ax.scatter(
        *cornea_inner_center_world,
        color="purple",
        s=20,
        marker="o",
        label="Inner Cornea Center",
    )
    ax.scatter(*pupil_center_world, color="red", s=20, marker="o", label="Pupil Center")
    ax.scatter(*fovea_world, color="orange", s=80, marker="*", label="Fovea")

    # Plot pupil boundary
    theta_pupil = np.linspace(0, 2 * np.pi, 50)
    cos_t = np.cos(theta_pupil)
    sin_t = np.sin(theta_pupil)

    # Calculate pupil boundary points
    pupil_boundary_local = (
        eye.pos_pupil.reshape(-1, 1)
        + eye.x_pupil.reshape(-1, 1) @ cos_t.reshape(1, -1)
        + eye.y_pupil.reshape(-1, 1) @ sin_t.reshape(1, -1)
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

    # Plot optical axis
    ax.plot(
        [eye_rotation_center[0], optical_axis_end[0]],
        [eye_rotation_center[1], optical_axis_end[1]],
        [eye_rotation_center[2], optical_axis_end[2]],
        "b--",
        linewidth=1,
        label="Optical Axis",
    )

    # Plot visual axis
    ax.plot(
        [fovea_world[0], visual_axis_end[0]],
        [fovea_world[1], visual_axis_end[1]],
        [fovea_world[2], visual_axis_end[2]],
        "r--",
        linewidth=1,
        label="Visual Axis",
    )

    # Plot target point
    ax.scatter(*target_point, color="green", s=100, marker="^", label="Target Point")

    # Plot all three axes
    axes_data = [(0, "X", "red"), (1, "Y", "green"), (2, "Z", "blue")]
    for axis_idx, label, color in axes_data:
        plot_axis(ax, eye_rotation_center, eye.trans, axis_idx, label, color)

    # Set labels and properties
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    ax.set_box_aspect([1, 1, 1])
    ax.set_aspect("equal")

    plt.title("Eye Anatomy")
    plt.show()


def to_3d(point):
    """Convert homogeneous (4D) coordinates to 3D Cartesian for plotting."""
    return point[:3] if len(point) > 3 else point


"""Eye tracking setup visualization module.

This module provides comprehensive visualization functions for eye tracking
setups, including 3D scene views and camera image views.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_setup(
    ax1,
    eye_data,
    look_at_target,
    lights,
    camera,
    cr_3d_list=None,
    ref_bounds=None,
    calib_points=None,
):
    """Plot the 3D eye tracking setup visualization.

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
        alpha=0.6,
        color="lightblue",
        edgecolor="blue",
        linewidth=0.1,
    )

    # Mark key anatomical points
    ax1.scatter(
        *eye_data["cornea_center_world"],
        color="green",
        s=100,
        label="Cornea Center",
    )
    ax1.scatter(
        *eye_data["pupil_world"],
        color="cornflowerblue",
        s=100,
        label="Pupil Center",
    )

    # Draw optical axis
    ax1.plot(
        [eye_data["cornea_center_world"][0], eye_data["optical_axis_end"][0]],
        [eye_data["cornea_center_world"][1], eye_data["optical_axis_end"][1]],
        [eye_data["cornea_center_world"][2], eye_data["optical_axis_end"][2]],
        "g--",
        linewidth=3,
        label="Optical Axis",
    )

    # Draw visual axis to target
    if len(look_at_target) == 3:
        look_at_target = np.append(look_at_target, 1)
    target_world = look_at_target

    ax1.plot(
        [eye_data["pupil_world"][0], target_world[0]],
        [eye_data["pupil_world"][1], target_world[1]],
        [eye_data["pupil_world"][2], target_world[2]],
        "r--",
        linewidth=3,
        label="Visual Axis",
    )

    # Add scene elements - multiple lights
    light_colors = ["yellow", "orange", "gold", "khaki"]  # Colors for different lights
    for i, light in enumerate(lights):
        light_pos = to_3d(light.position)
        color = light_colors[i % len(light_colors)]
        ax1.scatter(
            *light_pos, color=color, s=200, marker="*", label=f"Light Source {i+1}"
        )

    camera_pos = camera.position
    ax1.scatter(*camera_pos, color="black", s=200, marker="s", label="Camera")

    ax1.scatter(
        *to_3d(target_world), color="magenta", s=150, marker="D", label="Gaze Target"
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
                    *to_3d(cr_3d),
                    color=color,
                    s=80,
                    marker="o",
                    edgecolor="black",
                    linewidth=1,
                    label=f"CR {i+1}",
                )

                # Get corresponding light position
                light = lights[i]
                light_pos = to_3d(light.position)

                # Draw light ray paths (only from light to CR, not to camera)
                ax1.plot(
                    [light_pos[0], cr_3d[0]],
                    [light_pos[1], cr_3d[1]],
                    [light_pos[2], cr_3d[2]],
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
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))
    ax1.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x*1000:.0f}"))

    # Apply reference bounds if provided
    if ref_bounds:
        ax1.set_xlim(ref_bounds["x"])
        ax1.set_ylim(ref_bounds["y"])
        ax1.set_zlim(ref_bounds["z"])

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


def plot_camera_view_of_eye(
    ax2,
    camera_image,
    camera,
    cr_3d_list=None,
):
    """Plot the camera view of the eye.

    Args:
        ax2: 2D matplotlib axis
        camera_image: Dict with camera image data
        camera: Camera object with transformation and parameters
        cr_3d_list: List of corneal reflection 3D positions
    """
    ax2.cla()

    # Draw pupil in camera image
    if camera_image["pupil"] is not None and camera_image["pupil"].shape[1] > 2:
        pupil_points_img = camera_image["pupil"]
        closed_pupil_points = np.hstack((pupil_points_img, pupil_points_img[:, 0:1]))
        ax2.plot(
            closed_pupil_points[0, :],
            closed_pupil_points[1, :],
            color="cornflowerblue",
            linewidth=3,
            label="Pupil",
        )

    # Draw pupil center in camera image
    if camera_image["pc"] is not None:
        pupil_center_img = camera_image["pc"]
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
    if camera_image["cr"]:
        cr_colors = [
            "#FFE171",
            "#F9F871",
            "#FFD67C",
            "#C9AF41",
        ]  # Same colors as 3D view
        for i, cr_3d in enumerate(cr_3d_list or []):
            if (
                cr_3d is not None
                and i < len(camera_image["cr"])
                and camera_image["cr"][i] is not None
            ):
                cr_img, _, _ = camera.project(cr_3d)

                color = cr_colors[i % len(cr_colors)]
                ax2.scatter(
                    cr_img[0, 0],
                    cr_img[1, 0],
                    color=color,
                    s=80,
                    marker="o",
                    edgecolor="black",
                    linewidth=1,
                    label=f"CR {i+1}",
                )

    # Set camera image limits
    resolution = camera.resolution

    ax2.set_xlim(-resolution[0] / 2, resolution[0] / 2)
    ax2.set_ylim(-resolution[1] / 2, resolution[1] / 2)

    ax2.set_xlabel("X (pixels)")
    ax2.set_ylabel("Y (pixels)")
    ax2.set_title("Camera View of Eye")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")
    ax2.legend()

    # Add measurement annotations for multiple CRs
    if camera_image["pc"] is not None and camera_image["cr"]:
        cr_colors = ["#FFE171", "#F9F871", "#FFD67C", "#C9AF41"]
        for i, cr_3d in enumerate(cr_3d_list or []):
            if (
                cr_3d is not None
                and i < len(camera_image["cr"])
                and camera_image["cr"][i] is not None
            ):
                cr_img, _, _ = camera.project(cr_3d)
                pupil_center_img = camera_image["pc"]

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


def prepare_eye_data_for_plots(eye, look_at_target, lights, camera):
    """Prepare eye visualization data for plotting.

    Args:
        eye: Eye object with transformation matrix and anatomy
        look_at_target: Target point [x, y, z, 1] or [x, y, z]
        lights: List of Light objects with positions
        camera: Camera object with transformation and parameters

    Returns:
        dict: Contains eye_data, camera_image, and cr_3d_list for plotting
    """

    # Calculate all values once
    def transform_point(point):
        if len(point) == 3:
            point = np.append(point, 1)  # Convert to homogeneous coordinates
        world_point = eye.trans @ point
        return to_3d(world_point)  # Convert back to 3D for plotting

    # Rotate the eye toward the target
    eye.look_at(look_at_target)
    # Get eye anatomy points
    cornea_center = eye.pos_cornea
    pupil_center = eye.pos_pupil
    r_cornea = eye.r_cornea
    depth_cornea = eye.depth_cornea

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
                point = to_3d(cornea_center) + np.array(
                    [X_cornea[i, j], Y_cornea[i, j], Z_cornea[i, j]]
                )
                point_world = transform_point(np.append(point, 1))
                X_cornea[i, j] = point_world[0]
                Y_cornea[i, j] = point_world[1]
                Z_cornea[i, j] = point_world[2]

    # Calculate optical axis
    optical_axis_direction_local = np.array(
        [0, 0, -1, 0]
    )  # negative z in homogeneous coordinates
    optical_axis_direction_world = transform_point(optical_axis_direction_local)
    optical_axis_length = 0.1  # 100mm extension
    optical_axis_end = (
        cornea_center_world + optical_axis_direction_world * optical_axis_length
    )

    # Prepare eye data for 3D plot
    eye_data = {
        "X_cornea": X_cornea,
        "Y_cornea": Y_cornea,
        "Z_cornea": Z_cornea,
        "cornea_center_world": cornea_center_world,
        "pupil_world": pupil_world,
        "optical_axis_end": optical_axis_end,
    }

    # Get camera image data
    camera_image = camera.take_image(eye, lights)

    # Calculate 3D CR positions for each light
    cr_3d_list = []
    for light in lights:
        cr_3d = eye.find_cr(light, camera)
        cr_3d_list.append(cr_3d)

    return {
        "eye_data": eye_data,
        "camera_image": camera_image,
        "cr_3d_list": cr_3d_list,
    }


def plot_setup_and_camera_view(
    eye,
    look_at_target,
    lights,
    camera,
    ax1=None,
    ax2=None,
    fig=None,
    ref_bounds=None,
    calib_points=None,
):
    """Create comprehensive eye tracking visualization with 3D setup and camera view.

    Args:
        eye: Eye object with transformation matrix and anatomy
        look_at_target: Target point [x, y, z, 1] or [x, y, z]
        lights: List of Light objects with positions
        camera: Camera object with transformation and parameters
        ax1, ax2: Optional matplotlib axes for reuse
        fig: Optional matplotlib figure for reuse
        ref_bounds: Optional reference bounds dict with 'x', 'y', 'z' keys
        calib_points: Optional calibration points array to plot as black x markers

    Returns:
        fig: Matplotlib figure object
    """
    # Create figure and axes if not provided
    if ax1 is None or ax2 is None:
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)

    # Prepare all eye data
    prepared_data = prepare_eye_data_for_plots(eye, look_at_target, lights, camera)

    # Call the plotting functions
    plot_setup(
        ax1,
        prepared_data["eye_data"],
        look_at_target,
        lights,
        camera,
        prepared_data["cr_3d_list"],
        ref_bounds,
        calib_points,
    )

    if ax2 is not None:
        plot_camera_view_of_eye(
            ax2, prepared_data["camera_image"], camera, prepared_data["cr_3d_list"]
        )

    return fig


def setup_interactive_plot(eye_base, light, camera, look_at_target):
    """Setup interactive plot with reference bounds for consistent view.

    Args:
        eye_base: Base eye object
        light: Light object
        camera: Camera object
        look_at_target: Initial target point

    Returns:
        dict: Contains fig, ax1, ax2, ref_bounds for reuse
    """

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Create reference bounds from initial view
    e_ref = eye_base.copy()
    e_ref.look_at(look_at_target)
    cr_ref = e_ref.find_cr(light, camera)

    plot_setup_and_camera_view(
        e_ref, look_at_target, light, camera, cr_ref, ax1=ax1, ax2=ax2, fig=fig
    )
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    return {"fig": fig, "ax1": ax1, "ax2": ax2, "ref_bounds": ref_bounds}


def update_interactive_plot(plot_setup, eye_base, light, camera, look_at_target):
    """Update interactive plot with new target position.

    Args:
        plot_setup: Dict returned from setup_interactive_plot
        eye_base: Base eye object
        light: Light object
        camera: Camera object
        look_at_target: New target point
    """
    e = eye_base.copy()
    e.look_at(look_at_target)
    cr_3d = e.find_cr(light, camera)

    plot_setup_and_camera_view(
        e,
        look_at_target,
        light,
        camera,
        cr_3d,
        ax1=plot_setup["ax1"],
        ax2=plot_setup["ax2"],
        fig=plot_setup["fig"],
        ref_bounds=plot_setup["ref_bounds"],
    )

    plot_setup["fig"].suptitle(
        f"Target X={look_at_target[0]*1000:.1f} mm, Y={look_at_target[1]*1000:.1f} mm, Z={look_at_target[2]*1000:.1f} mm",
        fontsize=16,
    )
    plot_setup["fig"].canvas.draw_idle()
