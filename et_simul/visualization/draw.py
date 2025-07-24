"""Visualization functions for ET Simul.

This module provides 3D visualization and plotting functions for eye tracking
simulation components including eyes, cameras, lights, and complete scenes.
"""

import numpy as np
import matplotlib.pyplot as plt


def draw_eye_anatomy(eye, target_point=None, show_axes=True, show_annotations=True):
    """Draw detailed eye anatomy with optional gaze target.

    Args:
        eye: Eye structure
        target_point: Optional target point [x, y, z] for gaze visualization
        show_axes: Whether to show coordinate axes
        show_annotations: Whether to show anatomical labels
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Transform anatomical points to world coordinates
    def transform_point(point):
        if len(point) == 3:
            point = np.append(point, 1)
        return (eye.trans @ point)[:3]

    # Get key anatomical points
    cornea_center_world = transform_point(eye.pos_cornea[:3])
    pupil_center_world = transform_point(eye.pos_pupil[:3])
    apex_world = transform_point(eye.pos_apex[:3])

    # Draw corneal surface as a sphere (simplified)
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 20)
    r = eye.r_cornea

    # Create sphere coordinates
    x_sphere = r * np.outer(np.cos(u), np.sin(v))
    y_sphere = r * np.outer(np.sin(u), np.sin(v))
    z_sphere = r * np.outer(np.ones(np.size(u)), np.cos(v))

    # Only show the anterior surface (cap) - mask the posterior part
    mask = z_sphere > -r + eye.depth_cornea
    x_sphere[mask] = np.nan
    y_sphere[mask] = np.nan
    z_sphere[mask] = np.nan

    # Transform sphere points to world coordinates
    for i in range(x_sphere.shape[0]):
        for j in range(x_sphere.shape[1]):
            if not np.isnan(x_sphere[i, j]):
                # Point relative to cornea center
                point_local = np.array([x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]])
                # Add to cornea center in eye coordinates, then transform
                point_eye = eye.pos_cornea[:3] + point_local
                point_world = transform_point(point_eye)
                x_sphere[i, j] = point_world[0]
                y_sphere[i, j] = point_world[1]
                z_sphere[i, j] = point_world[2]

    # Plot cornea surface
    ax.plot_surface(
        x_sphere,
        y_sphere,
        z_sphere,
        alpha=0.3,
        color="lightblue",
        edgecolor="blue",
        linewidth=0.1,
    )

    # Draw corneal outline circle
    theta = np.linspace(0, 2 * np.pi, 50)
    cornea_radius = eye.r_cornea

    # Create circle in YZ plane around cornea center
    cornea_y = cornea_center_world[1] + cornea_radius * np.cos(theta)
    cornea_z = cornea_center_world[2] + cornea_radius * np.sin(theta)
    cornea_x = np.full_like(cornea_y, cornea_center_world[0])
    ax.plot(cornea_x, cornea_y, cornea_z, "c-", linewidth=3, label="Cornea")

    # Draw pupil circle
    pupil_radius = np.linalg.norm(eye.across_pupil[:3])
    pupil_y = pupil_center_world[1] + pupil_radius * np.cos(theta)
    pupil_z = pupil_center_world[2] + pupil_radius * np.sin(theta)
    pupil_x = np.full_like(pupil_y, pupil_center_world[0])
    ax.plot(pupil_x, pupil_y, pupil_z, "m-", linewidth=3, label="Pupil")

    # Mark key anatomical points
    ax.scatter(
        *cornea_center_world, color="cyan", s=100, marker="o", label="Cornea Center"
    )
    ax.scatter(
        *pupil_center_world, color="magenta", s=100, marker="s", label="Pupil Center"
    )
    ax.scatter(*apex_world, color="green", s=100, marker="^", label="Corneal Apex")

    # Draw optical axis (from apex through pupil center)
    optical_direction = pupil_center_world - apex_world
    optical_direction = optical_direction / np.linalg.norm(optical_direction)
    optical_end = apex_world + optical_direction * 0.05  # 5cm extension
    ax.plot(
        [apex_world[0], optical_end[0]],
        [apex_world[1], optical_end[1]],
        [apex_world[2], optical_end[2]],
        "k--",
        linewidth=3,
        label="Optical Axis",
    )

    # Show gaze direction if target provided
    if target_point is not None:
        if len(target_point) == 3:
            target_point = np.array(target_point)
        ax.plot(
            [pupil_center_world[0], target_point[0]],
            [pupil_center_world[1], target_point[1]],
            [pupil_center_world[2], target_point[2]],
            "r-",
            linewidth=3,
            label="Gaze Direction",
        )
        ax.scatter(
            *target_point[:3], color="red", s=150, marker="*", label="Gaze Target"
        )

    # Show coordinate axes
    if show_axes:
        eye_origin = transform_point([0, 0, 0])
        axis_length = 0.02

        # X, Y, Z axes
        for i, (color, label) in enumerate([("r", "X"), ("g", "Y"), ("b", "Z")]):
            axis_end = eye_origin + eye.trans[:3, i] * axis_length
            ax.plot(
                [eye_origin[0], axis_end[0]],
                [eye_origin[1], axis_end[1]],
                [eye_origin[2], axis_end[2]],
                color=color,
                linewidth=2,
                label=f"{label}-axis",
            )

    # Labels and formatting
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Eye Anatomy")

    if show_annotations:
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    # Set appropriate limits based on eye size
    center = transform_point([0, 0, 0])
    max_range = 0.02  # 2cm range
    ax.set_xlim([center[0] - max_range, center[0] + max_range])
    ax.set_ylim([center[1] - max_range, center[1] + max_range])
    ax.set_zlim([center[2] - max_range, center[2] + max_range])

    plt.tight_layout()
    return fig


"""Eye tracking setup visualization module.

This module provides comprehensive visualization functions for eye tracking
setups, including 3D scene views and camera image views.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_setup(
    ax1,
    eye_data,
    target_point,
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
        target_point: Target point [x, y, z, 1] or [x, y, z]
        lights: List of Light objects with positions
        camera: Camera object with transformation and parameters
        cr_3d_list: List of corneal reflection 3D positions
        ref_bounds: Optional reference bounds dict with 'x', 'y', 'z' keys
        calib_points: Optional calibration points array plot_setup_and_camera_view to plot as black x markers
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
        *eye_data["cornea_center_world"][:3],
        color="green",
        s=100,
        label="Cornea Center",
    )
    ax1.scatter(
        *eye_data["pupil_world"][:3],
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
    if len(target_point) == 3:
        target_point = np.append(target_point, 1)
    target_world = target_point

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
        light_pos = light.position[:3]
        color = light_colors[i % len(light_colors)]
        ax1.scatter(
            *light_pos, color=color, s=200, marker="*", label=f"Light Source {i+1}"
        )

    camera_pos = camera.trans[:3, 3]
    ax1.scatter(*camera_pos, color="black", s=200, marker="s", label="Camera")

    ax1.scatter(
        *target_world[:3], color="magenta", s=150, marker="D", label="Gaze Target"
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
                    *cr_3d[:3],
                    color=color,
                    s=80,
                    marker="o",
                    edgecolor="black",
                    linewidth=1,
                    label=f"CR {i+1}",
                )

                # Get corresponding light position
                light = lights[i]
                light_pos = light.position[:3]

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


def plot_setup_and_camera_view(
    eye,
    target_point,
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
        target_point: Target point [x, y, z, 1] or [x, y, z]
        lights: List of Light objects with positions
        camera: Camera object with transformation and parameters
        ax1, ax2: Optional matplotlib axes for reuse
        fig: Optional matplotlib figure for reuse
        ref_bounds: Optional reference bounds dict with 'x', 'y', 'z' keys
        calib_points: Optional calibration points array plot_setup_and_camera_view to plot as black x markers

    Returns:
        fig: Matplotlib figure object
    """
    # Create figure and axes if not provided
    if ax1 is None or ax2 is None:
        fig = plt.figure(figsize=(16, 8))
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)

    # Calculate all values once
    def transform_point(point):
        return eye.trans @ point

    # Get eye anatomy points
    cornea_center = eye.pos_cornea
    apex_point = eye.pos_apex
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
                point = cornea_center[:3] + np.array(
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
    optical_axis_direction_world = transform_point(optical_axis_direction_local)[:3]
    optical_axis_length = 0.1  # 100mm extension
    optical_axis_end = (
        cornea_center_world[:3] + optical_axis_direction_world * optical_axis_length
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

    # Call the plotting functions
    plot_setup(ax1, eye_data, target_point, lights, camera, cr_3d_list, ref_bounds, calib_points)

    plot_camera_view_of_eye(ax2, camera_image, camera, cr_3d_list)

    return fig


def setup_interactive_plot(eye_base, light, camera, target_point):
    """Setup interactive plot with reference bounds for consistent view.

    Args:
        eye_base: Base eye object
        light: Light object
        camera: Camera object
        target_point: Initial target point

    Returns:
        dict: Contains fig, ax1, ax2, ref_bounds for reuse
    """

    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Create reference bounds from initial view
    e_ref = eye_base.copy()
    e_ref.look_at(target_point)
    cr_ref = e_ref.find_cr(light, camera)

    plot_setup_and_camera_view(
        e_ref, target_point, light, camera, cr_ref, ax1=ax1, ax2=ax2, fig=fig
    )
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    return {"fig": fig, "ax1": ax1, "ax2": ax2, "ref_bounds": ref_bounds}


def update_interactive_plot(plot_setup, eye_base, light, camera, target_point):
    """Update interactive plot with new target position.

    Args:
        plot_setup: Dict returned from setup_interactive_plot
        eye_base: Base eye object
        light: Light object
        camera: Camera object
        target_point: New target point
    """
    e = eye_base.copy()
    e.look_at(target_point)
    cr_3d = e.find_cr(light, camera)

    plot_setup_and_camera_view(
        e,
        target_point,
        light,
        camera,
        cr_3d,
        ax1=plot_setup["ax1"],
        ax2=plot_setup["ax2"],
        fig=plot_setup["fig"],
        ref_bounds=plot_setup["ref_bounds"],
    )

    plot_setup["fig"].suptitle(
        f"Target X={target_point[0]*1000:.1f} mm, Y={target_point[1]*1000:.1f} mm, Z={target_point[2]*1000:.1f} mm",
        fontsize=16,
    )
    plot_setup["fig"].canvas.draw_idle()
