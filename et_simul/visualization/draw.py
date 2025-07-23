"""Visualization functions for ET Simul.

This module provides 3D visualization and plotting functions for eye tracking
simulation components including eyes, cameras, lights, and complete scenes.
"""

import numpy as np
import matplotlib.pyplot as plt


def draw_scene(camera, lights, eye):
    """Draw complete eye tracking scene.

    Args:
        camera: Camera structure
        lights: List of light structures
        eye: Eye structure
    """
    plt.figure(figsize=(12, 8))

    # 3D scene view
    ax = plt.subplot(1, 1, 1, projection="3d")  # Changed to 1,1,1 for single plot

    # Draw Camera
    pos = camera.trans[:3, 3]
    x_axis = camera.trans[:3, 0] * 0.05  # Scale for visibility
    y_axis = camera.trans[:3, 1] * 0.05
    z_axis = camera.trans[:3, 2] * 0.05

    ax.plot(
        [pos[0], pos[0] + x_axis[0]],
        [pos[1], pos[1] + x_axis[1]],
        [pos[2], pos[2] + x_axis[2]],
        "r-",
    )
    ax.plot(
        [pos[0], pos[0] + y_axis[0]],
        [pos[1], pos[1] + y_axis[1]],
        [pos[2], pos[2] + y_axis[2]],
        "g-",
    )
    ax.plot(
        [pos[0], pos[0] + z_axis[0]],
        [pos[1], pos[1] + z_axis[1]],
        [pos[2], pos[2] + z_axis[2]],
        "b-",
    )
    ax.plot([pos[0]], [pos[1]], [pos[2]], "ks", markersize=10)

    # Draw Eye
    eye_pos = eye.trans[:3, 3]
    x_axis = eye.trans[:3, 0] * 0.02
    y_axis = eye.trans[:3, 1] * 0.02
    z_axis = eye.trans[:3, 2] * 0.02

    ax.plot(
        [eye_pos[0], eye_pos[0] + x_axis[0]],
        [eye_pos[1], eye_pos[1] + x_axis[1]],
        [eye_pos[2], eye_pos[2] + x_axis[2]],
        "r-",
    )
    ax.plot(
        [eye_pos[0], eye_pos[0] + y_axis[0]],
        [eye_pos[1], eye_pos[1] + y_axis[1]],
        [eye_pos[2], eye_pos[2] + y_axis[2]],
        "g-",
    )
    ax.plot(
        [eye_pos[0], eye_pos[0] + z_axis[0]],
        [eye_pos[1], eye_pos[1] + z_axis[1]],
        [eye_pos[2], eye_pos[2] + z_axis[2]],
        "b-",
    )

    cornea_center = eye.trans @ eye.pos_cornea
    ax.plot(
        [cornea_center[0]], [cornea_center[1]], [cornea_center[2]], "co", markersize=8
    )

    pupil_center = eye.trans @ eye.pos_pupil
    ax.plot([pupil_center[0]], [pupil_center[1]], [pupil_center[2]], "mo", markersize=8)

    # Draw Lights
    for light in lights:
        pos = light.position[:3]
        ax.plot([pos[0]], [pos[1]], [pos[2]], "yo", markersize=10)

    # Set equal aspect ratio and labels
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("Eye Tracking Scene (3D View)")

    plt.tight_layout()
    plt.show()


def draw_setup(et):
    """Draws the setup of an eye tracker.

    et_draw_setup(et) draws the cameras and lights of the eye tracker 'et'
    along with an eye at a default position.

    Args:
        et: Eye tracker structure
    """
    from ..core.eye import Eye

    # Line 26-29: Default observer position
    observer_dist = 0.5
    observer_x = 0
    observer_y = 0.2

    # Line 31: e=eye_make(7.98e-3, [1 0 0; 0 0 1; 0 1 0]);
    rest_pos = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
    e = Eye(r_cornea=7.98e-3, rest_pos=rest_pos, fovea_displacement=False)

    # Line 32: e.trans(1:3,4)=[observer_x observer_dist observer_y]';
    e.trans[:3, 3] = np.array([observer_x, observer_dist, observer_y])

    # Line 34: draw_scene(et.cameras, et.lights, e)
    draw_scene(et.cameras[0], et.lights, e)


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
