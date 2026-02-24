"""Interactive plotting functions for dynamic visualizations.

Provides interactive eye tracking visualization with keyboard controls for real-time exploration.
Enables dynamic visualization of eye tracking with target and eye movement controls.
"""

import copy

import matplotlib.pyplot as plt
import numpy as np

from pyetsimul.log import info

from ..core import Camera, Eye, Light
from ..types import Position3D
from .coordinate_utils import prepare_eye_data_for_plots
from .integrated_plots import plot_setup_and_camera_view
from .interactive_controls import InteractiveControls
from .plot_config import create_plot_config
from .setup_plots import plot_setup


def plot_interactive_setup(eye_base: Eye, lights: list[Light], camera: Camera, target_point: Position3D) -> None:
    """Create and run interactive setup and camera view with keyboard controls.

    Args:
        eye_base: Base eye object
        lights: List of light objects
        camera: Camera object
        target_point: Initial target point

    """
    InteractiveControls.print_controls()

    config = create_plot_config()
    controls = InteractiveControls([eye_base], target_point)

    fig = plt.figure(figsize=config.layout.extra_wide, constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    e_ref = copy.deepcopy(eye_base)
    e_ref.look_at(target_point)
    plot_setup_and_camera_view(e_ref, target_point, camera, lights=lights, ax1=ax1, ax2=ax2, fig=fig)
    ref_bounds = {"x": ax1.get_xlim(), "y": ax1.get_ylim(), "z": ax1.get_zlim()}

    def update_plot() -> None:
        """Update the visualization with current eye and target positions."""
        e = copy.deepcopy(eye_base)
        e.look_at(controls.target_point)

        plot_setup_and_camera_view(
            e, controls.target_point, camera, lights=lights, ax1=ax1, ax2=ax2, fig=fig, ref_bounds=ref_bounds
        )

        eye_pos = eye_base.position
        fig.suptitle(
            f"Target X={controls.target_point.x:.1f} mm, Y={controls.target_point.y:.1f} mm, Z={controls.target_point.z:.1f} mm\n"
            f"Eye X={eye_pos.x:.1f} mm, Y={eye_pos.y:.1f} mm, Z={eye_pos.z:.1f} mm",
            fontsize=config.fonts.title,
        )
        fig.canvas.draw_idle()

    controls.set_update_callback(update_plot)
    fig.canvas.mpl_connect("key_press_event", controls.handle_key_press)
    update_plot()
    plt.show()


def plot_interactive_cameras(cameras: list[Camera], eye: Eye, target_point: Position3D) -> None:
    """Create and run interactive camera comparison with keyboard controls.

    Args:
        cameras: List of Camera objects
        eye: Eye object
        target_point: Initial target point

    """
    camera_names = [cam.name or f"Camera {i + 1}" for i, cam in enumerate(cameras)]
    info(f"Camera Comparison: {' vs '.join(camera_names)}")
    info("=" * 50)
    InteractiveControls.print_controls()
    info("=" * 50)

    config = create_plot_config()
    controls = InteractiveControls([eye], target_point, step_size=1)

    # Create figure with 3D view and camera comparison subplot
    fig = plt.figure(figsize=config.layout.wide_comparison)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Reference bounds for consistent 3D view (show all cameras)
    eye_ref = copy.deepcopy(eye)
    eye_ref.look_at(target_point)

    prepared_data_ref = prepare_eye_data_for_plots([eye_ref], [target_point], None, cameras)
    plot_setup(
        ax1,
        prepared_data_ref["eyes_data"],
        [target_point],
        None,
        cameras,
        prepared_data_ref["cr_3d_lists"],
        None,
        None,
    )
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    # Use centralized colors and markers for camera comparison
    colors = config.colors.camera_comparison
    markers = config.markers.camera_comparison

    def update() -> None:
        """Update the visualization with current eye position."""
        eye.look_at(controls.target_point)
        prepared_data = prepare_eye_data_for_plots([eye], [controls.target_point], None, cameras)

        # Clear axes
        ax1.cla()
        ax2.cla()

        # Plot 3D setup (show all cameras)
        plot_setup(
            ax1,
            prepared_data["eyes_data"],
            [controls.target_point],
            None,
            cameras,
            prepared_data["cr_3d_lists"],
            ref_bounds,
            None,
        )

        # Plot camera comparison for all cameras
        for i, camera in enumerate(cameras):
            camera_image = prepared_data["camera_images"][i]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            camera_name = camera.name or f"Camera {i + 1}"

            # Plot pupil boundary - closed loop
            if camera_image.pupil_boundary is not None:
                boundary = camera_image.pupil_boundary
                pupil_x = [p.x for p in boundary] + [boundary[0].x]
                pupil_y = [p.y for p in boundary] + [boundary[0].y]
                linestyle = (
                    config.lines.solid if i == 0 else config.lines.dashed
                )  # Solid line for first camera, dashed for others
                ax2.plot(
                    pupil_x,
                    pupil_y,
                    color=color,
                    linewidth=config.lines.thick_lines,
                    linestyle=linestyle,
                    label=f"Pupil ({camera_name})",
                )

            # Plot pupil center
            if camera_image.pupil_center is not None:
                center = camera_image.pupil_center.to_array()
                ax2.scatter(
                    center[0],
                    center[1],
                    color=color,
                    s=config.markers.small_details,
                    marker=marker,
                    linewidth=config.lines.thick_lines,
                    label=f"Center ({camera_name})",
                )

        # Set up axes
        resolution = cameras[0].camera_matrix.resolution
        ax2.set_xlim(-resolution.x / 2, resolution.x / 2)
        ax2.set_ylim(-resolution.y / 2, resolution.y / 2)
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        ax2.set_title(f"Camera View Comparison: {' vs '.join(camera_names)}")
        ax2.grid(config.elements.grid_enabled, alpha=config.lines.grid_alpha)
        if config.elements.equal_aspect:
            ax2.set_aspect("equal")
        ax2.legend()

        # Calculate and display distortion info
        if len(cameras) >= 2:
            # Compare first camera with others
            first_center = prepared_data["camera_images"][0].pupil_center
            if first_center is not None:
                first_center_array = first_center.to_array()

                # Calculate differences from first camera
                differences = []
                for i in range(1, len(cameras)):
                    camera_center = prepared_data["camera_images"][i].pupil_center
                    if camera_center is not None:
                        camera_center_array = camera_center.to_array()
                        center_diff = np.linalg.norm(camera_center_array - first_center_array)
                        camera_name = cameras[i].name or f"Camera {i + 1}"
                        differences.append(f"{camera_name}={center_diff:.3f}px")

                # Eye position info
                eye_pos = eye.position
                cam_pos = cameras[0].position
                distance = np.linalg.norm(
                    np.array([eye_pos.x, eye_pos.y, eye_pos.z]) - np.array([cam_pos.x, cam_pos.y, cam_pos.z])
                )

                fig.suptitle(
                    f"Target: ({controls.target_point.x:.0f}, {controls.target_point.y:.0f}, {controls.target_point.z:.0f})mm, "
                    f"Eye: ({eye_pos.x:.0f}, {eye_pos.y:.0f}, {eye_pos.z:.0f})mm, "
                    f"Distance: {distance:.0f}mm\n"
                    f"Pupil center differences: {', '.join(differences)}",
                    fontsize=config.fonts.subtitle,
                )

        fig.canvas.draw_idle()

    controls.set_update_callback(update)
    fig.canvas.mpl_connect("key_press_event", controls.handle_key_press)
    update()
    plt.show()


def plot_interactive_pupil_comparison(
    eye_elliptical: Eye, eye_realistic: Eye, camera: Camera, target_point: Position3D
) -> None:
    """Create and run interactive pupil comparison with keyboard controls.

    Args:
        eye_elliptical: Elliptical pupil eye object
        eye_realistic: Realistic pupil eye object
        camera: Camera object
        target_point: Initial target point

    """
    initial_pupil_radii = eye_elliptical.get_pupil_radii()

    info("Pupil Shape Comparison")
    info("=" * 50)
    InteractiveControls.print_controls(additional_controls={"Pupil Size ([/])": "[: smaller, ]: bigger"})

    config = create_plot_config()

    def handle_pupil_size_decrease(_event: object, _controls: InteractiveControls) -> None:
        """Handle [ key - make pupil smaller."""
        current_radii = eye_elliptical.get_pupil_radii()
        new_radius = max(0.5, current_radii[0] - 0.5)
        eye_elliptical.set_pupil_radii(new_radius, new_radius)
        eye_realistic.set_pupil_radii(new_radius, new_radius)

    def handle_pupil_size_increase(_event: object, _controls: InteractiveControls) -> None:
        """Handle ] key - make pupil bigger."""
        current_radii = eye_elliptical.get_pupil_radii()
        new_radius = min(5, current_radii[0] + 0.5)
        eye_elliptical.set_pupil_radii(new_radius, new_radius)
        eye_realistic.set_pupil_radii(new_radius, new_radius)

    def handle_reset_with_pupil(_event: object, controls: InteractiveControls) -> None:
        """Reset positions and pupil size."""
        # Let controls handle standard reset
        controls.reset_positions()
        # Sync realistic eye and reset pupil sizes
        eye_realistic.trans = eye_elliptical.trans.copy()
        eye_realistic.position = eye_elliptical.position
        eye_elliptical.set_pupil_radii(initial_pupil_radii[0], initial_pupil_radii[1])
        eye_realistic.set_pupil_radii(initial_pupil_radii[0], initial_pupil_radii[1])

    custom_handlers = {
        "[": handle_pupil_size_decrease,
        "]": handle_pupil_size_increase,
        " ": handle_reset_with_pupil,
    }

    controls = InteractiveControls([eye_elliptical], target_point, step_size=2.5, custom_handlers=custom_handlers)

    # Create figure with 3D view and camera comparison subplot
    fig = plt.figure(figsize=config.layout.wide_comparison)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    eye_ref = copy.deepcopy(eye_elliptical)
    eye_ref.look_at(target_point)
    prepared_data_ref = prepare_eye_data_for_plots([eye_ref], [target_point], None, [camera])
    plot_setup(
        ax1,
        prepared_data_ref["eyes_data"],
        [target_point],
        None,
        [camera],
        prepared_data_ref["cr_3d_lists"],
        None,
        None,
    )
    ref_bounds = {"x": ax1.get_xlim(), "y": ax1.get_ylim(), "z": ax1.get_zlim()}

    def update() -> None:
        """Update the visualization with current eye positions."""
        # Sync realistic eye to match elliptical eye position
        eye_realistic.trans = eye_elliptical.trans.copy()
        eye_realistic.position = eye_elliptical.position

        # Both eyes look at the same target
        eye_elliptical.look_at(controls.target_point)
        eye_realistic.look_at(controls.target_point)

        # Get pupil images in camera view
        elliptical_pupil_img, elliptical_center = eye_elliptical.get_pupil_in_camera_image(camera)
        realistic_pupil_img, realistic_center = eye_realistic.get_pupil_in_camera_image(camera)

        # Clear axes
        ax1.cla()
        ax2.cla()

        prepared_data = prepare_eye_data_for_plots([eye_elliptical], [controls.target_point], None, [camera])
        plot_setup(
            ax1,
            prepared_data["eyes_data"],
            [controls.target_point],
            None,
            [camera],
            prepared_data["cr_3d_lists"],
            ref_bounds,
            None,
        )

        # Plot camera comparison
        ax2.set_title("Pupil Shape Comparison")
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        ax2.grid(config.elements.grid_enabled, alpha=config.lines.grid_alpha)

        # Plot elliptical pupil with dashed line - closed loop
        if elliptical_pupil_img is not None:
            boundary = elliptical_pupil_img
            elliptical_x = [p.x for p in boundary] + [boundary[0].x]
            elliptical_y = [p.y for p in boundary] + [boundary[0].y]
            ax2.plot(
                elliptical_x,
                elliptical_y,
                color=config.colors.eyes[0],
                linestyle=config.lines.dashed,
                linewidth=config.lines.standard_lines,
                label="Elliptical Pupil",
            )

        if elliptical_center is not None:
            ax2.plot(
                elliptical_center.x,
                elliptical_center.y,
                color=config.colors.eyes[0],
                marker="x",
                markersize=6,
                label="Elliptical Center",
            )

        # Plot realistic pupil with solid line - closed loop
        if realistic_pupil_img is not None:
            boundary = realistic_pupil_img
            realistic_x = [p.x for p in boundary] + [boundary[0].x]
            realistic_y = [p.y for p in boundary] + [boundary[0].y]
            ax2.plot(
                realistic_x,
                realistic_y,
                color=config.colors.eyes[1],
                linestyle=config.lines.solid,
                linewidth=config.lines.standard_lines,
                label="Realistic Pupil",
            )

        if realistic_center is not None:
            ax2.plot(
                realistic_center.x,
                realistic_center.y,
                color=config.colors.eyes[1],
                marker="+",
                markersize=8,
                label="Realistic Center",
            )

        # Set up axes with camera resolution (same as camera comparison)
        resolution = camera.camera_matrix.resolution
        ax2.set_xlim(-resolution.x / 2, resolution.x / 2)
        ax2.set_ylim(-resolution.y / 2, resolution.y / 2)
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        ax2.set_title("Pupil Shape Comparison")
        ax2.grid(config.elements.grid_enabled, alpha=config.lines.grid_alpha)
        if config.elements.equal_aspect:
            ax2.set_aspect("equal")
        ax2.legend()

        # Update title with current positions and pupil size
        eye_pos = eye_elliptical.position
        pupil_radii = eye_elliptical.get_pupil_radii()
        pupil_diameter = pupil_radii[0] * 2  # diameter in mm
        fig.suptitle(
            f"Target X={controls.target_point.x:.1f} mm, Y={controls.target_point.y:.1f} mm, Z={controls.target_point.z:.1f} mm\n"
            f"Eye X={eye_pos.x:.1f} mm, Y={eye_pos.y:.1f} mm, Z={eye_pos.z:.1f} mm, Pupil diameter: {pupil_diameter:.1f}mm",
            fontsize=config.fonts.title,
        )
        fig.canvas.draw_idle()

    controls.set_update_callback(update)
    fig.canvas.mpl_connect("key_press_event", controls.handle_key_press)
    update()
    plt.show()
