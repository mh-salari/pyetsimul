"""Interactive plotting functions for dynamic visualizations.

Provides interactive eye tracking visualization with keyboard controls for real-time exploration.
Enables dynamic visualization of eye tracking with target and eye movement controls.
"""

import numpy as np
import matplotlib.pyplot as plt
import copy

from .integrated_plots import plot_setup_and_camera_view
from .coordinate_utils import prepare_eye_data_for_plots
from .setup_plots import plot_setup
from pyetsimul.types import Position3D


def plot_interactive_setup(eye_base, lights, camera, target_point):
    """Create and run interactive setup and camera view with keyboard controls.

    Args:
        eye_base: Base eye object
        lights: List of light objects
        camera: Camera object
        target_point: Initial target point
    """
    print("CONTROLS:")
    print("Target Movement (Arrow keys):")
    print("  ↑/↓: Move target up/down")
    print("  ←/→: Move target left/right")
    print()
    print("Eye Movement (I/K/J/L/./):")
    print("  I/K: Move eye up/down")
    print("  J/L: Move eye left/right")
    print("  ./,: Move eye closer/farther from camera")

    # Create figure with 3D and camera views
    fig = plt.figure(figsize=(18, 8), constrained_layout=True)
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Reference bounds from initial view
    e_ref = copy.deepcopy(eye_base)
    e_ref.look_at(target_point)
    plot_setup_and_camera_view(e_ref, target_point, camera, lights=lights, ax1=ax1, ax2=ax2, fig=fig)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    def update_plot():
        """Update the visualization with current eye and target positions."""
        e = copy.deepcopy(eye_base)
        e.look_at(target_point)

        # Update both 3D and camera views
        plot_setup_and_camera_view(
            e,
            target_point,
            camera,
            lights=lights,
            ax1=ax1,
            ax2=ax2,
            fig=fig,
            ref_bounds=ref_bounds,
        )

        # Update title with current positions
        eye_pos = eye_base.position
        fig.suptitle(
            f"Target X={target_point.x * 1000:.1f} mm, Y={target_point.y * 1000:.1f} mm, Z={target_point.z * 1000:.1f} mm\n"
            f"Eye X={eye_pos.x * 1000:.1f} mm, Y={eye_pos.y * 1000:.1f} mm, Z={eye_pos.z * 1000:.1f} mm",
            fontsize=14,
        )
        fig.canvas.draw_idle()

    def on_key_press(event):
        """Handle keyboard input for target and eye movement."""
        nonlocal target_point
        step_size = 2.5e-3  # 2.5 mm step size

        # TARGET MOVEMENT (Arrow keys)
        if event.key in ["up", "Up", "↑"]:
            target_point = Position3D(target_point.x, target_point.y, target_point.z + step_size)
        elif event.key in ["down", "Down", "↓"]:
            target_point = Position3D(target_point.x, target_point.y, target_point.z - step_size)
        elif event.key in ["left", "Left", "←"]:
            target_point = Position3D(target_point.x - step_size, target_point.y, target_point.z)
        elif event.key in ["right", "Right", "→"]:
            target_point = Position3D(target_point.x + step_size, target_point.y, target_point.z)

        # EYE MOVEMENT (I/K/J/L/./,)
        elif event.key == "j":
            eye_base.trans[0, 3] -= step_size  # Eye left
        elif event.key == "l":
            eye_base.trans[0, 3] += step_size  # Eye right
        elif event.key == "i":
            eye_base.trans[2, 3] += step_size  # Eye up
        elif event.key == "k":
            eye_base.trans[2, 3] -= step_size  # Eye down
        elif event.key == ".":
            eye_base.trans[1, 3] -= step_size  # Eye closer to camera
        elif event.key == ",":
            eye_base.trans[1, 3] += step_size  # Eye farther from camera

        update_plot()

    # Connect keyboard events and show plot
    fig.canvas.mpl_connect("key_press_event", on_key_press)
    update_plot()
    plt.show()


def plot_interactive_cameras(cameras, eye, target_point):
    """Create and run interactive camera comparison with keyboard controls.

    Args:
        cameras: List of Camera objects
        eye: Eye object
        target_point: Initial target point
    """
    # Store initial positions for reset functionality
    initial_eye_position = Position3D(eye.position.x, eye.position.y, eye.position.z)
    initial_target_position = Position3D(target_point.x, target_point.y, target_point.z)

    # Extract camera names for display
    camera_names = [cam.name or f"Camera {i + 1}" for i, cam in enumerate(cameras)]
    print(f"Camera Comparison: {' vs '.join(camera_names)}")
    print("=" * 50)
    print("CONTROLS:")
    print("Target Movement (Arrow keys):")
    print("  ↑/↓: Move target up/down")
    print("  ←/→: Move target left/right")
    print("Eye Movement (I/K/J/L/./):")
    print("  I/K: Move eye up/down")
    print("  J/L: Move eye left/right")
    print("  ./,: Move eye closer/farther from camera")
    print("Reset (Space): Reset eye and target to initial positions")
    print("=" * 50)

    # Create figure with 3D view and camera comparison subplot
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Reference bounds for consistent 3D view (use first camera)
    eye_ref = copy.deepcopy(eye)
    eye_ref.look_at(target_point)
    first_camera = cameras[0]

    prepared_data_ref = prepare_eye_data_for_plots(eye_ref, target_point, None, first_camera)
    plot_setup(
        ax1,
        prepared_data_ref["eye_data"],
        target_point,
        None,
        first_camera,
        prepared_data_ref["cr_3d_list"],
        None,
        None,
    )
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    # Define colors for different cameras
    colors = ["cornflowerblue", "red", "green", "orange", "purple", "brown", "pink", "gray"]
    markers = ["+", "x", "o", "s", "^", "v", "d", "p"]

    def update():
        """Update the visualization with current eye position."""
        nonlocal eye, target_point

        # Make eye look at target point
        eye.look_at(target_point)

        # Prepare data for all cameras
        prepared_data_list = []
        for camera in cameras:
            prepared_data = prepare_eye_data_for_plots(eye, target_point, None, camera)
            prepared_data_list.append(prepared_data)

        # Clear axes
        ax1.cla()
        ax2.cla()

        # Plot 3D setup (use first camera)
        plot_setup(
            ax1,
            prepared_data_list[0]["eye_data"],
            target_point,
            None,
            first_camera,
            prepared_data_list[0]["cr_3d_list"],
            ref_bounds,
            None,
        )

        # Plot camera comparison for all cameras
        for i, (camera, prepared_data) in enumerate(zip(cameras, prepared_data_list)):
            camera_image = prepared_data["camera_image"]
            color = colors[i % len(colors)]
            marker = markers[i % len(markers)]
            camera_name = camera.name or f"Camera {i + 1}"

            # Plot pupil boundary
            if camera_image.pupil_boundary is not None:
                pupil_points = camera_image.pupil_boundary
                closed_points = np.hstack((pupil_points, pupil_points[:, 0:1]))
                linestyle = "-" if i == 0 else "--"  # Solid line for first camera, dashed for others
                ax2.plot(
                    closed_points[0, :],
                    closed_points[1, :],
                    color=color,
                    linewidth=1.5,
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
                    s=50,
                    marker=marker,
                    linewidth=2,
                    label=f"Center ({camera_name})",
                )

        # Set up axes
        resolution = first_camera.camera_matrix.resolution
        ax2.set_xlim(-resolution.x / 2, resolution.x / 2)
        ax2.set_ylim(-resolution.y / 2, resolution.y / 2)
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        ax2.set_title(f"Camera View Comparison: {' vs '.join(camera_names)}")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")
        ax2.legend()

        # Calculate and display distortion info
        if len(prepared_data_list) >= 2:
            # Compare first camera with others
            first_center = prepared_data_list[0]["camera_image"].pupil_center
            if first_center is not None:
                first_center_array = first_center.to_array()

                # Calculate differences from first camera
                differences = []
                for i, prepared_data in enumerate(prepared_data_list[1:], 1):
                    camera_center = prepared_data["camera_image"].pupil_center
                    if camera_center is not None:
                        camera_center_array = camera_center.to_array()
                        center_diff = np.linalg.norm(camera_center_array - first_center_array)
                        camera_name = cameras[i].name or f"Camera {i + 1}"
                        differences.append(f"{camera_name}={center_diff:.3f}px")

                # Eye position info
                eye_pos = eye.position
                cam_pos = first_camera.position
                distance = np.linalg.norm(
                    np.array([eye_pos.x, eye_pos.y, eye_pos.z]) - np.array([cam_pos.x, cam_pos.y, cam_pos.z])
                )

                fig.suptitle(
                    f"Target: ({target_point.x * 1000:.0f}, {target_point.y * 1000:.0f}, {target_point.z * 1000:.0f})mm, "
                    f"Eye: ({eye_pos.x * 1000:.0f}, {eye_pos.y * 1000:.0f}, {eye_pos.z * 1000:.0f})mm, "
                    f"Distance: {distance * 1000:.0f}mm\n"
                    f"Pupil center differences: {', '.join(differences)}",
                    fontsize=12,
                )

        fig.canvas.draw_idle()

    def on_key(event):
        """Handle keyboard input for target and eye movement."""
        nonlocal eye, target_point, initial_eye_position, initial_target_position
        step = 1e-3  # 1mm step size

        # RESET POSITIONS (Space)
        if event.key == " ":
            # Reset eye to initial position
            eye.trans[0, 3] = initial_eye_position.x
            eye.trans[1, 3] = initial_eye_position.y
            eye.trans[2, 3] = initial_eye_position.z
            eye.position = Position3D(eye.trans[0, 3], eye.trans[1, 3], eye.trans[2, 3])

            # Reset target to initial position
            target_point = Position3D(initial_target_position.x, initial_target_position.y, initial_target_position.z)

        # TARGET MOVEMENT (Arrow keys)
        elif event.key in ["up", "Up", "↑"]:
            target_point = Position3D(target_point.x, target_point.y, target_point.z + step)
        elif event.key in ["down", "Down", "↓"]:
            target_point = Position3D(target_point.x, target_point.y, target_point.z - step)
        elif event.key in ["left", "Left", "←"]:
            target_point = Position3D(target_point.x - step, target_point.y, target_point.z)
        elif event.key in ["right", "Right", "→"]:
            target_point = Position3D(target_point.x + step, target_point.y, target_point.z)

        # EYE MOVEMENT (I/K/J/L/./,)
        elif event.key == "j":
            eye.trans[0, 3] -= step  # Eye left
        elif event.key == "l":
            eye.trans[0, 3] += step  # Eye right
        elif event.key == "i":
            eye.trans[2, 3] += step  # Eye up
        elif event.key == "k":
            eye.trans[2, 3] -= step  # Eye down
        elif event.key == ".":
            eye.trans[1, 3] -= step  # Eye closer to camera
        elif event.key == ",":
            eye.trans[1, 3] += step  # Eye farther from camera
        else:
            return  # Ignore other keys

        # Update eye position property to match transformation matrix
        eye.position = Position3D(eye.trans[0, 3], eye.trans[1, 3], eye.trans[2, 3])
        update()

    # Connect keyboard events
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial update
    update()

    plt.show()


def plot_interactive_pupil_comparison(eye_elliptical, eye_realistic, camera, target_point):
    """Create and run interactive pupil comparison with keyboard controls.

    Args:
        eye_elliptical: Elliptical pupil eye object
        eye_realistic: Realistic pupil eye object
        camera: Camera object
        target_point: Initial target point
    """
    # Store initial positions and pupil size for reset functionality
    initial_eye_position = Position3D(eye_elliptical.position.x, eye_elliptical.position.y, eye_elliptical.position.z)
    initial_target_position = Position3D(target_point.x, target_point.y, target_point.z)
    initial_pupil_radii = eye_elliptical.get_pupil_radii()

    print("Pupil Shape Comparison")
    print("=" * 50)
    print("CONTROLS:")
    print("Target Movement (Arrow keys):")
    print("  ↑/↓: Move target up/down")
    print("  ←/→: Move target left/right")
    print("Eye Movement (I/K/J/L/./):")
    print("  I/K: Move eye up/down")
    print("  J/L: Move eye left/right")
    print("  ./,: Move eye closer/farther from camera")
    print("Pupil Size ([/]):")
    print("  [: Make pupil smaller")
    print("  ]: Make pupil bigger")
    print("Reset (Space): Reset eye and target to initial positions")

    # Create figure with 3D view and camera comparison subplot
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax2 = fig.add_subplot(1, 2, 2)

    # Reference bounds for consistent 3D view (same as camera comparison)
    eye_ref = copy.deepcopy(eye_elliptical)
    eye_ref.look_at(target_point)
    prepared_data_ref = prepare_eye_data_for_plots(eye_ref, target_point, None, camera)
    plot_setup(
        ax1,
        prepared_data_ref["eye_data"],
        target_point,
        None,
        camera,
        prepared_data_ref["cr_3d_list"],
        None,
        None,
    )
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    def update():
        """Update the visualization with current eye positions."""
        nonlocal target_point

        # Make both eyes look at target point
        eye_elliptical.look_at(target_point)
        eye_realistic.look_at(target_point)

        # Get pupil images in camera view
        elliptical_pupil_img, elliptical_center = eye_elliptical.get_pupil_in_camera_image(camera)
        realistic_pupil_img, realistic_center = eye_realistic.get_pupil_in_camera_image(camera)

        # Clear axes
        ax1.cla()
        ax2.cla()

        # Prepare data for 3D setup (use elliptical eye for 3D view)
        prepared_data = prepare_eye_data_for_plots(eye_elliptical, target_point, None, camera)

        # Plot 3D setup (same as camera comparison)
        plot_setup(
            ax1,
            prepared_data["eye_data"],
            target_point,
            None,
            camera,
            prepared_data["cr_3d_list"],
            ref_bounds,
            None,
        )

        # Plot camera comparison
        ax2.set_title("Pupil Shape Comparison")
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        ax2.grid(True, alpha=0.3)

        # Plot elliptical pupil with dashed line
        if elliptical_pupil_img is not None:
            elliptical_closed = np.column_stack([elliptical_pupil_img, elliptical_pupil_img[:, 0:1]])
            ax2.plot(elliptical_closed[0, :], elliptical_closed[1, :], "b--", linewidth=1, label="Elliptical Pupil")

        if elliptical_center is not None:
            ax2.plot(elliptical_center.x, elliptical_center.y, "bx", markersize=6, label="Elliptical Center")

        # Plot realistic pupil with solid line
        if realistic_pupil_img is not None:
            realistic_closed = np.column_stack([realistic_pupil_img, realistic_pupil_img[:, 0:1]])
            ax2.plot(realistic_closed[0, :], realistic_closed[1, :], "r-", linewidth=1, label="Realistic Pupil")

        if realistic_center is not None:
            ax2.plot(realistic_center.x, realistic_center.y, "r+", markersize=8, label="Realistic Center")

        # Set up axes with camera resolution (same as camera comparison)
        resolution = camera.camera_matrix.resolution
        ax2.set_xlim(-resolution.x / 2, resolution.x / 2)
        ax2.set_ylim(-resolution.y / 2, resolution.y / 2)
        ax2.set_xlabel("X (pixels)")
        ax2.set_ylabel("Y (pixels)")
        ax2.set_title("Pupil Shape Comparison")
        ax2.grid(True, alpha=0.3)
        ax2.set_aspect("equal")
        ax2.legend()

        # Update title with current positions and pupil size
        eye_pos = eye_elliptical.position
        pupil_radii = eye_elliptical.get_pupil_radii()
        pupil_diameter = pupil_radii[0] * 2000  # Convert to mm
        fig.suptitle(
            f"Target X={target_point.x * 1000:.1f} mm, Y={target_point.y * 1000:.1f} mm, Z={target_point.z * 1000:.1f} mm\n"
            f"Eye X={eye_pos.x * 1000:.1f} mm, Y={eye_pos.y * 1000:.1f} mm, Z={eye_pos.z * 1000:.1f} mm, Pupil diameter: {pupil_diameter:.1f}mm",
            fontsize=14,
        )
        fig.canvas.draw_idle()

    def on_key(event):
        """Handle keyboard input for target and eye movement."""
        nonlocal target_point
        step_size = 2.5e-3  # 2.5 mm step size

        # TARGET MOVEMENT (Arrow keys)
        if event.key in ["up", "Up", "↑"]:
            target_point = Position3D(target_point.x, target_point.y, target_point.z + step_size)
        elif event.key in ["down", "Down", "↓"]:
            target_point = Position3D(target_point.x, target_point.y, target_point.z - step_size)
        elif event.key in ["left", "Left", "←"]:
            target_point = Position3D(target_point.x - step_size, target_point.y, target_point.z)
        elif event.key in ["right", "Right", "→"]:
            target_point = Position3D(target_point.x + step_size, target_point.y, target_point.z)

        # EYE MOVEMENT (I/K/J/L/./,)
        elif event.key == "j":
            eye_elliptical.trans[0, 3] -= step_size  # Eye left
            eye_realistic.trans[0, 3] -= step_size
        elif event.key == "l":
            eye_elliptical.trans[0, 3] += step_size  # Eye right
            eye_realistic.trans[0, 3] += step_size
        elif event.key == "i":
            eye_elliptical.trans[2, 3] += step_size  # Eye up
            eye_realistic.trans[2, 3] += step_size
        elif event.key == "k":
            eye_elliptical.trans[2, 3] -= step_size  # Eye down
            eye_realistic.trans[2, 3] -= step_size
        elif event.key == ".":
            eye_elliptical.trans[1, 3] += step_size  # Eye closer
            eye_realistic.trans[1, 3] += step_size
        elif event.key == ",":
            eye_elliptical.trans[1, 3] -= step_size  # Eye farther
            eye_realistic.trans[1, 3] -= step_size

        # RESET (Space)
        elif event.key == " ":
            eye_elliptical.position = initial_eye_position
            eye_realistic.position = initial_eye_position
            target_point = initial_target_position
            # Reset pupil size to initial size
            eye_elliptical.set_pupil_radii(initial_pupil_radii[0], initial_pupil_radii[1])
            eye_realistic.set_pupil_radii(initial_pupil_radii[0], initial_pupil_radii[1])

        # PUPIL SIZE ([/])
        elif event.key == "[":
            # Make pupil smaller
            current_radii = eye_elliptical.get_pupil_radii()
            new_radius = max(0.5e-3, current_radii[0] - 0.5e-3)  # Minimum 0.5mm radius
            eye_elliptical.set_pupil_radii(new_radius, new_radius)
            eye_realistic.set_pupil_radii(new_radius, new_radius)
        elif event.key == "]":
            # Make pupil bigger
            current_radii = eye_elliptical.get_pupil_radii()
            new_radius = min(5e-3, current_radii[0] + 0.5e-3)  # Maximum 5mm radius
            eye_elliptical.set_pupil_radii(new_radius, new_radius)
            eye_realistic.set_pupil_radii(new_radius, new_radius)

        update()

    # Connect the key press event
    fig.canvas.mpl_connect("key_press_event", on_key)

    # Initial plot
    update()

    plt.show()
