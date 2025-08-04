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
from et_simul.types import Position3D


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
