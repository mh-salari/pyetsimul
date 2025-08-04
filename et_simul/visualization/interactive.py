"""Interactive plotting functions for dynamic visualizations.

Provides setup and update functions for real-time interactive eye tracking plots.
Enables exploration of system parameters and target positions.
"""

import matplotlib.pyplot as plt

from .integrated_plots import plot_setup_and_camera_view


def setup_interactive_plot(eye_base, light, camera, look_at_target):
    """Setup interactive plot with reference bounds for consistent view.

    Initializes 3D and camera view plots for interactive eye tracking visualization.
    Computes reference bounds for consistent axis scaling during updates.

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

    plot_setup_and_camera_view(e_ref, look_at_target, light, camera, cr_ref, ax1=ax1, ax2=ax2, fig=fig)
    xlim = ax1.get_xlim()
    ylim = ax1.get_ylim()
    zlim = ax1.get_zlim()
    ref_bounds = {"x": xlim, "y": ylim, "z": zlim}

    return {"fig": fig, "ax1": ax1, "ax2": ax2, "ref_bounds": ref_bounds}


def update_interactive_plot(plot_setup, eye_base, light, camera, look_at_target):
    """Update interactive plot with new target position.

    Updates 3D and camera view plots for a new gaze target.
    Maintains consistent axis scaling using reference bounds.

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
        f"Target X={look_at_target.x * 1000:.1f} mm, Y={look_at_target.y * 1000:.1f} mm, Z={look_at_target.z * 1000:.1f} mm",
        fontsize=16,
    )
    plot_setup["fig"].canvas.draw_idle()
