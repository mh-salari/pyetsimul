"""Integrated plotting functions for complete visualizations.

Provides high-level functions that combine 3D setup and camera views for comprehensive eye tracking visualization.
"""

import matplotlib.pyplot as plt


from .coordinate_utils import prepare_eye_data_for_plots
from .setup_plots import plot_setup
from .camera_view import plot_camera_view_of_eye


def plot_setup_and_camera_view(
    eye,
    look_at_target,
    camera,
    lights=None,
    calib_points=None,
    ax1=None,
    ax2=None,
    fig=None,
    ref_bounds=None,
):
    """Create comprehensive eye tracking visualization with 3D setup and camera view.

    Integrates 3D scene and camera image for full system visualization.
    Useful for debugging, demonstration, and analysis of eye tracking setups.

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
    axes_provided = ax1 is not None and ax2 is not None
    if ax1 is None or ax2 is None:
        if fig is None:
            fig = plt.figure(figsize=(16, 8))
        else:
            fig.clear()  # Clear existing plots when reusing figure
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
        plot_camera_view_of_eye(ax2, prepared_data["camera_image"], camera, prepared_data["cr_3d_list"])

    # Show plot if axes were not provided (user didn't create their own figure)
    if not axes_provided:
        plt.show()

    return fig
