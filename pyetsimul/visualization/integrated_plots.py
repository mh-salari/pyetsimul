"""Integrated plotting functions for complete visualizations.

Provides high-level functions that combine 3D setup and camera views for comprehensive eye tracking visualization.
"""

import matplotlib.pyplot as plt


from .coordinate_utils import prepare_eye_data_for_plots
from .setup_plots import plot_setup
from .camera_view import plot_camera_view_of_eye


def plot_setup_and_camera_view(
    eyes,
    look_at_targets,
    cameras=None,
    lights=None,
    calib_points=None,
    ax1=None,
    ax2=None,
    fig=None,
    ref_bounds=None,
):
    """Create comprehensive eye tracking visualization with 3D setup and camera view.

    Shows 3D scene and camera image for complete system visualization.

    Args:
        eyes: List of Eye objects
        look_at_targets: List of target points for each eye
        cameras: Optional list of Camera objects
        lights: Optional list of Light objects with positions
        calib_points: Optional calibration points array
        ax1, ax2: Optional matplotlib axes for reuse
        fig: Optional matplotlib figure for reuse
        ref_bounds: Optional reference bounds dict with 'x', 'y', 'z' keys

    Returns:
        fig: Matplotlib figure object
    """
    # Ensure inputs are lists
    if not isinstance(eyes, list):
        eyes = [eyes]
    if not isinstance(look_at_targets, list):
        look_at_targets = [look_at_targets]
    if cameras is not None and not isinstance(cameras, list):
        cameras = [cameras]

    # Create figure and axes if not provided
    axes_provided = ax1 is not None and ax2 is not None
    if ax1 is None or ax2 is None:
        if fig is None:
            fig = plt.figure(figsize=(20, 8))
        else:
            fig.clear()
        ax1 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)

        # Adjust subplot spacing to make room for legend
        plt.subplots_adjust(right=0.85)

    # Prepare eye data
    prepared_data = prepare_eye_data_for_plots(eyes, look_at_targets, lights, cameras)

    # Plot 3D setup
    eye_colors, camera_colors = plot_setup(
        ax1,
        prepared_data["eyes_data"],
        look_at_targets,
        lights,
        cameras,
        prepared_data["cr_3d_lists"],
        ref_bounds,
        calib_points,
    )

    # Plot camera views
    if ax2 is not None and cameras:
        plot_camera_view_of_eye(
            prepared_data["camera_images"],
            cameras,
            prepared_data["cr_3d_lists"],
            ax=ax2,
            eye_colors=eye_colors,
            camera_colors=camera_colors,
        )

    # Show plot if axes were not provided
    if not axes_provided:
        plt.show()

    return fig
