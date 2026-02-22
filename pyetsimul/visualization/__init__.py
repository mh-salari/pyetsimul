"""Visualization tools for eye tracking and experimental setups.

Provides unified access to 3D, camera, and interactive visualization functions for eye tracking analysis and demonstration.
"""

# Import functions from specialized modules
# Import individual modules for direct access
from . import (
    analysis_plots,
    camera_view,
    coordinate_utils,
    gaze_accuracy_plots,
    integrated_plots,
    interactive,
    interactive_gaze_plot,
    setup_plots,
)
from .analysis_plots import plot_error_vectors_2d, plot_error_vectors_3d
from .camera_view import plot_camera_view_of_eye
from .coordinate_utils import prepare_eye_data_for_plots
from .eye_anatomy import plot_eye_anatomy
from .gaze_accuracy_plots import GazeAccuracyPlotter
from .integrated_plots import plot_setup_and_camera_view
from .interactive import plot_interactive_cameras, plot_interactive_pupil_comparison, plot_interactive_setup
from .interactive_gaze_plot import create_interactive_gaze_plot
from .setup_plots import plot_axis, plot_setup, transform_surface

__all__ = [
    "GazeAccuracyPlotter",
    "analysis_plots",
    "camera_view",
    "coordinate_utils",
    "create_interactive_gaze_plot",
    "gaze_accuracy_plots",
    "integrated_plots",
    "interactive",
    "interactive_gaze_plot",
    "plot_axis",
    "plot_camera_view_of_eye",
    "plot_error_vectors_2d",
    "plot_error_vectors_3d",
    "plot_eye_anatomy",
    "plot_interactive_cameras",
    "plot_interactive_pupil_comparison",
    "plot_interactive_setup",
    "plot_setup",
    "plot_setup_and_camera_view",
    "prepare_eye_data_for_plots",
    "setup_plots",
    "transform_surface",
]
