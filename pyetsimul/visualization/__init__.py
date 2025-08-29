"""Visualization tools for eye tracking and experimental setups.

Provides unified access to 3D, camera, and interactive visualization functions for eye tracking analysis and demonstration.
"""

# Import functions from specialized modules
from .setup_plots import transform_surface, plot_axis, plot_setup
from .camera_view import plot_camera_view_of_eye
from .coordinate_utils import prepare_eye_data_for_plots
from .integrated_plots import plot_setup_and_camera_view
from .interactive import plot_interactive_setup, plot_interactive_cameras, plot_interactive_pupil_comparison
from .eye_anatomy import plot_eye_anatomy
from .gaze_accuracy_plots import GazeAccuracyPlotter
from .analysis_plots import plot_error_vectors_2d, plot_error_vectors_3d
from .interactive_calibration import create_interactive_calibration_plot

# Import individual modules for direct access
from . import setup_plots
from . import camera_view
from . import coordinate_utils
from . import integrated_plots
from . import interactive
from . import gaze_accuracy_plots
from . import analysis_plots
from . import interactive_calibration

__all__ = [
    "setup_plots",
    "camera_view",
    "coordinate_utils",
    "integrated_plots",
    "interactive",
    "gaze_accuracy_plots",
    "analysis_plots",
    "interactive_calibration",
    "plot_eye_anatomy",
    "GazeAccuracyPlotter",
    "plot_error_vectors_2d",
    "plot_error_vectors_3d",
    "create_interactive_calibration_plot",
    "transform_surface",
    "plot_axis",
    "plot_setup",
    "plot_camera_view_of_eye",
    "prepare_eye_data_for_plots",
    "plot_setup_and_camera_view",
    "plot_interactive_setup",
    "plot_interactive_cameras",
    "plot_interactive_pupil_comparison",
]
