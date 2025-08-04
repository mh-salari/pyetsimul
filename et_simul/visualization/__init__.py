"""Visualization tools for eye tracking and experimental setups.

Provides unified access to 3D, camera, and interactive visualization functions for eye tracking analysis and demonstration.
"""

# Import functions from specialized modules
from .setup_plots import transform_surface, plot_axis, plot_setup
from .camera_view import plot_camera_view_of_eye
from .coordinate_utils import prepare_eye_data_for_plots
from .integrated_plots import plot_setup_and_camera_view
from .interactive import setup_interactive_plot, update_interactive_plot
from .eye_anatomy import plot_eye_anatomy

# Import individual modules for direct access
from . import setup_plots
from . import camera_view
from . import coordinate_utils
from . import integrated_plots
from . import interactive

__all__ = [
    "setup_plots",
    "camera_view",
    "coordinate_utils",
    "integrated_plots",
    "interactive",
    "plot_eye_anatomy",
    "transform_surface",
    "plot_axis",
    "plot_setup",
    "plot_camera_view_of_eye",
    "prepare_eye_data_for_plots",
    "plot_setup_and_camera_view",
    "setup_interactive_plot",
    "update_interactive_plot",
]
