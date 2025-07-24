"""Visualization tools for eye and setups.
"""

from . import draw
from .draw import (
    draw_eye_anatomy,
    plot_setup,
    plot_camera_view_of_eye,
    plot_setup_and_camera_view,
    setup_interactive_plot,
    update_interactive_plot,
)

__all__ = [
    "draw",
    "draw_eye_anatomy",
    "plot_setup",
    "plot_camera_view_of_eye", 
    "plot_setup_and_camera_view",
    "setup_interactive_plot",
    "update_interactive_plot",
]
