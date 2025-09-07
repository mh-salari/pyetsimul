"""Centralized plot styling configuration for PyEtSimul visualization system.

This module provides structured configuration classes for consistent styling across
all visualization components. It defines element-specific styling that preserves
functional visual distinctions while eliminating code duplication.
"""

from dataclasses import dataclass


@dataclass
class ColorPalettes:
    """Color definitions organized by element type for consistent visual coding."""

    # Eye identification colors - one per eye in multi-eye setups
    eyes: list[str]|None = None
    eyes_light: list[str]|None = None  # Lighter variants for surface rendering

    # Camera identification colors
    cameras: list[str]|None = None

    # Light source colors
    lights: list[str]|None = None

    # Anatomical structure colors
    eye_globe: str = "lightgray"
    cornea_outer: str = "steelblue"
    cornea_inner: str = "darkturquoise"
    pupil: str = "black"
    fovea: str = "orange"
    eyelid: str = "#836641"

    # Scene element colors
    target: str = "hotpink"
    corneal_reflection: str = "gold"
    rotation_center: str = "navy"
    calibration_points: str = "black"

    # Axis colors
    optical_axis: str = "green"
    visual_axis: str = "red"

    # Corneal reflection hex codes for different light sources
    corneal_reflections_detailed: list[str]|None = None

    # Interactive comparison colors
    camera_comparison: list[str]|None = None

    def __post_init__(self):
        """Initialize default color arrays if not provided."""
        if self.eyes is None:
            self.eyes = ["blue", "red", "green", "purple", "orange", "brown"]

        if self.eyes_light is None:
            self.eyes_light = ["lightblue", "lightcoral", "lightgreen", "plum", "moccasin", "tan"]

        if self.cameras is None:
            self.cameras = ["black", "gray", "darkgreen", "darkblue", "purple", "brown"]

        if self.lights is None:
            self.lights = ["yellow", "orange", "gold", "khaki"]

        if self.corneal_reflections_detailed is None:
            self.corneal_reflections_detailed = ["#FFE171", "#F9F871", "#FFD67C", "#C9AF41"]

        if self.camera_comparison is None:
            self.camera_comparison = ["cornflowerblue", "red", "green", "orange", "purple", "brown", "pink", "gray"]


@dataclass
class MarkerConfig:
    """Marker styling organized by visual importance hierarchy."""

    # Size hierarchy - larger markers indicate greater visual importance
    scene_elements: int = 200  # Lights, cameras - primary scene components
    key_landmarks: int = 150  # Targets, major anatomical points
    landmarks: int = 100  # Cornea centers, pupil centers
    detail_elements: int = 50  # Corneal reflections, calibration points
    small_details: int = 25  # Small pupil centers, minor features
    surface_points: int = 3  # Surface texture points

    # Specialized anatomical marker sizes
    cornea_surface_anterior: int = 1  # Anterior corneal surface points
    cornea_surface_posterior: int = 1  # Posterior corneal surface points
    cornea_center_outer: int = 30  # Outer cornea center marker
    cornea_center_inner: int = 20  # Inner cornea center marker

    # Marker styles for different comparison contexts
    camera_comparison: list[str]|None = None

    def __post_init__(self):
        """Initialize default marker style arrays."""
        if self.camera_comparison is None:
            self.camera_comparison = ["+", "x", "o", "s", "^", "v", "d", "p"]


@dataclass
class LineConfig:
    """Line styling organized by semantic purpose."""

    # Line widths by purpose
    thick_lines: float = 2.0  # Primary boundaries, important axes
    standard_lines: float = 1.0  # Default line width
    thin_lines: float = 0.1  # Surface mesh lines

    # Line styles by meaning
    solid: str = "-"  # Physical boundaries
    dashed: str = "--"  # Conceptual elements (axes, pointing directions)

    # Alpha values for visual layering
    primary_alpha: float = 0.9  # Elements that should stand out
    secondary_alpha: float = 0.7  # Supporting elements
    background_alpha: float = 0.5  # Reference/pointing lines
    grid_alpha: float = 0.3  # Background grids


@dataclass
class FontConfig:
    """Font sizing with clear hierarchical structure."""

    title: int = 14  # Main plot titles
    subtitle: int = 12  # Section titles, axis labels
    legend: int = 10  # Legend text
    annotation: int = 8  # Small annotations, detailed labels

    # Font weight for emphasis
    bold_weight: str = "bold"


@dataclass
class LayoutConfig:
    """Layout parameters for consistent plot structure."""

    # Standard figure sizes by plot type
    single_plot: tuple[int, int] = (10, 8)
    wide_comparison: tuple[int, int] = (16, 8)
    extra_wide: tuple[int, int] = (18, 8)
    integrated_view: tuple[int, int] = (20, 8)
    anatomy_detail: tuple[int, int] = (14, 10)

    # Legend positioning
    legend_outside_right: dict[str, str|tuple[float, float]]|None = None
    legend_upper_left: dict[str, str]|None = None

    def __post_init__(self):
        """Initialize legend positioning dictionaries."""
        if self.legend_outside_right is None:
            self.legend_outside_right = {"bbox_to_anchor": (1.05, 1), "loc": "upper left"}

        if self.legend_upper_left is None:
            self.legend_upper_left = {"loc": "upper left"}


@dataclass
class ElementConfig:
    """Specific styling for individual plot elements."""

    # Grid configuration
    grid_enabled: bool = True

    # Axis configuration
    equal_aspect: bool = True  # Use equal aspect ratio for spatial plots

    # Camera border styling
    camera_border_width: float = 2.0
    camera_border_alpha: float = 0.8

    # Pupil boundary styling
    pupil_boundary_width: float = 1.0
    pupil_boundary_alpha: float = 0.9

    # Corneal reflection styling
    corneal_reflection_width: float = 1.5

    # Eyelid styling
    eyelid_width: float = 2.0


@dataclass
class PlotConfig:
    """Complete plot styling configuration combining all component configurations."""

    colors: ColorPalettes
    markers: MarkerConfig
    lines: LineConfig
    fonts: FontConfig
    layout: LayoutConfig
    elements: ElementConfig


def create_plot_config() -> PlotConfig:
    """Factory function to create default plot configuration.

    Returns:
        PlotConfig: Complete styling configuration with PyEtSimul defaults
    """
    return PlotConfig(
        colors=ColorPalettes(),
        markers=MarkerConfig(),
        lines=LineConfig(),
        fonts=FontConfig(),
        layout=LayoutConfig(),
        elements=ElementConfig(),
    )
