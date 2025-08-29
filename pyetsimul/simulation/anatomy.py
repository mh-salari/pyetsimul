"""Anatomical parameter variations for eye tracking simulation."""

from typing import List
from .generic import GenericEyeVariation
from .composed_variation import ComposedVariation
from ..core.cornea import ConicCornea
from ..core.default_configs import CorneaDefaults


class PupilSizeVariation(GenericEyeVariation):
    """Pupil diameter variation with proper unit display."""

    def __init__(self, diameter_range: List[float], num_steps: int = 10):
        super().__init__("pupil_diameter", diameter_range, num_steps)

    def describe(self) -> str:
        min_mm = self.value_range[0] * 1000
        max_mm = self.value_range[1] * 1000
        return f"pupil diameter {min_mm:.1f}-{max_mm:.1f}mm ({self.num_steps} steps)"


class AngleKappaAlphaVariation(GenericEyeVariation):
    """Angle kappa alpha (horizontal) variation with proper unit display."""

    def __init__(self, alpha_range_deg: List[float], num_steps: int):
        super().__init__("fovea_alpha_deg", alpha_range_deg, num_steps)

    def describe(self) -> str:
        return f"angle κ α {self.value_range[0]:.1f}°-{self.value_range[1]:.1f}° ({self.num_steps} steps)"


class AngleKappaBetaVariation(GenericEyeVariation):
    """Angle kappa beta (vertical) variation with proper unit display."""

    def __init__(self, beta_range_deg: List[float], num_steps: int):
        super().__init__("fovea_beta_deg", beta_range_deg, num_steps)

    def describe(self) -> str:
        return f"angle κ β {self.value_range[0]:.1f}°-{self.value_range[1]:.1f}° ({self.num_steps} steps)"


def AngleKappaVariation(
    alpha_range_deg: List[float], beta_range_deg: List[float], num_steps: int
) -> ComposedVariation:
    """Angle kappa variation affecting both horizontal (alpha) and vertical (beta) components."""
    variations = [
        AngleKappaAlphaVariation(alpha_range_deg, num_steps),
        AngleKappaBetaVariation(beta_range_deg, num_steps),
    ]
    return ComposedVariation(variations, "angle_kappa")


class CorneaRadiusVariation(GenericEyeVariation):
    """Corneal anterior radius variation with proper unit display."""

    def __init__(self, radius_range_m: List[float], num_steps: int):
        super().__init__("cornea.anterior_radius", radius_range_m, num_steps)

    def describe(self) -> str:
        min_mm = self.value_range[0] * 1000
        max_mm = self.value_range[1] * 1000
        return f"corneal radius {min_mm:.1f}-{max_mm:.1f}mm ({self.num_steps} steps)"


class CorneaThicknessVariation(GenericEyeVariation):
    """Corneal thickness variation that works with both SphericalCornea and ConicCornea."""

    def __init__(self, thickness_range_m: List[float], num_steps: int):
        """Initialize corneal thickness variation.

        Args:
            thickness_range_m: [min_thickness, max_thickness] in meters
            num_steps: Number of thickness steps
        """
        super().__init__("cornea_thickness", thickness_range_m, num_steps)

    def describe(self) -> str:
        min_mm = self.value_range[0] * 1000
        max_mm = self.value_range[1] * 1000
        return f"corneal thickness {min_mm:.2f}-{max_mm:.2f}mm ({self.num_steps} steps)"

    def apply_to_eye(self, eye, value: float) -> None:
        """Apply thickness value by setting appropriate cornea parameter."""

        cornea = eye.cornea

        # For ConicCornea: set thickness_offset directly
        if isinstance(cornea, ConicCornea):
            cornea.thickness_offset = value
            return

        # For SphericalCornea: calculate anterior_radius that gives desired thickness
        # thickness_offset = scale * _thickness_offset_default
        # scale = anterior_radius / _r_cornea_default
        # So: anterior_radius = (desired_thickness / _thickness_offset_default) * _r_cornea_default
        desired_radius = (value / CorneaDefaults.THICKNESS_OFFSET) * CorneaDefaults.ANTERIOR_RADIUS
        cornea.anterior_radius = desired_radius
