"""Anatomical parameter variations for eye tracking simulation."""

from typing import TYPE_CHECKING

from ..core.cornea import ConicCornea
from ..core.default_configs import CorneaDefaults
from .composed_variation import ComposedVariation
from .generic import GenericEyeVariation

if TYPE_CHECKING:
    from ..core.eye import Eye
    from ..core.pupil_decentration import PupilDecentrationConfig


class PupilSizeVariation(GenericEyeVariation):
    """Pupil diameter variation with proper unit display."""

    def __init__(self, diameter_range: list[float], num_steps: int = 10) -> None:
        """Initialize pupil diameter variation.

        Args:
            diameter_range: Range of pupil diameters in meters
            num_steps: Number of steps in variation

        """
        super().__init__("pupil_diameter", diameter_range, num_steps)

    def describe(self) -> str:
        """Return a human-readable description of the pupil diameter variation."""
        min_mm = self.value_range[0] * 1000
        max_mm = self.value_range[1] * 1000
        return f"pupil diameter {min_mm:.1f}-{max_mm:.1f}mm ({self.num_steps} steps)"


class AngleKappaAlphaVariation(GenericEyeVariation):
    """Angle kappa alpha (horizontal) variation with proper unit display."""

    def __init__(self, alpha_range_deg: list[float], num_steps: int) -> None:
        """Initialize angle kappa alpha variation.

        Args:
            alpha_range_deg: Range of alpha angles in degrees
            num_steps: Number of steps in variation

        """
        super().__init__("fovea_alpha_deg", alpha_range_deg, num_steps)

    def describe(self) -> str:
        """Return a human-readable description of the angle kappa alpha variation."""
        return f"angle κ α {self.value_range[0]:.1f}°-{self.value_range[1]:.1f}° ({self.num_steps} steps)"


class AngleKappaBetaVariation(GenericEyeVariation):
    """Angle kappa beta (vertical) variation with proper unit display."""

    def __init__(self, beta_range_deg: list[float], num_steps: int) -> None:
        """Initialize angle kappa beta variation.

        Args:
            beta_range_deg: Range of beta angles in degrees
            num_steps: Number of steps in variation

        """
        super().__init__("fovea_beta_deg", beta_range_deg, num_steps)

    def describe(self) -> str:
        """Return a human-readable description of the angle kappa beta variation."""
        return f"angle κ β {self.value_range[0]:.1f}°-{self.value_range[1]:.1f}° ({self.num_steps} steps)"


def AngleKappaVariation(  # noqa: N802
    alpha_range_deg: list[float], beta_range_deg: list[float], num_steps: int
) -> ComposedVariation:
    """Angle kappa variation affecting both horizontal (alpha) and vertical (beta) components."""
    variations = [
        AngleKappaAlphaVariation(alpha_range_deg, num_steps),
        AngleKappaBetaVariation(beta_range_deg, num_steps),
    ]
    return ComposedVariation(variations, "angle_kappa")


class CorneaRadiusVariation(GenericEyeVariation):
    """Corneal anterior radius variation with proper unit display."""

    def __init__(self, radius_range_m: list[float], num_steps: int) -> None:
        """Initialize corneal radius variation.

        Args:
            radius_range_m: Range of anterior radii in meters
            num_steps: Number of steps in variation

        """
        super().__init__("cornea.anterior_radius", radius_range_m, num_steps)

    def describe(self) -> str:
        """Return a human-readable description of the corneal radius variation."""
        min_mm = self.value_range[0] * 1000
        max_mm = self.value_range[1] * 1000
        return f"corneal radius {min_mm:.1f}-{max_mm:.1f}mm ({self.num_steps} steps)"


class CorneaThicknessVariation(GenericEyeVariation):
    """Corneal thickness variation that works with both SphericalCornea and ConicCornea."""

    def __init__(self, thickness_range_m: list[float], num_steps: int) -> None:
        """Initialize corneal thickness variation.

        Args:
            thickness_range_m: [min_thickness, max_thickness] in meters
            num_steps: Number of thickness steps

        """
        super().__init__("cornea_thickness", thickness_range_m, num_steps)

    def describe(self) -> str:
        """Return a human-readable description of the corneal thickness variation."""
        min_mm = self.value_range[0] * 1000
        max_mm = self.value_range[1] * 1000
        return f"corneal thickness {min_mm:.2f}-{max_mm:.2f}mm ({self.num_steps} steps)"

    def apply_to_eye(self, eye: "Eye", value: float) -> None:  # noqa: PLR6301
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


class PupilSizeWithDecentrationVariation(GenericEyeVariation):
    """Pupil size variation with decentration effects enabled."""

    def __init__(
        self, diameter_range: list[float], decentration_config: "PupilDecentrationConfig", num_steps: int = 10
    ) -> None:
        """Initialize pupil size variation with decentration.

        Args:
            diameter_range: Range of pupil diameters in meters
            decentration_config: PupilDecentrationConfig to apply
            num_steps: Number of steps in variation

        """
        super().__init__("pupil_diameter", diameter_range, num_steps)
        self.decentration_config = decentration_config

    def describe(self) -> str:
        """Return description of pupil size variation with decentration."""
        min_mm = self.value_range[0] * 1000
        max_mm = self.value_range[1] * 1000

        config = self.decentration_config
        baseline_mm = config.baseline_diameter * 1000 if config.baseline_diameter else "auto"

        decentration_details = f"{config.model_name}, baseline={baseline_mm}mm"
        if config.use_individual_variation:
            seed_info = f"seed={config.individual_seed}" if config.individual_seed else "random"
            decentration_details += f", individual_variation ({seed_info})"
        else:
            params_str = ", ".join(f"{k}={v:.2f}" for k, v in config.get_model_params().items())
            decentration_details += f", population averages: {params_str}"

        return f"pupil diameter {min_mm:.1f}-{max_mm:.1f}mm with decentration ({self.num_steps} steps, {decentration_details})"

    def apply_to_eye(self, eye: "Eye", value: float) -> None:
        """Apply decentration config and set pupil diameter."""
        # Apply the decentration config
        eye.decentration_config = self.decentration_config
        # Set pupil diameter - this will trigger decentration
        eye.set_pupil_diameter(value)


class PupilDecentrationVariation(GenericEyeVariation):
    """Pupil decentration variation without changing pupil size.

    Uses move_pupil_position to apply offsets in x, y, and/or z directions.
    """

    def __init__(
        self,
        dx_range: list[float] | None = None,
        dy_range: list[float] | None = None,
        dz_range: list[float] | None = None,
        num_steps: int = 10,
    ) -> None:
        """Initialize pupil decentration variation.

        Args:
            dx_range: [min_dx, max_dx] range for x offset in meters
            dy_range: [min_dy, max_dy] range for y offset in meters
            dz_range: [min_dz, max_dz] range for z offset in meters
            num_steps: Number of steps in variation

        """
        if dx_range is None and dy_range is None and dz_range is None:
            raise ValueError("At least one of dx_range, dy_range, or dz_range must be provided")

        # Use first available range as primary for GenericEyeVariation
        primary_range = dx_range or dy_range or dz_range
        super().__init__("pupil_decentration", primary_range, num_steps)

        self.dx_range = dx_range
        self.dy_range = dy_range
        self.dz_range = dz_range

    def describe(self) -> str:
        """Return description of pupil decentration variation."""
        ranges_desc = []
        if self.dx_range:
            min_dx, max_dx = self.dx_range
            ranges_desc.append(f"dx: {min_dx * 1000:.2f}-{max_dx * 1000:.2f}mm")
        if self.dy_range:
            min_dy, max_dy = self.dy_range
            ranges_desc.append(f"dy: {min_dy * 1000:.2f}-{max_dy * 1000:.2f}mm")
        if self.dz_range:
            min_dz, max_dz = self.dz_range
            ranges_desc.append(f"dz: {min_dz * 1000:.2f}-{max_dz * 1000:.2f}mm")

        ranges_str = ", ".join(ranges_desc)
        return f"pupil decentration {ranges_str} ({self.num_steps} steps)"

    def apply_to_eye(self, eye: "Eye", value: float) -> None:
        """Apply pupil decentration by moving pupil position."""
        # Calculate interpolation factor (0 to 1)
        factor = (value - self.value_range[0]) / (self.value_range[1] - self.value_range[0])

        # Calculate offsets for each axis
        dx = 0.0
        dy = 0.0
        dz = 0.0

        if self.dx_range:
            min_dx, max_dx = self.dx_range
            dx = min_dx + factor * (max_dx - min_dx)

        if self.dy_range:
            min_dy, max_dy = self.dy_range
            dy = min_dy + factor * (max_dy - min_dy)

        if self.dz_range:
            min_dz, max_dz = self.dz_range
            dz = min_dz + factor * (max_dz - min_dz)

        # Apply the pupil position offset
        eye.move_pupil_position(dx, dy, dz)
