"""Eyelid model as a spherical eyelid with an elliptical opening."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from ..types import Position3D
from .default_configs import EyelidDefaults


@dataclass
class Eyelid:
    """Spherical eyelid with elliptical opening on the front hemisphere.

    Attributes:
        center: Sphere center (eye rotation center in eye coordinates).
        sphere_radius: Sphere radius (typically eye axial_length/2 in meters).
        phi_max: Aperture half-angle defining anatomical footprint (mapped from limbus if needed).
        openness: Fraction of the spherical-cap area that is open (0..1). Used to size the ellipse.
        lower_cap_fraction: Fraction of the spherical-cap area allocated to the lower eyelid (for reference).
        ellipse_width_to_height: Ratio of ellipse width/height (horizontal almond look).
    """

    center: Position3D
    sphere_radius: float
    phi_max: float
    openness: float
    lower_cap_fraction: float
    ellipse_width_to_height: float

    def compute_opening_band(self) -> Tuple[float, float]:
        """Compute spherical band [phi1, phi2] using area relations for reference.

        Returns:
            (phi1, phi2) in radians
        """

        S = self.sphere_radius
        phi_max = self.phi_max

        def cap_area(phi: float) -> float:
            return 2.0 * np.pi * S * S * (1.0 - np.cos(phi))

        def band_area(phi1: float, phi2: float) -> float:
            return 2.0 * np.pi * S * S * (np.cos(phi1) - np.cos(phi2))

        A_max = cap_area(phi_max)
        target_lower = self.lower_cap_fraction * A_max

        # Solve for fixed lower cap phi1_fixed
        lo, hi = 0.0, phi_max
        for _ in range(EyelidDefaults.BISECTION_ITERATIONS_PHI1):
            mid = 0.5 * (lo + hi)
            if cap_area(mid) < target_lower:
                lo = mid
            else:
                hi = mid
        phi1_fixed = 0.5 * (lo + hi)

        # Solve for phi2 to match total openness area within cap
        target_open_area = float(self.openness) * A_max
        max_area_with_fixed = band_area(phi1_fixed, phi_max)

        if target_open_area <= max_area_with_fixed:
            phi1 = phi1_fixed
            lo, hi = phi1, phi_max
            for _ in range(EyelidDefaults.BISECTION_ITERATIONS_PHI2):
                mid = 0.5 * (lo + hi)
                if band_area(phi1, mid) < target_open_area:
                    lo = mid
                else:
                    hi = mid
            phi2 = 0.5 * (lo + hi)
        else:
            phi2 = phi_max
            lo, hi = 0.0, phi_max
            for _ in range(EyelidDefaults.BISECTION_ITERATIONS_AREA):
                mid = 0.5 * (lo + hi)
                if band_area(mid, phi2) > target_open_area:
                    lo = mid
                else:
                    hi = mid
            phi1 = 0.5 * (lo + hi)

        return float(phi1), float(phi2)

    def _max_xy_radius(self) -> float:
        """Max radial extent in the XY plane inside the anatomical footprint.

        Uses sphere radius and `phi_max` to constrain the eyelid horizontal span to
        the limbus-like footprint so left/right edges remain fixed.
        """

        return float(self.sphere_radius) * float(np.sin(self.phi_max))

    def ellipse_axes(self) -> Tuple[float, float]:
        """Compute ellipse width and height with fixed width and variable height.

        - Width is fixed by the footprint to keep left/right edges stationary
        - Height scales with openness in [0, 1] and reaches the footprint at 1.0

        Returns:
            (width, height) of the elliptical opening in the rotated ellipse frame.
        """

        r_xy = self._max_xy_radius()
        # Fix width at the footprint chord (limbus). At full openness the opening matches the footprint circle.
        width = EyelidDefaults.ELLIPSE_WIDTH_MULTIPLIER * r_xy
        height_max = EyelidDefaults.HEIGHT_MULTIPLIER * r_xy
        openness = float(np.clip(self.openness, 0.0, 1.0))
        height = openness * height_max
        return float(width), float(height)

    def ellipse_center_offset(self) -> float:
        """Vertical center offset (in rotated ellipse frame) to lock the lower edge.

        Keeps the lower eyelid approximately stationary by fixing the ellipse's
        lower edge at a constant position derived from a 5% (configurable) share
        of the maximum vertical opening height. Only the upper edge moves.

        Returns:
            y_offset of the ellipse center in the rotated ellipse frame (meters).
        """

        r_xy = self._max_xy_radius()
        height_max = EyelidDefaults.HEIGHT_MULTIPLIER * r_xy
        openness = float(np.clip(self.openness, 0.0, 1.0))
        height = openness * height_max

        # Lock the lower edge at the footprint (limbus) to avoid corneal occlusion at full openness.
        y_bottom_locked = -r_xy
        y_center = y_bottom_locked + 0.5 * height
        return float(y_center)

    def point_within_eyelid(self, p: Position3D) -> bool:
        """Predicate: True if a point lies on the eyelid surface (not the opening).

        Assumes points near the spherical surface. Front hemisphere is toward -Z.

        Args:
            p: Point to test
        """

        cx, cy, cz = self.center.x, self.center.y, self.center.z
        px, py, pz = p.x, p.y, p.z

        # Back hemisphere (pz > cz) is always eyelid skin for occlusion purposes
        if pz > cz:
            return True

        # No rotation applied
        x_rot = px - cx
        y_rot = py - cy

        width, height = self.ellipse_axes()
        if width <= 0.0 or height <= 0.0:
            return True

        # Shift to the ellipse center to keep the lower edge stationary
        y_center = self.ellipse_center_offset()
        y_rel = y_rot - y_center
        ellipse_test = (x_rot / (width / 2.0)) ** 2 + (y_rel / (height / 2.0)) ** 2

        # Inside ellipse is opening (not eyelid)
        if ellipse_test <= 1.0:
            return False

        return True

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "center": self.center.serialize(),
            "sphere_radius": float(self.sphere_radius),
            "phi_max": float(self.phi_max),
            "openness": float(self.openness),
            "lower_cap_fraction": float(self.lower_cap_fraction),
            "ellipse_width_to_height": float(self.ellipse_width_to_height),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Eyelid":
        """Deserialize from dictionary representation."""
        return cls(
            center=Position3D.deserialize(data["center"]),
            sphere_radius=data["sphere_radius"],
            phi_max=data["phi_max"],
            openness=data["openness"],
            lower_cap_fraction=data["lower_cap_fraction"],
            ellipse_width_to_height=data["ellipse_width_to_height"],
        )


def create_eyelid(
    center: Position3D,
    sphere_radius: float,
    phi_max: float,
    openness: float,
    lower_cap_fraction: float = EyelidDefaults.LOWER_CAP_FRACTION,
    ellipse_width_to_height: float = EyelidDefaults.ELLIPSE_WIDTH_TO_HEIGHT,
) -> Eyelid:
    """Factory function to create an Eyelid with explicit parameters.

    Eye should pass its center and axial radius to avoid circular dependencies.
    """
    return Eyelid(
        center=center,
        sphere_radius=sphere_radius,
        phi_max=phi_max,
        openness=openness,
        lower_cap_fraction=lower_cap_fraction,
        ellipse_width_to_height=ellipse_width_to_height,
    )
