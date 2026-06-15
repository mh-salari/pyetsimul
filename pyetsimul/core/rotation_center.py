"""Gaze-direction-dependent eye rotation centre.

By default the eye model rotates the whole eye rigidly about a single fixed point at the geometric
centre of the eyeball sphere, a fixed distance behind the corneal apex. The human eye has no unique
centre of rotation: it lies about 15 mm behind the cornea in horizontal gaze and about 12.5 mm in
vertical gaze (Atchison & Smith, *Optics of the Human Eye*, 2nd ed., 2023, Section 1.7, after Fry &
Hill 1962 and Ohlendorf et al. 2022).

:class:`RotationCenter` makes the rotation-centre depth (corneal apex to pivot, in millimetres)
depend on the gaze direction, blending the horizontal and vertical values by the horizontal fraction
of the gaze eccentricity. It changes only *where* the rigid eye pivots; the cornea, pupil and every
optical surface are left untouched. Attaching it to an :class:`~pyetsimul.core.eye.Eye` is opt-in;
without it the eye keeps its single fixed centre.
"""

from dataclasses import dataclass


@dataclass
class RotationCenter:
    """Horizontal and vertical rotation-centre depths (corneal apex to pivot, in mm).

    ``horizontal_depth_mm`` applies to purely horizontal gaze and ``vertical_depth_mm`` to purely
    vertical gaze; oblique gaze blends them linearly by the horizontal fraction of the gaze
    eccentricity. Setting both equal to the model's geometric apex-to-centre distance reproduces the
    single fixed-centre behaviour exactly.
    """

    horizontal_depth_mm: float
    vertical_depth_mm: float

    def depth_for(self, horizontal_fraction: float) -> float:
        """Rotation-centre depth (mm) for a gaze with the given horizontal eccentricity fraction.

        Args:
            horizontal_fraction: Horizontal share of the gaze eccentricity, in ``[0, 1]`` (0 = purely
                vertical gaze, 1 = purely horizontal gaze).

        Returns:
            The blended corneal-apex-to-pivot depth in millimetres.

        """
        return self.vertical_depth_mm + (self.horizontal_depth_mm - self.vertical_depth_mm) * horizontal_fraction
