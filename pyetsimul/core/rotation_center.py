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
    """Separate horizontal (azimuth) and vertical (elevation) rotation centres.

    Each centre is a corneal-apex-to-pivot depth (mm) plus an optional lateral displacement off the
    optical axis: the horizontal centre is displaced nasally by ``horizontal_nasal_mm`` and the
    vertical centre superiorly by ``vertical_superior_mm``. Fry & Hill (1962, 1963) measured the
    azimuthal centre about 14.7 mm behind the apex and 0.79 mm nasal, and the elevation centre about
    12.2 mm behind the apex and 0.33 mm superior; Aguirre (2019, Sci. Rep. 9:9360; ``gkaModelEye``
    ``+human/rotationCenters.m``) uses two such centres to model the appearance of the rotated eye.

    ``depth_for`` and ``lateral_for`` blend the two centres linearly by the horizontal fraction of the
    gaze eccentricity, so purely horizontal gaze pivots about the azimuth centre and purely vertical
    gaze about the elevation centre. The lateral magnitudes default to zero, and setting both depths
    equal to the model's geometric apex-to-centre distance reproduces the single fixed-centre
    behaviour exactly.
    """

    horizontal_depth_mm: float
    vertical_depth_mm: float
    horizontal_nasal_mm: float = 0.0
    vertical_superior_mm: float = 0.0
    # Optional gaze-varying vertical centre: the elevation rotation-centre depth may differ between up-gaze and
    # down-gaze, since the human centre of rotation is not a single fixed point even within the vertical
    # meridian. ``None`` falls back to ``vertical_depth_mm`` for that direction, so an unset pair reproduces the
    # single fixed vertical depth.
    vertical_up_depth_mm: float | None = None
    vertical_down_depth_mm: float | None = None
    # When True, azimuth and elevation rotate about their own centres applied sequentially (Fick), as in
    # Aguirre (2019) / gkaModelEye, instead of blending both into a single on-axis pivot. The depth/lateral
    # fields above become the azimuth (horizontal) and elevation (vertical) Fick centres.
    fick: bool = False

    def vertical_depth(self, elevation: float) -> float:
        """Elevation rotation-centre depth (mm) for a gaze elevation (rad; ``>= 0`` up, ``< 0`` down).

        Returns ``vertical_up_depth_mm`` for up-gaze and ``vertical_down_depth_mm`` for down-gaze when those
        are set, otherwise ``vertical_depth_mm`` for both -- so an unset gaze-varying centre reproduces the
        single vertical depth exactly.
        """
        if elevation >= 0.0:
            return self.vertical_depth_mm if self.vertical_up_depth_mm is None else self.vertical_up_depth_mm
        return self.vertical_depth_mm if self.vertical_down_depth_mm is None else self.vertical_down_depth_mm

    def depth_for(self, horizontal_fraction: float, elevation: float = 0.0) -> float:
        """Rotation-centre depth (mm) for a gaze with the given horizontal eccentricity fraction.

        Args:
            horizontal_fraction: Horizontal share of the gaze eccentricity, in ``[0, 1]`` (0 = purely
                vertical gaze, 1 = purely horizontal gaze).
            elevation: Gaze elevation (rad; ``>= 0`` up, ``< 0`` down), selecting the up/down vertical
                centre when a gaze-varying vertical depth is set.

        Returns:
            The blended corneal-apex-to-pivot depth in millimetres.

        """
        vertical = self.vertical_depth(elevation)
        return vertical + (self.horizontal_depth_mm - vertical) * horizontal_fraction

    def lateral_for(self, horizontal_fraction: float, which_eye: str) -> tuple[float, float]:
        """Eye-local lateral pivot offset ``(x, y)`` in mm for the given horizontal fraction.

        The azimuth centre's nasal displacement is signed by eye side (nasal is +x for the left eye
        and -x for the right, matching :mod:`~pyetsimul.core.off_axis_pupil`); the elevation centre's
        superior displacement is +y for both eyes. Each is weighted by how horizontal (nasal) or
        vertical (superior) the gaze is, so the lateral offset vanishes at the centre and matches the
        relevant centre at the cardinal directions.

        Args:
            horizontal_fraction: Horizontal share of the gaze eccentricity, in ``[0, 1]``.
            which_eye: ``"left"`` or ``"right"``; sets the nasal sign.

        Returns:
            The eye-local ``(x, y)`` pivot offset in millimetres.

        """
        nasal = -self.horizontal_nasal_mm if which_eye == "right" else self.horizontal_nasal_mm
        return nasal * horizontal_fraction, self.vertical_superior_mm * (1.0 - horizontal_fraction)
