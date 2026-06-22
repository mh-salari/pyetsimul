"""Static off-axis pupil position.

The centre of a real pupil does not sit on the eye's optical axis. Relative to the limbus centre it is
displaced nasally and superiorly, on average by 0.27 mm and 0.20 mm respectively with about 0.1 mm of
between-subject variation (Wyatt 1995). The displacement is a fixed property of the eye and does not
change with pupil size, which separates it from pupil decentration, the size-dependent centre shift
modelled in pupil_decentration.py.

References:
    Wyatt HJ. The form of the human pupil. Vision Research 1995, 35(14), 2021-2036.
    Atchison DA & Smith G. Optics of the Human Eye, 2nd ed., 2023, section 3.3 (Pupil Centration).

"""

from dataclasses import dataclass

from ..types import Position3D
from .default_configs import PupilDefaults


@dataclass
class OffAxisPupilConfig:
    """Static displacement of the pupil centre from its on-axis base position.

    The displacement is fixed and does not change with pupil size, which separates it from the size-dependent
    pupil decentration modelled in pupil_decentration.py. It has a lateral part in the eye-local frame and an
    axial part along the optical axis.

    By default the lateral nasal magnitude is signed by eye side: nasal points toward the nose, so it is +x for
    the left eye and -x for the right eye, with superior +y for both, matching the average nasal/superior pupil
    centration (Wyatt 1995). Set ``raw`` to override that convention and apply the nasal magnitude as +x for
    both eyes, so the lateral magnitudes act as a raw eye-local (x, y) offset; use this for a displacement
    measured or fitted per eye rather than assumed anatomically symmetric. The axial magnitude shifts the pupil
    along the optical axis (its depth behind the cornea) and is the same for both eyes.

    Attributes:
        enabled: Whether the displacement is applied. Off by default, so the pupil stays at its base position.
        nasal_mm: Nasal displacement by default, or the eye-local +x offset when raw, in mm. Filled from the
            population mean (Wyatt 1995) by resolve_defaults() when enabled and left None.
        superior_mm: Superior (eye-local +y) displacement in mm; filled the same way.
        raw: Whether to override the per-eye nasal sign convention. False (default) signs the nasal magnitude
            +x for the left eye and -x for the right; True applies it as +x for both eyes.
        axial_mm: Axial (eye-local +z) displacement in mm along the optical axis; positive moves the pupil
            deeper. Zero by default, the same for both eyes.

    """

    enabled: bool = False
    nasal_mm: float | None = None
    superior_mm: float | None = None
    raw: bool = False
    axial_mm: float = 0.0

    def resolve_defaults(self) -> None:
        """Fill nasal_mm/superior_mm from the population mean when enabled and not set explicitly."""
        if not self.enabled:
            return
        if self.nasal_mm is None:
            self.nasal_mm = PupilDefaults.OFFSET_FROM_LIMBUS[0]
        if self.superior_mm is None:
            self.superior_mm = PupilDefaults.OFFSET_FROM_LIMBUS[1]

    def offset_for_eye(self, which_eye: str) -> Position3D:
        """Eye-local pupil-centre offset (mm) for the given eye side; zero when disabled.

        The nasal magnitude is negated for the right eye unless raw is set; superior is +y throughout.
        """
        if not self.enabled:
            return Position3D(0.0, 0.0, 0.0)
        self.resolve_defaults()
        flip_nasal = not self.raw and which_eye == "right"
        x = -self.nasal_mm if flip_nasal else self.nasal_mm
        return Position3D(x, self.superior_mm, self.axial_mm)
