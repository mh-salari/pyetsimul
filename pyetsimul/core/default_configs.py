"""Centralized default parameters for PyEtSimul core components.

Defaults for anatomical and hardware parameters.
"""

from dataclasses import dataclass

from pyetsimul.log import info


@dataclass
class EyeAnatomyDefaults:
    """Anatomical parameters based on Böhme et al. 2008, Boff & Lincoln 1988."""

    AXIAL_LENGTH: float = 24.75e-3  # m
    PUPIL_RADIUS: float = 3.0e-3  # m
    N_AQUEOUS_HUMOR: float = 1.336
    FOVEA_ALPHA_DEG: float = 6.0  # deg
    FOVEA_BETA_DEG: float = 2.0  # deg
    EYELID_OPENNESS: float = 1.0


@dataclass
class CorneaDefaults:
    """Corneal parameters from Böhme et al. 2008, Goncharov & Dainty 2007."""

    # Spherical model
    ANTERIOR_RADIUS: float = 7.98e-3  # m
    POSTERIOR_RADIUS: float = 6.22e-3  # m
    REFRACTIVE_INDEX: float = 1.376
    THICKNESS_OFFSET: float = 1.15e-3  # m
    CORNEA_DEPTH: float = 3.54e-3  # m
    CENTER_TO_ROTATION: float = 10.20e-3  # m

    # Conic model (30-year subject)
    CONIC_ANTERIOR_RADIUS: float = 7.76e-3  # m
    CONIC_ANTERIOR_K: float = -0.10
    CONIC_POSTERIOR_RADIUS: float = 6.52e-3  # m
    CONIC_POSTERIOR_K: float = -0.30
    CONIC_THICKNESS_OFFSET: float = 0.55e-3  # m


@dataclass
class PupilDefaults:
    """Pupil parameters based on Wyatt 1995."""

    BOUNDARY_POINTS_ELLIPTICAL: int = 100
    BOUNDARY_POINTS_FACTORY: int = 20
    BOUNDARY_POINTS_REALISTIC: int = 360
    BASE_RADIUS: float = 2.5e-3  # m
    NONCIRCULARITY: float = 0.0166
    ELLIPSE_CONTRIBUTION: float = 0.5
    MAJOR_AXIS_ANGLE: float = 0.0  # rad
    OFFSET_FROM_LIMBUS: tuple[float, float] = (0.27e-3, 0.20e-3)  # m
    N_HARMONICS: int = 6
    REFERENCE_AGE: float = 35.8  # years


@dataclass
class CameraDefaults:
    """Camera parameters for typical eye tracking configurations."""

    FOCAL_LENGTH: float = 2880.0  # pixels
    PRINCIPAL_POINT_X: float = 640.0  # pixels
    PRINCIPAL_POINT_Y: float = 512.0  # pixels
    RESOLUTION_WIDTH: int = 1280  # pixels
    RESOLUTION_HEIGHT: int = 1024  # pixels
    MEASUREMENT_ERROR: float = 0.0  # pixels


@dataclass
class EyelidDefaults:
    """Eyelid shape and numerical parameters."""

    LOWER_CAP_FRACTION: float = 0.5
    ELLIPSE_WIDTH_TO_HEIGHT: float = 1.5
    ELLIPSE_WIDTH_MULTIPLIER: float = 2.0
    HEIGHT_MULTIPLIER: float = 2.0
    BISECTION_ITERATIONS_PHI1: int = 80
    BISECTION_ITERATIONS_PHI2: int = 100
    BISECTION_ITERATIONS_AREA: int = 120


@dataclass
class PupilDecentrationDefaults:
    """Pupil decentration parameters from Wildenmann & Schaeffel (2013).

    Reference: Wildenmann U & Schaeffel F. Variations of pupil centration and their
    effects on video eye tracking. Ophthalmic Physiol Opt 2013, 33, 634-641.

    """

    # Right eye (OD): horizontal -0.03±0.07
    RIGHT_EYE_X_COEFF: float = -0.03  # mm/mm
    RIGHT_EYE_X_STD: float = 0.07  # mm/mm
    # Right eye (OD): vertical -0.04±0.06 mm/mm
    RIGHT_EYE_Y_COEFF: float = -0.04  # mm/mm
    RIGHT_EYE_Y_STD: float = 0.06  # mm/mm

    # Left eye (OS): horizontal +0.03±0.04
    LEFT_EYE_X_COEFF: float = 0.03  # mm/mm
    LEFT_EYE_X_STD: float = 0.04  # mm/mm
    # Left eye (OS): vertical -0.05±0.12 mm/mm
    LEFT_EYE_Y_COEFF: float = -0.05  # mm/mm
    LEFT_EYE_Y_STD: float = 0.12  # mm/mm

    # Baseline diameter used in the original study
    BASELINE_DIAMETER: float = 4.75e-3  # m (4.75±0.52 mm at 800 lux)


def pprint_anatomical_defaults() -> None:
    """Print anatomical default values in formatted table."""
    info("Eye Anatomy:")
    info(f"  Axial Length: {EyeAnatomyDefaults.AXIAL_LENGTH * 1000:.2f} mm")
    info(f"  Pupil Radius: {EyeAnatomyDefaults.PUPIL_RADIUS * 1000:.1f} mm")
    info(f"  Aqueous Humor n: {EyeAnatomyDefaults.N_AQUEOUS_HUMOR:.3f}")
    info(f"  Fovea α: {EyeAnatomyDefaults.FOVEA_ALPHA_DEG:.1f}°")
    info(f"  Fovea β: {EyeAnatomyDefaults.FOVEA_BETA_DEG:.1f}°")

    info("Cornea:")
    info(f"  Anterior Radius: {CorneaDefaults.ANTERIOR_RADIUS * 1000:.2f} mm")
    info(f"  Posterior Radius: {CorneaDefaults.POSTERIOR_RADIUS * 1000:.2f} mm")
    info(f"  Refractive Index: {CorneaDefaults.REFRACTIVE_INDEX:.3f}")
    info(f"  Thickness: {CorneaDefaults.THICKNESS_OFFSET * 1000:.2f} mm")


def pprint_hardware_defaults() -> None:
    """Print hardware default values in formatted table."""
    info("Camera:")
    info(f"  Focal Length: {CameraDefaults.FOCAL_LENGTH:.0f} pixels")
    info(f"  Resolution: {CameraDefaults.RESOLUTION_WIDTH}x{CameraDefaults.RESOLUTION_HEIGHT}")
    info(f"  Principal Point: ({CameraDefaults.PRINCIPAL_POINT_X:.0f}, {CameraDefaults.PRINCIPAL_POINT_Y:.0f})")

    info("Pupil:")
    info(f"  Base Radius: {PupilDefaults.BASE_RADIUS * 1000:.1f} mm")
    info(f"  Noncircularity: {PupilDefaults.NONCIRCULARITY:.4f}")
    info(f"  Boundary Points: {PupilDefaults.BOUNDARY_POINTS_REALISTIC}")


def pprint_all_defaults() -> None:
    """Print all default parameters in formatted tables."""
    info("=== PyEtSimul Default Parameters ===")
    pprint_anatomical_defaults()
    info()
    pprint_hardware_defaults()
