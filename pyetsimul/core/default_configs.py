"""Centralized default parameters for PyEtSimul core components.

Defaults for anatomical and hardware parameters.
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any


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
    OFFSET_FROM_LIMBUS: Tuple[float, float] = (0.27e-3, 0.20e-3)  # m
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


def create_myopic_eye_config() -> Dict[str, Any]:
    """Myopic eye configuration."""
    return {"axial_length": 26.5e-3}


def create_hyperopic_eye_config() -> Dict[str, Any]:
    """Hyperopic eye configuration."""
    return {"axial_length": 23.0e-3}


def create_elderly_eye_config() -> Dict[str, Any]:
    """Elderly subject configuration."""
    return {"pupil_radius": 2.5e-3, "fovea_alpha_deg": 5.5}


def create_child_eye_config() -> Dict[str, Any]:
    """Pediatric eye configuration."""
    return {"axial_length": 22.0e-3}


def create_desktop_camera_config() -> Dict[str, Any]:
    """Desktop eye tracker camera configuration."""
    return {"focal_length": 800.0, "resolution": (640, 480)}


def create_lab_camera_config() -> Dict[str, Any]:
    """Research camera configuration."""
    return {"focal_length": 2880.0, "resolution": (1280, 1024)}


def create_mobile_camera_config() -> Dict[str, Any]:
    """Mobile camera configuration."""
    return {"focal_length": 600.0, "resolution": (320, 240)}


def print_anatomical_defaults() -> None:
    """Print anatomical default values."""
    print("Eye Anatomy:")
    print(f"  Axial Length: {EyeAnatomyDefaults.AXIAL_LENGTH * 1000:.2f} mm")
    print(f"  Pupil Radius: {EyeAnatomyDefaults.PUPIL_RADIUS * 1000:.1f} mm")
    print(f"  Aqueous Humor n: {EyeAnatomyDefaults.N_AQUEOUS_HUMOR:.3f}")
    print(f"  Fovea α: {EyeAnatomyDefaults.FOVEA_ALPHA_DEG:.1f}°")
    print(f"  Fovea β: {EyeAnatomyDefaults.FOVEA_BETA_DEG:.1f}°")

    print("Cornea:")
    print(f"  Anterior Radius: {CorneaDefaults.ANTERIOR_RADIUS * 1000:.2f} mm")
    print(f"  Posterior Radius: {CorneaDefaults.POSTERIOR_RADIUS * 1000:.2f} mm")
    print(f"  Refractive Index: {CorneaDefaults.REFRACTIVE_INDEX:.3f}")
    print(f"  Thickness: {CorneaDefaults.THICKNESS_OFFSET * 1000:.2f} mm")


def print_hardware_defaults() -> None:
    """Print hardware default values."""
    print("Camera:")
    print(f"  Focal Length: {CameraDefaults.FOCAL_LENGTH:.0f} pixels")
    print(f"  Resolution: {CameraDefaults.RESOLUTION_WIDTH}×{CameraDefaults.RESOLUTION_HEIGHT}")
    print(f"  Principal Point: ({CameraDefaults.PRINCIPAL_POINT_X:.0f}, {CameraDefaults.PRINCIPAL_POINT_Y:.0f})")

    print("Pupil:")
    print(f"  Base Radius: {PupilDefaults.BASE_RADIUS * 1000:.1f} mm")
    print(f"  Noncircularity: {PupilDefaults.NONCIRCULARITY:.4f}")
    print(f"  Boundary Points: {PupilDefaults.BOUNDARY_POINTS_REALISTIC}")


def print_all_defaults() -> None:
    """Print all default parameters."""
    print("=== PyEtSimul Default Parameters ===")
    print_anatomical_defaults()
    print()
    print_hardware_defaults()
