"""Pupil model definitions for eye tracking simulation.

Defines abstract and concrete pupil models (elliptical, realistic) for boundary generation and anatomical accuracy.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from ..types import Position3D, Direction3D, TransformationMatrix


class Pupil(ABC):
    """Abstract base class for pupil representations.

    Defines common interface for different pupil models (elliptical, realistic, etc.).
    Provides unified interface for boundary point generation and radius management.

    Args:
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
    """

    def __init__(self, pos_pupil: Position3D, x_pupil: Direction3D, y_pupil: Direction3D, N: int = 100):
        self.pos_pupil = pos_pupil
        self.x_pupil = x_pupil
        self.y_pupil = y_pupil
        self.N = N  # Number of boundary points for this pupil

    @abstractmethod
    def get_boundary_points(self, N: Optional[int] = None) -> np.ndarray:
        """Generate pupil boundary points.

        Args:
            N: Number of boundary points (defaults to self.N if not provided)

        Returns:
            4×N matrix of points on pupil boundary (homogeneous coordinates)
        """
        pass

    @abstractmethod
    def get_radii(self) -> tuple[float, float]:
        """Get pupil radii from both axes.

        Returns:
            Tuple of (x_radius, y_radius) in meters
        """
        pass

    @abstractmethod
    def set_radii(self, x_radius: float = None, y_radius: float = None) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)
        """
        pass

    def get_center_world_coords(self, eye_transform: TransformationMatrix) -> Position3D:
        """Get pupil center in world coordinates.

        Transforms pupil center from eye coordinates to world coordinates.

        Args:
            eye_transform: 4x4 transformation matrix from eye to world coordinates

        Returns:
            Pupil center position in world coordinates
        """
        world_homogeneous = eye_transform @ np.array(self.pos_pupil)
        return Position3D.from_array(world_homogeneous)

    def get_noncircularity(self) -> float:
        """Calculate noncircularity.

        Default implementation returns 0.0 (perfect circle).
        Subclasses can override for more sophisticated calculations.

        Returns:
            Noncircularity value (0.0 for perfect circle)
        """
        return 0.0


class EllipticalPupil(Pupil):
    """Elliptical pupil implementation using parametric representation.

    Implements simple elliptical pupil model using parametric formula.
    Uses cos(α)*x_pupil + sin(α)*y_pupil for boundary generation.

    Args:
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
    """

    def get_boundary_points(self, N: Optional[int] = None) -> np.ndarray:
        """Generate elliptical pupil boundary points using parametric representation.

        Args:
            N: Number of boundary points (defaults to self.N if not provided)

        Returns:
            4×N matrix of points on pupil boundary (homogeneous coordinates)
        """
        if N is None:
            N = self.N
        alpha = 2 * np.pi * np.arange(N) / N

        # Convert to homogeneous arrays for computation
        pos_homogeneous = np.array(self.pos_pupil).reshape(-1, 1)
        x_homogeneous = np.array(self.x_pupil).reshape(-1, 1)
        y_homogeneous = np.array(self.y_pupil).reshape(-1, 1)

        # Parametric pupil boundary: pos_pupil + cos(α)*x_pupil + sin(α)*y_pupil
        pupil_points = (
            np.tile(pos_homogeneous, (1, N))
            + x_homogeneous @ np.cos(alpha).reshape(1, -1)
            + y_homogeneous @ np.sin(alpha).reshape(1, -1)
        )

        return pupil_points

    def get_radii(self) -> tuple[float, float]:
        """Get pupil radii from both axes.

        Returns:
            Tuple of (x_radius, y_radius) in meters
        """
        x_radius = self.x_pupil.magnitude()
        y_radius = self.y_pupil.magnitude()
        return x_radius, y_radius

    def set_radii(self, x_radius: float = None, y_radius: float = None) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)

        Raises:
            ValueError: If both radii are None
        """
        if x_radius is None and y_radius is None:
            raise ValueError("At least one radius must be specified")

        if x_radius is not None:
            self.x_pupil = Direction3D(x_radius, 0, 0)
        if y_radius is not None:
            self.y_pupil = Direction3D(0, y_radius, 0)


@dataclass
class RealisticPupilParams:
    """Parameters for realistic human pupil shape generation.

    Contains parameters for generating non-circular human pupil shapes based on Wyatt (1995).
    Implements Fourier series representation of pupil boundary shapes.

    Key findings from Wyatt (1995):
    - No truly circular pupils exist in humans
    - Average noncircularity: 0.0166 in both light and dark conditions
    - Elliptical component contributes ~50% of shape deviation
    - Individual shape signatures remain stable over time
    - Pupils are consistently displaced nasal and superior to limbus center

    Based on: Wyatt, H.J. (1995). "The Form of the Human Pupil."
    Vision Research, 35(14), 2021-2036.

    Attributes:
        base_radius: Average pupil radius in mm
        noncircularity: Measure of deviation from circularity (0 = perfect circle)
        ellipse_contribution: Fraction of noncircularity from elliptical component
        major_axis_angle: Orientation of ellipse major axis in radians (0=vertical)
        pupil_offset_from_limbus: (nasal, superior) displacement from limbus center in meters
        n_harmonics: Number of Fourier harmonics to include in shape generation
        age: Subject age in years (affects noncircularity and pupil size)
    """

    base_radius: float = 2.5  # mm, average radius
    noncircularity: float = 0.0166  # typical human noncircularity
    ellipse_contribution: float = 0.5  # fraction of noncircularity from ellipse
    major_axis_angle: float = 0.0  # radians, 0=vertical, π/2=horizontal
    pupil_offset_from_limbus: Tuple[float, float] = (0.27e-3, 0.20e-3)  # (nasal, superior) in meters
    n_harmonics: int = 6  # number of harmonics to include
    age: float = 35.8  # age in years (study mean from Wyatt 1995)


class RealisticPupil(Pupil):
    """Realistic human pupil implementation using Fourier series representation.

    Generates non-circular human pupil boundaries based on Wyatt (1995) analysis.
    Implements Fourier series: R(θ) = r_ave + Σ r_n cos(n(θ - φ_n)).
    Uses paper's noncircularity formula: NC² = (1/2) Σ(r_n/r_ave)² for n=2 to N.

    Based on: Wyatt, H.J. (1995). "The Form of the Human Pupil."
    Vision Research, 35(14), 2021-2036.

    Args:
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
        params: Parameters for realistic pupil shape generation
    """

    def __init__(
        self,
        pos_pupil: Position3D,
        x_pupil: Direction3D,
        y_pupil: Direction3D,
        params: Optional[RealisticPupilParams] = None,
        N: int = 360,
    ):
        super().__init__(pos_pupil, x_pupil, y_pupil, N)
        self.params = params or RealisticPupilParams()

        # Extract current pupil size from elliptical parameters
        current_radius_x = x_pupil.magnitude()
        current_radius_y = y_pupil.magnitude()
        avg_radius_m = (current_radius_x + current_radius_y) / 2
        diameter_mm = avg_radius_m * 2 * 1000  # Convert to mm

        # Initialize realistic pupil
        self.params.base_radius = diameter_mm / 2

        # Initialize attributes that will be set dynamically
        self.r2 = None
        self.harmonics = {}

        # Set dilated/constricted condition and orientation based on pupil size
        self._update_condition_and_orientation(diameter_mm)

        self._generate_harmonics()

    def _generate_harmonics(self):
        """Generate harmonic amplitudes for realistic pupil shape using Wyatt (1995) formula."""
        # Apply age effects to base parameters
        age_offset = self.params.age - 35.8  # offset from study mean
        noncircularity_age_adjusted = self.params.noncircularity + (age_offset / 10) * 0.0015

        # Get lighting-dependent ellipse contribution
        ellipse_contrib = 0.596 if self._is_dilated else 0.477  # From paper

        # Use paper's formula: NC² = (1/2) Σ(rₙ/r_ave)² for n=2 to N
        # We need to work backwards from target NC to get harmonic amplitudes

        # 2nd harmonic (elliptical component)
        ellipse_nc_contribution = ellipse_contrib * (noncircularity_age_adjusted**2)
        self.r2 = np.sqrt(2 * ellipse_nc_contribution) * self.params.base_radius

        # Higher harmonics - distribute remaining noncircularity
        remaining_nc_squared = (noncircularity_age_adjusted**2) * (1 - ellipse_contrib)
        n_higher_harmonics = self.params.n_harmonics - 2  # excluding 2nd harmonic

        self.harmonics = {}
        if n_higher_harmonics > 0:
            # Distribute remaining NC² among higher harmonics with exponential decay
            weights = np.array([np.exp(-(n - 2) * 0.7) for n in range(3, self.params.n_harmonics + 1)])
            weights = weights / np.sum(weights)  # Normalize

            for i, n in enumerate(range(3, self.params.n_harmonics + 1)):
                # Each harmonic gets a fraction of remaining NC²
                harmonic_nc_squared = remaining_nc_squared * weights[i]
                amplitude = np.sqrt(2 * harmonic_nc_squared) * self.params.base_radius

                self.harmonics[n] = {
                    "amplitude": amplitude,
                    "phase": np.random.uniform(0, 2 * np.pi),  # Individual variation
                }

    def _update_condition_and_orientation(self, diameter_mm: float):
        """Update dilated/constricted condition and major axis orientation based on pupil size.

        Args:
            diameter_mm: Pupil diameter in millimeters
        """
        # Determine orientation based on size (using paper's reference values)
        # Large pupils (dilated) tend to have vertical ellipse orientation
        # Small pupils (constricted) tend to have horizontal ellipse orientation
        if diameter_mm >= 4.0:  # Closer to dilated condition size (4.93mm)
            self._is_dilated = True
            # Major axis orientation: clusters around vertical (0°)
            concentration = 3.0  # Controls spread (~±30° for this value)
            self.params.major_axis_angle = np.random.vonmises(0, concentration)
        else:  # Closer to constricted condition size (3.09mm)
            self._is_dilated = False
            # Major axis orientation: clusters around horizontal (±90°)
            base_angle = np.random.choice([np.pi / 2, -np.pi / 2])
            concentration = 3.0
            self.params.major_axis_angle = np.random.vonmises(base_angle, concentration)

    def set_diameter(self, diameter_mm: float):
        """Set pupil diameter and automatically determine shape characteristics.

        Args:
            diameter_mm: Pupil diameter in millimeters
        """
        self.params.base_radius = diameter_mm / 2

        # Update condition and orientation based on new size
        self._update_condition_and_orientation(diameter_mm)

        # Regenerate harmonics with new conditions
        self._generate_harmonics()

        # Update elliptical parameters to match average radius for compatibility
        avg_radius_m = (diameter_mm / 2) * 1e-3  # Convert mm to meters
        self.x_pupil = Direction3D(avg_radius_m, 0, 0)
        self.y_pupil = Direction3D(0, avg_radius_m, 0)

    def get_boundary_points(self, N: Optional[int] = None) -> np.ndarray:
        """Generate realistic pupil boundary points using Fourier series.

        Args:
            N: Number of boundary points (defaults to self.N if not provided)

        Returns:
            4×N matrix of points on pupil boundary (homogeneous coordinates)
        """
        if N is None:
            N = self.N
        theta = np.linspace(0, 2 * np.pi, N, endpoint=False)

        # Start with average radius (in mm)
        radius_mm = np.full_like(theta, self.params.base_radius)

        # Add 2nd harmonic (elliptical component)
        if self.r2 is not None:
            radius_mm += self.r2 * np.cos(2 * (theta - self.params.major_axis_angle))

        # Add higher harmonics for individual variation
        if self.harmonics:
            for n, harmonic in self.harmonics.items():
                radius_mm += harmonic["amplitude"] * np.cos(n * (theta - harmonic["phase"]))

        # Convert to Cartesian coordinates in meters
        radius_m = radius_mm * 1e-3  # Convert mm to meters
        x = radius_m * np.cos(theta)
        y = radius_m * np.sin(theta)

        # Create 4×N homogeneous coordinate matrix centered at pupil position
        pupil_points = np.zeros((4, N))
        pupil_points[0, :] = self.pos_pupil.x + x
        pupil_points[1, :] = self.pos_pupil.y + y
        pupil_points[2, :] = self.pos_pupil.z
        pupil_points[3, :] = 1.0

        return pupil_points

    def get_radii(self) -> tuple[float, float]:
        """Get pupil radii from both axes.

        For realistic pupil, returns effective radii based on current parameters.

        Returns:
            Tuple of (x_radius, y_radius) in meters
        """
        avg_radius_m = self.params.base_radius * 1e-3  # Convert mm to meters
        return avg_radius_m, avg_radius_m

    def set_radii(self, x_radius: float = None, y_radius: float = None) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)

        Raises:
            ValueError: If both radii are None
        """
        if x_radius is None and y_radius is None:
            raise ValueError("At least one radius must be specified")

        # For realistic pupil, use average radius and update realistic parameters
        if x_radius is not None and y_radius is not None:
            avg_radius_m = (x_radius + y_radius) / 2
        elif x_radius is not None:
            avg_radius_m = x_radius
        else:
            avg_radius_m = y_radius

        diameter_mm = avg_radius_m * 2 * 1000  # Convert to mm
        self.set_diameter(diameter_mm)

    def get_noncircularity(self) -> float:
        """Calculate actual noncircularity using Wyatt (1995) exact formula.

        Returns:
            Noncircularity value using paper's formula: NC² = (1/2) * Σ(rₙ/r_ave)² for n=2 to N
        """
        # Paper's formula: NC² = (1/2) * Σ(rₙ/r_ave)² for n=2 to N
        # Where rₙ are the harmonic amplitudes
        r_ave = self.params.base_radius
        nc_squared = 0.0

        # Add 2nd harmonic contribution (elliptical)
        if self.r2 is not None:
            nc_squared += 0.5 * (self.r2 / r_ave) ** 2

        # Add higher harmonic contributions
        if self.harmonics:
            for harmonic in self.harmonics.values():
                nc_squared += 0.5 * (harmonic["amplitude"] / r_ave) ** 2

        return np.sqrt(nc_squared)


def create_pupil(
    pupil_type: str, pos_pupil: Position3D, x_pupil: Direction3D, y_pupil: Direction3D, **kwargs
) -> Pupil:
    """Factory function to create pupil instances.

    Provides unified interface for creating different pupil models.
    Supports elliptical and realistic pupil geometries.

    Args:
        pupil_type: Type of pupil ("elliptical" or "realistic")
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
        **kwargs: Additional parameters for specific pupil types
                 - N: Number of boundary points (default: 100 for elliptical, 360 for realistic)
                 - params: RealisticPupilParams for realistic pupil

    Returns:
        Pupil instance of the requested type

    Raises:
        ValueError: If pupil_type is not supported
    """
    if pupil_type == "elliptical":
        N = kwargs.get("N", 20)
        return EllipticalPupil(pos_pupil, x_pupil, y_pupil, N)
    elif pupil_type == "realistic":
        params = kwargs.get("params", None)
        N = kwargs.get("N", 360)
        return RealisticPupil(pos_pupil, x_pupil, y_pupil, params, N)
    else:
        raise ValueError(f"Unsupported pupil type: {pupil_type}. Supported types: 'elliptical', 'realistic'")
