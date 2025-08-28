"""Pupil model definitions for eye tracking simulation.

Defines abstract and concrete pupil models (elliptical, realistic) for boundary generation and anatomical accuracy.
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional
from tabulate import tabulate
from ..types import Position3D, Direction3D, TransformationMatrix
from .default_configs import PupilDefaults


class Pupil(ABC):
    """Abstract base class for pupil representations.

    Defines common interface for different pupil models (elliptical, realistic, etc.).
    Provides unified interface for boundary point generation and radius management.

    Args:
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
    """

    def __init__(
        self,
        pos_pupil: Position3D,
        x_pupil: Direction3D,
        y_pupil: Direction3D,
        N: int = PupilDefaults.BOUNDARY_POINTS_ELLIPTICAL,
    ):
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
    def set_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)
        """
        pass

    @abstractmethod
    def set_diameter(self, diameter: float) -> None:
        """Set pupil diameter and update geometry.

        Args:
            diameter: Pupil diameter in meters
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

    def __str__(self) -> str:
        """Basic string representation of the pupil."""
        pos = self.pos_pupil
        x_radius, y_radius = self.get_radii()
        return f"{self.__class__.__name__}(pos=({pos.x * 1000:.1f}, {pos.y * 1000:.1f}, {pos.z * 1000:.1f})mm, r=({x_radius * 1000:.1f}, {y_radius * 1000:.1f})mm)"

    def pprint(self) -> None:
        """Print detailed pupil parameters in a formatted table."""
        pos = self.pos_pupil
        x_radius, y_radius = self.get_radii()

        # Base parameters
        data = [
            ["Pupil type", self.__class__.__name__],
            ["Position (x,y,z) mm", f"({pos.x * 1000:.3f}, {pos.y * 1000:.3f}, {pos.z * 1000:.3f})"],
            ["X-axis radius (mm)", f"{x_radius * 1000:.3f}"],
            ["Y-axis radius (mm)", f"{y_radius * 1000:.3f}"],
            ["Boundary points", str(self.N)],
            ["Noncircularity", f"{self.get_noncircularity():.6f}"],
        ]

        # Add RealisticPupil-specific parameters
        if isinstance(self, RealisticPupil):
            data.append(["Base radius (mm)", f"{self.params.base_radius * 1000:.3f}"])
            data.append(["Random seed", str(self.params.random_seed) if self.params.random_seed else "Random"])

        headers = ["Parameter", "Value"]
        print(f"{self.__class__.__name__} Parameters:")
        print(tabulate(data, headers=headers, tablefmt="grid"))


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

    def set_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)
        """
        self.x_pupil = Direction3D(x_radius, 0, 0)
        self.y_pupil = Direction3D(0, y_radius, 0)

    def set_diameter(self, diameter: float) -> None:
        """Set pupil diameter and update geometry.

        Args:
            diameter: Pupil diameter in meters
        """
        radius = diameter / 2
        self.set_radii(x_radius=radius, y_radius=radius)

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "pos_pupil": self.pos_pupil.serialize(),
            "x_pupil": self.x_pupil.serialize(),
            "y_pupil": self.y_pupil.serialize(),
            "N": self.N,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "EllipticalPupil":
        """Deserialize from dictionary representation."""
        return cls(
            pos_pupil=Position3D.deserialize(data["pos_pupil"]),
            x_pupil=Direction3D.deserialize(data["x_pupil"]),
            y_pupil=Direction3D.deserialize(data["y_pupil"]),
            N=data["N"],
        )


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
        base_radius: Average pupil radius in meters
        noncircularity: Measure of deviation from circularity (0 = perfect circle)
        ellipse_contribution: Fraction of noncircularity from elliptical component
        major_axis_angle: Orientation of ellipse major axis in radians (0=vertical)
        pupil_offset_from_limbus: (nasal, superior) displacement from limbus center in meters
        n_harmonics: Number of Fourier harmonics to include in shape generation
        age: Subject age in years (affects noncircularity and pupil size)
        random_seed: Random seed for reproducible shape generation (None for random)
    """

    base_radius: float = PupilDefaults.BASE_RADIUS  # meters
    noncircularity: float = PupilDefaults.NONCIRCULARITY
    ellipse_contribution: float = PupilDefaults.ELLIPSE_CONTRIBUTION
    major_axis_angle: float = PupilDefaults.MAJOR_AXIS_ANGLE
    pupil_offset_from_limbus: Tuple[float, float] = PupilDefaults.OFFSET_FROM_LIMBUS
    n_harmonics: int = PupilDefaults.N_HARMONICS
    age: float = PupilDefaults.REFERENCE_AGE
    random_seed: Optional[int] = None  # seed for reproducible random generation (None = random)


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
        N: int = PupilDefaults.BOUNDARY_POINTS_REALISTIC,
    ):
        super().__init__(pos_pupil, x_pupil, y_pupil, N)
        self.params = params or RealisticPupilParams()

        # Extract current pupil size from elliptical parameters
        current_radius_x = x_pupil.magnitude()
        current_radius_y = y_pupil.magnitude()
        avg_radius = (current_radius_x + current_radius_y) / 2

        # Initialize realistic pupil
        self.params.base_radius = avg_radius

        # Initialize attributes that will be set dynamically
        self.r2 = None
        self.harmonics = {}

        # Set dilated/constricted condition and orientation based on pupil size
        diameter = avg_radius * 2
        self._update_condition_and_orientation(diameter)

        self._generate_harmonics()

    def _set_random_seed(self):
        """Set random seed if specified, for reproducible shape generation."""
        if self.params.random_seed is not None:
            np.random.seed(self.params.random_seed)

    def _generate_harmonics(self):
        """Generate harmonic amplitudes for realistic pupil shape using Wyatt (1995) formula."""
        self._set_random_seed()
        # Apply age effects to base parameters
        age_offset = self.params.age - PupilDefaults.REFERENCE_AGE
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

    def _update_condition_and_orientation(self, diameter: float):
        """Update dilated/constricted condition and major axis orientation based on pupil size.

        Args:
            diameter: Pupil diameter in meters
        """
        self._set_random_seed()
        # Determine orientation based on size (using paper's reference values)
        # Large pupils (dilated) tend to have vertical ellipse orientation
        # Small pupils (constricted) tend to have horizontal ellipse orientation
        if diameter >= 0.004:  # Closer to dilated condition size (4.93mm = 0.00493m)
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

    def set_diameter(self, diameter: float):
        """Set pupil diameter and automatically determine shape characteristics.

        Args:
            diameter: Pupil diameter in meters
        """
        self.params.base_radius = diameter / 2

        # Update condition and orientation based on new size
        self._update_condition_and_orientation(diameter)

        # Regenerate harmonics with new conditions
        self._generate_harmonics()

        # Update elliptical parameters to match average radius for compatibility
        radius = diameter / 2
        self.x_pupil = Direction3D(radius, 0, 0)
        self.y_pupil = Direction3D(0, radius, 0)

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

        # Start with average radius (in meters)
        radius = np.full_like(theta, self.params.base_radius)

        # Add 2nd harmonic (elliptical component)
        if self.r2 is not None:
            radius += self.r2 * np.cos(2 * (theta - self.params.major_axis_angle))

        # Add higher harmonics for individual variation
        if self.harmonics:
            for n, harmonic in self.harmonics.items():
                radius += harmonic["amplitude"] * np.cos(n * (theta - harmonic["phase"]))

        # Convert to Cartesian coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

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
        return self.params.base_radius, self.params.base_radius

    def set_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)
        """
        # For realistic pupil, use average radius and update realistic parameters
        avg_radius = (x_radius + y_radius) / 2
        diameter = avg_radius * 2
        self.set_diameter(diameter)

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
                 - params: RealisticPupilParams for realistic pupil (including random_seed for deterministic shapes)

    Returns:
        Pupil instance of the requested type

    Raises:
        ValueError: If pupil_type is not supported
    """
    if pupil_type == "elliptical":
        N = kwargs.get("N", PupilDefaults.BOUNDARY_POINTS_FACTORY)
        return EllipticalPupil(pos_pupil, x_pupil, y_pupil, N)
    elif pupil_type == "realistic":
        params = kwargs.get("params", None)
        N = kwargs.get("N", PupilDefaults.BOUNDARY_POINTS_REALISTIC)
        return RealisticPupil(pos_pupil, x_pupil, y_pupil, params, N)
    else:
        raise ValueError(f"Unsupported pupil type: {pupil_type}. Supported types: 'elliptical', 'realistic'")
