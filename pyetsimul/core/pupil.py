"""Pupil model definitions for eye tracking simulation.

Defines abstract and concrete pupil models (elliptical, realistic) for boundary generation and anatomical accuracy.
"""

import math
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, replace

import numpy as np

from pyetsimul.log import info, table

from ..types import Direction3D, Position3D, TransformationMatrix
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
        n: int = PupilDefaults.BOUNDARY_POINTS_ELLIPTICAL,
    ) -> None:
        """Initialize elliptical pupil.

        Args:
            pos_pupil: Center position of pupil
            x_pupil: X-axis radius/direction vector
            y_pupil: Y-axis radius/direction vector
            n: Number of boundary points

        """
        self.pos_pupil = pos_pupil
        self.x_pupil = x_pupil
        self.y_pupil = y_pupil
        self.n = n  # Number of boundary points for this pupil

    @abstractmethod
    def get_boundary_points(self, n: int | None = None) -> np.ndarray:
        """Generate pupil boundary points.

        Args:
            n: Number of boundary points (defaults to self.n if not provided)

        Returns:
            4xn matrix of points on pupil boundary (homogeneous coordinates)

        """

    @abstractmethod
    def get_radii(self) -> tuple[float, float]:
        """Get pupil radii from both axes.

        Returns:
            Tuple of (x_radius, y_radius) in mm

        """

    @abstractmethod
    def set_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (mm)
            y_radius: Pupil radius in Y direction (mm)

        """

    @abstractmethod
    def set_diameter(self, diameter: float) -> None:
        """Set pupil diameter and update geometry.

        Args:
            diameter: Pupil diameter in mm

        """

    @abstractmethod
    def get_diameter(self) -> float:
        """Get pupil diameter.

        Returns:
            Pupil diameter in mm

        """

    @abstractmethod
    def serialize(self) -> dict:
        """Serialize to dictionary representation."""

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

    def get_noncircularity(self) -> float:  # noqa: PLR6301
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
        return f"{self.__class__.__name__}(pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})mm, r=({x_radius:.1f}, {y_radius:.1f})mm)"

    def pprint(self) -> None:
        """Print detailed pupil parameters in a formatted table."""
        pos = self.pos_pupil
        x_radius, y_radius = self.get_radii()

        # Base parameters
        data = [
            ["Pupil type", self.__class__.__name__],
            ["Position (x,y,z) mm", f"({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})"],
            ["X-axis radius (mm)", f"{x_radius:.3f}"],
            ["Y-axis radius (mm)", f"{y_radius:.3f}"],
            ["Boundary points", str(self.n)],
            ["Noncircularity", f"{self.get_noncircularity():.6f}"],
        ]

        # Add RealisticPupil-specific parameters
        if isinstance(self, RealisticPupil):
            data.extend([
                ["Base radius (mm)", f"{self.params.base_radius:.3f}"],
                ["Random seed", str(self.params.random_seed) if self.params.random_seed else "Random"],
            ])

        headers = ["Parameter", "Value"]
        info(f"{self.__class__.__name__} Parameters:")
        table(data, headers=headers, tablefmt="grid")


class EllipticalPupil(Pupil):
    """Elliptical pupil implementation using parametric representation.

    Implements simple elliptical pupil model using parametric formula.
    Uses cos(α)*x_pupil + sin(α)*y_pupil for boundary generation.  # noqa: RUF002

    Args:
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
        n: Number of boundary points
        aspect_ratio: Pupil ellipse major/minor ratio; 1.0 is a circle.
        aspect_angle: Major-axis orientation in radians; 0 aligns the major axis with +x.

    """

    def __init__(
        self,
        pos_pupil: Position3D,
        x_pupil: Direction3D,
        y_pupil: Direction3D,
        n: int = PupilDefaults.BOUNDARY_POINTS_ELLIPTICAL,
        aspect_ratio: float = 1.0,
        aspect_angle: float = 0.0,
    ) -> None:
        """Initialize elliptical pupil with an optional elliptical aspect ratio."""
        super().__init__(pos_pupil, x_pupil, y_pupil, n)
        self.aspect_ratio = aspect_ratio  # major/minor; 1.0 = circle
        self.aspect_angle = aspect_angle  # major-axis orientation (rad); 0 = +x (horizontal)

    def get_boundary_points(self, n: int | None = None) -> np.ndarray:
        """Generate elliptical pupil boundary points using parametric representation.

        Args:
            n: Number of boundary points (defaults to self.n if not provided)

        Returns:
            4xn matrix of points on pupil boundary (homogeneous coordinates)

        """
        n_points = self.n if n is None else n
        alpha = 2 * np.pi * np.arange(n_points) / n_points

        # Convert to homogeneous arrays for computation
        pos_homogeneous = np.array(self.pos_pupil).reshape(-1, 1)
        x_homogeneous = np.array(self.x_pupil).reshape(-1, 1)
        y_homogeneous = np.array(self.y_pupil).reshape(-1, 1)

        # Parametric pupil boundary: pos_pupil + cos(α)*x_pupil + sin(α)*y_pupil
        pupil_points = (
            np.tile(pos_homogeneous, (1, n_points))
            + x_homogeneous @ np.cos(alpha).reshape(1, -1)
            + y_homogeneous @ np.sin(alpha).reshape(1, -1)
        )

        return pupil_points

    def get_radii(self) -> tuple[float, float]:
        """Get pupil radii from both axes.

        Returns:
            Tuple of (x_radius, y_radius) in mm

        """
        x_radius = self.x_pupil.magnitude()
        y_radius = self.y_pupil.magnitude()
        return x_radius, y_radius

    def set_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (mm)
            y_radius: Pupil radius in Y direction (mm)

        """
        self.x_pupil = Direction3D(x_radius, 0, 0)
        self.y_pupil = Direction3D(0, y_radius, 0)

    def set_diameter(self, diameter: float) -> None:
        """Set pupil diameter, applying the elliptical aspect ratio and orientation.

        Args:
            diameter: Pupil diameter in mm

        """
        radius = diameter / 2.0
        # Equal-area semi-axes: major = r*sqrt(A) along aspect_angle, minor = r/sqrt(A)
        # perpendicular. aspect_ratio == 1 reduces to the circle (the eye-tracking default).
        major = radius * math.sqrt(self.aspect_ratio)
        minor = radius / math.sqrt(self.aspect_ratio)
        cos_a, sin_a = math.cos(self.aspect_angle), math.sin(self.aspect_angle)
        self.x_pupil = Direction3D(major * cos_a, major * sin_a, 0.0)
        self.y_pupil = Direction3D(-minor * sin_a, minor * cos_a, 0.0)

    def get_diameter(self) -> float:
        """Get pupil diameter.

        For elliptical pupils, returns average diameter.

        Returns:
            Pupil diameter in mm

        """
        x_radius, y_radius = self.get_radii()
        return x_radius + y_radius

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "pos_pupil": self.pos_pupil.serialize(),
            "x_pupil": self.x_pupil.serialize(),
            "y_pupil": self.y_pupil.serialize(),
            "n": self.n,
            "aspect_ratio": self.aspect_ratio,
            "aspect_angle": self.aspect_angle,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "EllipticalPupil":
        """Deserialize from dictionary representation."""
        return cls(
            pos_pupil=Position3D.deserialize(data["pos_pupil"]),
            x_pupil=Direction3D.deserialize(data["x_pupil"]),
            y_pupil=Direction3D.deserialize(data["y_pupil"]),
            n=data["n"],
            aspect_ratio=data.get("aspect_ratio", 1.0),
            aspect_angle=data.get("aspect_angle", 0.0),
        )


# A pupil at least this wide is in the dilated state. The value lies between the mean diameters that
# Wyatt (1995) measured in light (3.09 mm) and in darkness (4.93 mm).
DILATED_FROM_DIAMETER_MM = 4.0
# Share of the squared noncircularity carried by the elliptical (second) harmonic (Wyatt 1995).
ELLIPSE_CONTRIBUTION_DILATED = 0.596
ELLIPSE_CONTRIBUTION_CONSTRICTED = 0.477
# Von Mises concentration of the major-axis orientation about its mean: vertical for the dilated pupil,
# horizontal for the constricted one. This value gives a spread of about 35 degrees.
_MAJOR_AXIS_CONCENTRATION = 3.0
# Decay of the amplitude weights from one higher harmonic to the next.
_HARMONIC_WEIGHT_DECAY = 0.7
# Increase of the noncircularity per decade of age.
_NONCIRCULARITY_PER_DECADE = 0.0015


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

    Based on: Wyatt, H.J. (1995). "The Form of the Human Pupil."
    Vision Research, 35(14), 2021-2036.

    Attributes:
        noncircularity: Measure of deviation from circularity (0 = perfect circle)
        ellipse_contribution: Share of the squared noncircularity carried by the elliptical component, in
            [0, 1]. None uses the shares of the paper: 0.596 for a dilated and 0.477 for a constricted pupil.
        major_axis_angle: Orientation of the ellipse major axis in radians (0 = vertical), used at every
            size. None draws one orientation for the dilated pupil (about vertical) and one for the
            constricted pupil (about horizontal), as the paper found.
        n_harmonics: Number of Fourier harmonics to include in shape generation
        age: Subject age in years (affects noncircularity)
        random_seed: Seed of the pupil's shape signature (None for a different signature per pupil)

    """

    noncircularity: float = PupilDefaults.NONCIRCULARITY
    ellipse_contribution: float | None = None
    major_axis_angle: float | None = None
    n_harmonics: int = PupilDefaults.N_HARMONICS
    age: float = PupilDefaults.REFERENCE_AGE
    random_seed: int | None = None

    def __post_init__(self) -> None:
        """Reject an elliptical share outside [0, 1]."""
        if self.ellipse_contribution is not None and not 0.0 <= self.ellipse_contribution <= 1.0:
            raise ValueError(f"ellipse_contribution must be in [0, 1], got {self.ellipse_contribution}")


class RealisticPupil(Pupil):
    """Realistic human pupil implementation using Fourier series representation.

    Generates non-circular human pupil boundaries based on Wyatt (1995) analysis.
    Implements Fourier series: R(θ) = r_ave + Σ r_n cos(n(θ - φ_n)).
    Uses paper's noncircularity formula: NC² = (1/2) Σ(r_n/r_ave)² for n=2 to N.

    Each pupil has a shape signature that is drawn once, when the pupil is created, and kept at every size:
    the phases of the higher harmonics, and the major-axis orientation of its dilated and of its constricted
    state. The paper found the shape of a pupil stable over time, similar in light and darkness apart from
    the orientation of the ellipse, and different between subjects. With ``random_seed`` the signature is the
    same in every run; without it every pupil has its own.

    Based on: Wyatt, H.J. (1995). "The Form of the Human Pupil."
    Vision Research, 35(14), 2021-2036.

    Args:
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
        params: Parameters for realistic pupil shape generation
        n: Number of boundary points

    """

    def __init__(
        self,
        pos_pupil: Position3D,
        x_pupil: Direction3D,
        y_pupil: Direction3D,
        params: RealisticPupilParams | None = None,
        n: int = PupilDefaults.BOUNDARY_POINTS_REALISTIC,
    ) -> None:
        """Initialize realistic pupil.

        Args:
            pos_pupil: Center position
            x_pupil: Vector defining X-axis radius/direction
            y_pupil: Vector defining Y-axis radius/direction
            params: Parameters for realistic pupil shape generation
            n: Number of boundary points

        """
        super().__init__(pos_pupil, x_pupil, y_pupil, n)
        # A copy, so a later change to the caller's object does not reach this pupil.
        self.params = RealisticPupilParams() if params is None else replace(params)

        # The shape signature, drawn once in this order.
        rng = np.random.default_rng(self.params.random_seed)
        self._phases = {n_harmonic: rng.uniform(0, 2 * np.pi) for n_harmonic in range(3, self.params.n_harmonics + 1)}
        # Major axis about vertical (0) for the dilated pupil, about horizontal (+/-90 deg) for the constricted.
        self._dilated_axis_angle = float(rng.vonmises(0.0, _MAJOR_AXIS_CONCENTRATION))
        horizontal = rng.choice([np.pi / 2, -np.pi / 2])
        self._constricted_axis_angle = float(rng.vonmises(horizontal, _MAJOR_AXIS_CONCENTRATION))

        self.r2 = 0.0
        self.harmonics: dict[int, dict[str, float]] = {}
        self._set_radius((x_pupil.magnitude() + y_pupil.magnitude()) / 2)

    @property
    def is_dilated(self) -> bool:
        """Whether the pupil is in the dilated state: at least ``DILATED_FROM_DIAMETER_MM`` wide."""
        return 2 * self._radius >= DILATED_FROM_DIAMETER_MM

    @property
    def major_axis_angle(self) -> float:
        """Orientation of the ellipse major axis at the current size, in radians (0 = vertical)."""
        if self.params.major_axis_angle is not None:
            return self.params.major_axis_angle
        return self._dilated_axis_angle if self.is_dilated else self._constricted_axis_angle

    def _set_radius(self, radius: float) -> None:
        """Set the mean radius and the harmonic amplitudes that go with it, by the formula of Wyatt (1995).

        The amplitudes follow from NC² = (1/2) Σ(rₙ/r_ave)² for n = 2 to N: the elliptical harmonic takes its
        share of NC², and the higher harmonics divide the rest with weights that decay with n.
        """
        self._radius = radius
        age_offset = self.params.age - PupilDefaults.REFERENCE_AGE
        noncircularity = self.params.noncircularity + (age_offset / 10) * _NONCIRCULARITY_PER_DECADE

        ellipse_share = self.params.ellipse_contribution
        if ellipse_share is None:
            ellipse_share = ELLIPSE_CONTRIBUTION_DILATED if self.is_dilated else ELLIPSE_CONTRIBUTION_CONSTRICTED

        self.r2 = np.sqrt(2 * ellipse_share * noncircularity**2) * radius

        self.harmonics = {}
        if self._phases:
            weights = np.array([np.exp(-(n_harmonic - 2) * _HARMONIC_WEIGHT_DECAY) for n_harmonic in self._phases])
            weights /= np.sum(weights)
            remaining_nc_squared = noncircularity**2 * (1 - ellipse_share)
            for weight, (n_harmonic, phase) in zip(weights, self._phases.items(), strict=True):
                amplitude = np.sqrt(2 * remaining_nc_squared * weight) * radius
                self.harmonics[n_harmonic] = {"amplitude": amplitude, "phase": phase}

        # The elliptical parameters of the base class hold the mean radius.
        self.x_pupil = Direction3D(radius, 0, 0)
        self.y_pupil = Direction3D(0, radius, 0)

    def set_diameter(self, diameter: float) -> None:
        """Set pupil diameter. The shape signature stays; the state follows the size.

        Args:
            diameter: Pupil diameter in mm

        """
        self._set_radius(diameter / 2)

    def get_boundary_points(self, n: int | None = None) -> np.ndarray:
        """Generate realistic pupil boundary points using Fourier series.

        Args:
            n: Number of boundary points (defaults to self.n if not provided)

        Returns:
            4xn matrix of points on pupil boundary (homogeneous coordinates)

        """
        n_points = self.n if n is None else n
        theta = np.linspace(0, 2 * np.pi, n_points, endpoint=False)

        # Start with average radius (in mm)
        radius = np.full_like(theta, self._radius)

        # Add 2nd harmonic (elliptical component)
        radius += self.r2 * np.cos(2 * (theta - self.major_axis_angle))

        # Add higher harmonics for individual variation
        for harmonic_n, harmonic in self.harmonics.items():
            radius += harmonic["amplitude"] * np.cos(harmonic_n * (theta - harmonic["phase"]))

        # Convert to Cartesian coordinates
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)

        # Create 4xn homogeneous coordinate matrix centered at pupil position
        pupil_points = np.zeros((4, n_points))
        pupil_points[0, :] = self.pos_pupil.x + x
        pupil_points[1, :] = self.pos_pupil.y + y
        pupil_points[2, :] = self.pos_pupil.z
        pupil_points[3, :] = 1.0

        return pupil_points

    def get_radii(self) -> tuple[float, float]:
        """Get pupil radii from both axes.

        For realistic pupil, both are the mean radius.

        Returns:
            Tuple of (x_radius, y_radius) in mm

        """
        return self._radius, self._radius

    def set_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update geometry.

        Args:
            x_radius: Pupil radius in X direction (mm)
            y_radius: Pupil radius in Y direction (mm)

        """
        # For realistic pupil, the mean of the two radii is the mean radius
        self._set_radius((x_radius + y_radius) / 2)

    def get_diameter(self) -> float:
        """Get pupil diameter.

        Returns:
            Pupil diameter in mm (twice the mean radius)

        """
        return self._radius * 2

    def get_noncircularity(self) -> float:
        """Calculate actual noncircularity using Wyatt (1995) exact formula.

        Returns:
            Noncircularity value using paper's formula: NC² = (1/2) * Σ(rₙ/r_ave)² for n=2 to N

        """
        # Paper's formula: NC² = (1/2) * Σ(rₙ/r_ave)² for n=2 to N
        # Where rₙ are the harmonic amplitudes
        nc_squared = 0.5 * (self.r2 / self._radius) ** 2
        for harmonic in self.harmonics.values():
            nc_squared += 0.5 * (harmonic["amplitude"] / self._radius) ** 2

        return np.sqrt(nc_squared)

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "pos_pupil": self.pos_pupil.serialize(),
            "x_pupil": self.x_pupil.serialize(),
            "y_pupil": self.y_pupil.serialize(),
            "params": asdict(self.params),
            "n": self.n,
            "phases": {str(n_harmonic): phase for n_harmonic, phase in self._phases.items()},
            "dilated_axis_angle": self._dilated_axis_angle,
            "constricted_axis_angle": self._constricted_axis_angle,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "RealisticPupil":
        """Deserialize from dictionary representation."""
        pupil = cls(
            pos_pupil=Position3D.deserialize(data["pos_pupil"]),
            x_pupil=Direction3D.deserialize(data["x_pupil"]),
            y_pupil=Direction3D.deserialize(data["y_pupil"]),
            params=RealisticPupilParams(**data["params"]),
            n=data["n"],
        )

        # The saved shape signature replaces the one drawn for the new pupil.
        pupil._phases = {int(n_harmonic): phase for n_harmonic, phase in data["phases"].items()}
        pupil._dilated_axis_angle = data["dilated_axis_angle"]
        pupil._constricted_axis_angle = data["constricted_axis_angle"]
        pupil._set_radius(pupil._radius)

        return pupil


class ContourPupil(Pupil):
    """Pupil with an explicit, arbitrary boundary -- e.g. a measured pupil contour.

    The boundary is stored as 2-D offsets from the pupil centre in the pupil plane, normalised to unit
    mean radius; the diameter then scales them. Unlike EllipticalPupil (parametric) and RealisticPupil
    (Fourier), this accepts any closed outline, so a measured pupil boundary can be used directly.

    Args:
        pos_pupil: Centre position.
        contour: (N, 2) boundary points in the pupil plane (any scale/offset; recentred + normalised here).
        n: Boundary-point count (defaults to len(contour)).

    """

    def __init__(
        self,
        pos_pupil: Position3D,
        contour: np.ndarray,
        n: int | None = None,
    ) -> None:
        """Initialise from an explicit boundary; the contour is recentred and normalised to unit mean radius."""
        # A copy, because the recentring below is done in place and must not reach the caller's array.
        pts = np.array(contour, dtype=float).reshape(-1, 2)
        pts -= pts.mean(axis=0)
        mean_radius = float(np.hypot(pts[:, 0], pts[:, 1]).mean())
        self._unit = pts / mean_radius  # shape at unit mean radius
        self._radius = mean_radius
        super().__init__(
            pos_pupil,
            Direction3D(mean_radius, 0.0, 0.0),
            Direction3D(0.0, mean_radius, 0.0),
            len(pts) if n is None else n,
        )

    def get_boundary_points(self, n: int | None = None) -> np.ndarray:  # noqa: ARG002
        """4xN homogeneous boundary points (the stored contour at the current diameter); ``n`` is ignored."""
        pts = self._unit * self._radius
        out = np.ones((4, pts.shape[0]))
        out[0, :] = self.pos_pupil.x + pts[:, 0]
        out[1, :] = self.pos_pupil.y + pts[:, 1]
        out[2, :] = self.pos_pupil.z
        return out

    def get_radii(self) -> tuple[float, float]:
        """Mean radius on both axes (the contour carries the shape, not these)."""
        return self._radius, self._radius

    def set_radii(self, x_radius: float, y_radius: float) -> None:
        """Scale the contour to the given mean radius."""
        self._radius = (x_radius + y_radius) / 2.0

    def set_diameter(self, diameter: float) -> None:
        """Scale the contour to the given mean diameter (shape preserved)."""
        self._radius = diameter / 2.0

    def get_diameter(self) -> float:
        """Mean diameter of the contour."""
        return 2.0 * self._radius

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "pos_pupil": self.pos_pupil.serialize(),
            "contour": (self._unit * self._radius).tolist(),
            "n": self.n,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ContourPupil":
        """Deserialize from dictionary representation."""
        return cls(pos_pupil=Position3D.deserialize(data["pos_pupil"]), contour=np.array(data["contour"]), n=data["n"])


def create_pupil(
    pupil_type: str, pos_pupil: Position3D, x_pupil: Direction3D, y_pupil: Direction3D, **kwargs: object
) -> Pupil:
    """Factory function to create pupil instances.

    Provides unified interface for creating different pupil models.
    Supports elliptical, realistic, and contour pupil geometries.

    Args:
        pupil_type: Type of pupil ("elliptical", "realistic", or "contour")
        pos_pupil: Center position
        x_pupil: Vector defining X-axis radius/direction
        y_pupil: Vector defining Y-axis radius/direction
        **kwargs: Additional parameters for specific pupil types
                 - n: Number of boundary points (default: 100 for elliptical, 360 for realistic)
                 - params: RealisticPupilParams for realistic pupil (including random_seed for deterministic shapes)
                 - contour: (N, 2) boundary points in the pupil plane; required for the contour pupil

    Returns:
        Pupil instance of the requested type

    Raises:
        ValueError: If pupil_type is not supported

    """
    if pupil_type == "elliptical":
        n: int = kwargs.get("n", PupilDefaults.BOUNDARY_POINTS_ELLIPTICAL)
        return EllipticalPupil(pos_pupil, x_pupil, y_pupil, n)
    if pupil_type == "realistic":
        params = kwargs.get("params")
        n: int = kwargs.get("n", PupilDefaults.BOUNDARY_POINTS_REALISTIC)
        return RealisticPupil(pos_pupil, x_pupil, y_pupil, params, n)
    if pupil_type == "contour":
        return ContourPupil(pos_pupil, kwargs["contour"])
    raise ValueError(f"Unsupported pupil type: {pupil_type}. Supported types: 'elliptical', 'realistic', 'contour'.")
