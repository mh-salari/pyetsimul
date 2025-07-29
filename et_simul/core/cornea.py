import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from ..geometry import intersections

if TYPE_CHECKING:
    from .eye import Eye


@dataclass
class Cornea(ABC):
    """Abstract base class for different corneal models.

    This class defines the common interface that all corneal models must implement,
    ensuring that they can be used interchangeably within the eye model.

    This class is based on the original MATLAB implementation from the
    et_simul project — © 2008 Martin Böhme, University of Lübeck.
    Python port © 2025 Mohammadhossein Salari.
    Licensed under the GNU GPL v3.0 or later.
    """

    center: Optional[np.ndarray] = None

    @abstractmethod
    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """Calculates the intersection point of a light ray with the cornea."""
        pass

    @abstractmethod
    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Calculates the normal vector at a given point on the cornea's surface."""
        pass

    @abstractmethod
    def point_within_cornea(self, p: np.ndarray, eye: "Eye") -> bool:
        """Tests whether a point lies within the cornea boundaries."""
        pass


@dataclass
class SphericalCornea(Cornea):
    """Represents a cornea with a simple spherical surface.

    Attributes:
        radius (float): The radius of corneal curvature.
        center (np.ndarray): Inherited from parent. If None, will be calculated by Eye based on anatomical scaling.
    """

    radius: float = 7.98e-3  # Default corneal radius

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """Calculates intersection for a spherical cornea.

        Returns the intersection point closer to the ray origin.
        """
        pos, _ = intersections.intersect_ray_sphere(ray_origin, ray_direction, self.center, self.radius)
        return pos

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Calculates the normal vector for a spherical surface."""
        # Ensure we are working with 3D coordinates for vector math
        center_3d = self.center[:3]
        point_3d = point[:3]

        normal = (point_3d - center_3d) / self.radius
        return normal / np.linalg.norm(normal)

    def point_within_cornea(self, p: np.ndarray, eye: "Eye") -> bool:
        """Tests whether a point lies within the spherical cornea boundaries.

        Args:
            p: Point to test (4D homogeneous coordinates)
            eye: Eye object containing transformation matrix, apex position, and cornea depth

        Returns:
            bool: True if point lies within cornea boundaries, False otherwise
        """
        # Transform point to local eye coordinates
        p_local = np.linalg.solve(eye.trans, p)

        # Calculate direction from apex to cornea center
        direction = self.center - eye.pos_apex
        diff = p_local - eye.pos_apex

        # Use dot product for projection distance calculation
        projection_distance = np.dot(diff[:3], direction[:3]) / np.linalg.norm(direction[:3])
        return projection_distance < eye.depth_cornea


@dataclass
class SpheroidCornea(Cornea):
    """Represents a cornea with a prolate spheroid surface.

    This model provides a more realistic representation of the corneal shape.

    Attributes:
        center (np.ndarray): The 4D homogeneous coordinate of the spheroid center.
        a (float): Semi-axis length (x-axis).
        b (float): Semi-axis length (y-axis).
        c (float): Semi-axis length (z-axis, optical axis).
        Q (float): Asphericity parameter (conic constant). Q < 0 for prolate spheroid.
    """

    a: float = 7.98e-3  # Default semi-axis length (x-axis)
    b: float = 7.98e-3  # Default semi-axis length (y-axis)
    c: float = 7.98e-3  # Default semi-axis length (z-axis, optical axis)
    Q: float = -0.18  # Default Q-value for anterior cornea

    def __post_init__(self):
        if not (self.a == self.b and self.a != self.c):
            # This can be a warning or an error depending on desired strictness
            print(f"Warning: For a prolate spheroid, expected a = b ≠ c. Got a={self.a}, b={self.b}, c={self.c}")

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """Calculates intersection for a spheroid cornea.

        Returns the intersection point closer to the ray origin.
        """
        pos, _ = intersections.intersect_ray_spheroid(ray_origin, ray_direction, self.center, self.a, self.b, self.c)
        return pos

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Calculates the normal vector for a spheroid surface."""
        return intersections.spheroid_surface_normal(point, self.center, self.a, self.b, self.c)

    def point_within_cornea(self, p: np.ndarray, eye: "Eye") -> bool:
        """Tests whether a point lies within the spheroid cornea boundaries."""
        # For a spheroid, this check is more complex than for a sphere.
        # A common simplification is to still use the axial depth.
        p_local = np.linalg.solve(eye.trans, p)
        direction = self.center - eye.pos_apex
        diff = p_local - eye.pos_apex
        projection_distance = np.dot(diff[:3], direction[:3]) / np.linalg.norm(direction[:3])
        return projection_distance < eye.depth_cornea


def create_cornea(cornea_model_type: str, center: np.ndarray, **kwargs) -> Cornea:
    """Factory function to create a cornea object of a specified type.

    Args:
        cornea_model_type (str): The type of cornea model to create.
                                 Supported types: "spherical", "spheroid".
        center (np.ndarray): The center of the cornea.
        **kwargs: Additional parameters required for the specific cornea model.
                  For "spherical": radius (float)
                  For "spheroid": a, b, c (float), Q (float, optional)

    Returns:
        Cornea: An instance of the specified Cornea subclass.

    Raises:
        ValueError: If an unsupported cornea_model_type is provided.
    """
    if cornea_model_type == "spherical":
        if "radius" not in kwargs:
            raise ValueError("'radius' is required for spherical cornea model.")
        return SphericalCornea(center=center, radius=kwargs["radius"])
    elif cornea_model_type == "spheroid":
        required_keys = ["a", "b", "c"]
        if not all(key in kwargs for key in required_keys):
            raise ValueError("'a', 'b', and 'c' are required for spheroid cornea model.")
        return SpheroidCornea(
            center=center,
            a=kwargs["a"],
            b=kwargs["b"],
            c=kwargs["c"],
            Q=kwargs.get("Q", -0.18),  # Use default if not provided
        )
    else:
        raise ValueError(f"Unknown cornea model type: '{cornea_model_type}'")
