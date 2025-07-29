import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING
from ..geometry import intersections
from ..optics import reflections, refractions

if TYPE_CHECKING:
    from .eye import Eye


@dataclass
class Cornea(ABC):
    """Abstract base class for different corneal models.

    This class defines the common interface that all corneal models must implement,
    ensuring that they can be used interchangeably within the eye model.
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

    @abstractmethod
    def find_reflection(
        self, light_pos: np.ndarray, camera_pos: np.ndarray, eye_transform: np.ndarray
    ) -> Optional[np.ndarray]:
        """Finds position of a glint on the corneal surface."""
        pass

    @abstractmethod
    def find_refraction(
        self,
        camera_pos: np.ndarray,
        object_pos: np.ndarray,
        n_outside: float,
        n_cornea: float,
        eye_transform: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Finds position where refraction occurs on the corneal surface."""
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

    def find_reflection(
        self, light_pos: np.ndarray, camera_pos: np.ndarray, eye_transform: np.ndarray
    ) -> Optional[np.ndarray]:
        """Finds position of a glint on the spherical corneal surface."""
        world_center = eye_transform @ self.center
        return reflections.find_reflection_sphere(light_pos, camera_pos, world_center, self.radius)

    def find_refraction(
        self,
        camera_pos: np.ndarray,
        object_pos: np.ndarray,
        n_outside: float,
        n_cornea: float,
        eye_transform: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Finds position where refraction occurs on the spherical corneal surface."""
        world_center = eye_transform @ self.center
        return refractions.find_refraction_sphere(
            camera_pos, object_pos, world_center, self.radius, n_outside, n_cornea
        )


@dataclass
class ConicCornea(Cornea):
    """Represents a cornea with a conic section surface.

    This model provides a realistic representation of corneal shape using proper
    conic section geometry with asphericity parameter Q-value, replacing the
    basic ellipsoid approximation with mathematically accurate corneal modeling.

    Attributes:
        center (np.ndarray): The 4D homogeneous coordinate of the conic center (typically corneal apex).
        r_apical (float): Apical radius of curvature in meters.
        Q (float): Asphericity parameter (conic constant):
                  - Q = 0: Perfect sphere
                  - Q < 0: Prolate conic (typical cornea, flattens toward periphery)
                  - Q > 0: Oblate conic (steepens toward periphery)
    """

    r_apical: float = 7.98e-3  # Default apical radius of curvature (meters)
    Q: float = -0.18  # Default Q-value for anterior cornea (prolate)

    def __post_init__(self):
        # Validate Q parameter ranges
        if self.Q > 1:
            print(f"Warning: Q = {self.Q} > 1 may represent unusual corneal geometry")

        # Calculate p-value for reference
        p = self.Q + 1
        if p <= 0:
            print(f"Warning: p-value = {p} ≤ 0 may cause numerical issues in conic calculations")

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """Calculates intersection for a conic cornea.

        Returns the intersection point closer to the ray origin.
        """
        pos, _ = intersections.intersect_ray_conic(ray_origin, ray_direction, self.center, self.r_apical, self.Q)
        return pos

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Calculates the normal vector for a conic surface."""
        return intersections.conic_surface_normal(point, self.center, self.r_apical, self.Q)

    def point_within_cornea(self, p: np.ndarray, eye: "Eye") -> bool:
        """Tests whether a point lies within the conic cornea boundaries."""
        # For a conic section, this check uses the axial depth approximation.
        # This is appropriate since the Q-value primarily affects peripheral curvature.
        p_local = np.linalg.solve(eye.trans, p)
        direction = self.center - eye.pos_apex
        diff = p_local - eye.pos_apex
        projection_distance = np.dot(diff[:3], direction[:3]) / np.linalg.norm(direction[:3])
        return projection_distance < eye.depth_cornea

    def find_reflection(
        self, light_pos: np.ndarray, camera_pos: np.ndarray, eye_transform: np.ndarray
    ) -> Optional[np.ndarray]:
        """Finds position of a glint on the conic corneal surface."""
        world_center = eye_transform @ self.center
        return reflections.find_reflection_conic(light_pos, camera_pos, world_center, self.r_apical, self.Q)

    def find_refraction(
        self,
        camera_pos: np.ndarray,
        object_pos: np.ndarray,
        n_outside: float,
        n_cornea: float,
        eye_transform: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Finds position where refraction occurs on the conic corneal surface."""
        world_center = eye_transform @ self.center
        return refractions.find_refraction_conic(
            camera_pos, object_pos, world_center, self.r_apical, self.Q, n_outside, n_cornea
        )


def create_cornea(cornea_model_type: str, center: np.ndarray, **kwargs) -> Cornea:
    """Factory function to create a cornea object of a specified type.

    Args:
        cornea_model_type (str): The type of cornea model to create.
                                 Supported types: "spherical", "conic".
        center (np.ndarray): The center of the cornea.
        **kwargs: Additional parameters required for the specific cornea model.
                  For "spherical": radius (float)
                  For "conic": r_apical (float), Q (float, optional)

    Returns:
        Cornea: An instance of the specified Cornea subclass.

    Raises:
        ValueError: If an unsupported cornea_model_type is provided.
    """
    if cornea_model_type == "spherical":
        if "radius" not in kwargs:
            raise ValueError("'radius' is required for spherical cornea model.")
        return SphericalCornea(center=center, radius=kwargs["radius"])
    elif cornea_model_type == "conic":
        if "r_apical" not in kwargs:
            raise ValueError("'r_apical' is required for conic cornea model.")
        return ConicCornea(
            center=center,
            r_apical=kwargs["r_apical"],
            Q=kwargs.get("Q", -0.18),  # Use default if not provided
        )
    else:
        raise ValueError(f"Unknown cornea model type: '{cornea_model_type}'")
