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
    """Represents a cornea with dual spherical surfaces (anterior and posterior).

    Attributes:
        anterior_radius (float): The radius of the anterior (outer) corneal surface.
        posterior_radius (float): The radius of the posterior (inner) corneal surface.
        thickness (float): Central corneal thickness.
        center (np.ndarray): Inherited from parent. If None, will be calculated by Eye based on anatomical scaling.
    """

    anterior_radius: float = 7.98e-3  # Default anterior corneal radius
    refractive_index: float = 1.376  # Refractive index of cornea
    
    # Reference values for scaling (from Boff and Lincoln [1988])
    _posterior_radius_default: float = 6.22e-3  # Default posterior corneal radius
    _thickness_offset_default: float = 1.15e-3  # Default corneal thickness offset

    # Sphere-specific constants from Boff and Lincoln [1988, Section 1.210]
    _r_cornea_default: float = 7.98e-3  # Reference corneal radius for scaling
    _cornea_depth_default: float = 3.54e-3  # Reference corneal depth (apex to limbus)
    _cornea_center_to_rotation_center_default: float = 10.20e-3  # Reference distance

    @property
    def posterior_radius(self) -> float:
        """Calculate the scaled posterior corneal radius.
        
        Returns:
            Posterior corneal radius in meters (scaled from reference)
        """
        scale = self.get_scale_factor()
        return scale * self._posterior_radius_default

    @property
    def thickness_offset(self) -> float:
        """Calculate the scaled thickness offset.
        
        Returns:
            Thickness offset in meters (scaled from reference)
        """
        scale = self.get_scale_factor()
        return scale * self._thickness_offset_default

    @property
    def thickness(self) -> float:
        """Calculate the central corneal thickness based on radii and offset.
        
        For a dual-surface spherical cornea, the central thickness is the distance
        between the anterior and posterior surfaces along the optical axis.
        
        Returns:
            Central corneal thickness in meters
        """
        # Distance between surface centers along optical axis
        center_distance = abs(self.anterior_radius - self.posterior_radius - self.thickness_offset)
        return center_distance

    def get_posterior_center(self) -> np.ndarray:
        """Calculate the center of the posterior surface based on thickness."""
        thickness_term = self.anterior_radius - self.posterior_radius - self.thickness_offset
        return self.center - np.array([0, 0, thickness_term, 0])

    def calculate_center_position(
        self, scale: float, axial_length: float, cornea_center_to_rotation_center: float
    ) -> np.ndarray:
        """Calculate the center position for spherical cornea based on anatomical parameters.

        This implements the original MATLAB/Eye logic for positioning the spherical cornea center.

        Args:
            scale: Scaling factor based on corneal radius
            axial_length: Total axial length of eye (m)
            cornea_center_to_rotation_center: Distance from corneal center to rotation center (m)

        Returns:
            4D homogeneous coordinates of cornea center
        """
        cornea_z_offset = axial_length - 2 * cornea_center_to_rotation_center
        return np.array([0, 0, -scale * cornea_z_offset, 1])

    def get_apex_position(self) -> np.ndarray:
        """Calculate the apex position for spherical cornea.

        For spherical cornea, apex is at center + [0, 0, -radius, 0]

        Returns:
            4D homogeneous coordinates of corneal apex
        """
        return self.center + np.array([0, 0, -self.anterior_radius, 0])

    def get_scale_factor(self) -> float:
        """Calculate the scaling factor for this spherical cornea.

        The scale factor is used to proportionally scale all eye dimensions
        based on how this cornea's radius differs from the reference radius.

        Returns:
            Scale factor (dimensionless)
        """
        return self.anterior_radius / self._r_cornea_default

    def get_corneal_depth(self) -> float:
        """Calculate the scaled corneal depth for this spherical cornea.

        Returns:
            Corneal depth in meters (scaled from reference depth)
        """
        scale = self.get_scale_factor()
        return scale * self._cornea_depth_default

    def setup_eye_geometry(self, axial_length: float) -> dict:
        """Setup all sphere-specific eye geometry parameters.

        This method encapsulates all the sphere-specific scaling logic
        that was previously scattered in the Eye class.

        Args:
            axial_length: Total axial length of the eye (general eye parameter)

        Returns:
            Dictionary containing all calculated geometry parameters
        """
        scale = self.get_scale_factor()

        # Calculate center position if not already set
        if self.center is None:
            self.center = self.calculate_center_position(
                scale, axial_length, self._cornea_center_to_rotation_center_default
            )

        return {
            "scale": scale,
            "corneal_depth": self.get_corneal_depth(),
            "apex_position": self.get_apex_position(),
            "cornea_center_to_rotation_center": self._cornea_center_to_rotation_center_default,
        }

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """Calculates intersection for a spherical cornea.

        Returns the intersection point closer to the ray origin.
        """
        pos, _ = intersections.intersect_ray_sphere(ray_origin, ray_direction, self.center, self.anterior_radius)
        return pos

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Calculates the normal vector for a spherical surface."""
        # Ensure we are working with 3D coordinates for vector math
        center_3d = self.center[:3]
        point_3d = point[:3]

        normal = (point_3d - center_3d) / self.anterior_radius
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
        direction = self.center - eye.cornea.get_apex_position()
        diff = p_local - eye.cornea.get_apex_position()

        # Use dot product for projection distance calculation
        projection_distance = np.dot(diff[:3], direction[:3]) / np.linalg.norm(direction[:3])
        return projection_distance < eye.cornea.get_corneal_depth()

    def find_reflection(
        self, light_pos: np.ndarray, camera_pos: np.ndarray, eye_transform: np.ndarray
    ) -> Optional[np.ndarray]:
        """Finds position of a glint on the spherical corneal surface."""
        world_center = eye_transform @ self.center
        return reflections.find_reflection_sphere(light_pos, camera_pos, world_center, self.anterior_radius)

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
            camera_pos, object_pos, world_center, self.anterior_radius, n_outside, n_cornea
        )


@dataclass
class ConicCornea(Cornea):
    """Represents a cornea with dual conic surfaces (anterior and posterior).

    This model provides a realistic representation of corneal shape using proper
    dual-surface conic section geometry with asphericity parameters for both surfaces.
    Default parameters are based on 30-year-old values from Goncharov & Dainty (2007).

    Attributes:
        center (np.ndarray): The 4D homogeneous coordinate of the conic center (typically corneal apex).
        anterior_radius (float): Anterior surface apical radius of curvature in meters.
        anterior_Q (float): Anterior surface asphericity parameter (conic constant).
        posterior_radius (float): Posterior surface apical radius of curvature in meters.
        posterior_Q (float): Posterior surface asphericity parameter (conic constant).
        thickness (float): Central corneal thickness.
        refractive_index (float): Refractive index of cornea.
        thickness_offset (float): Corneal thickness offset.

        Q-value meanings:
                  - Q = 0: Perfect sphere
                  - Q < 0: Prolate conic (typical cornea, flattens toward periphery)
                  - Q > 0: Oblate conic (steepens toward periphery)
    """

    # Anterior surface (30-year defaults from Goncharov & Dainty 2007)
    anterior_radius: float = 7.76e-3  # Anterior radius
    anterior_Q: float = 0.10  # Anterior Q-value

    # Posterior surface (30-year defaults from Goncharov & Dainty 2007)
    posterior_radius: float = 6.52e-3  # Posterior radius
    posterior_Q: float = 0.30  # Posterior Q-value

    # Corneal properties
    thickness: float = 0.55e-3  # Central corneal thickness
    refractive_index: float = 1.376  # Refractive index of cornea
    thickness_offset: float = 1.15e-3  # Corneal thickness offset

    # Backward compatibility properties
    @property
    def r_apical(self) -> float:
        """Backward compatibility: returns anterior radius."""
        return self.anterior_radius

    @property
    def Q(self) -> float:
        """Backward compatibility: returns anterior Q."""
        return self.anterior_Q

    def get_posterior_center(self) -> np.ndarray:
        """Calculate the center of the posterior surface based on thickness."""
        thickness_term = self.anterior_radius - self.posterior_radius - self.thickness_offset
        return self.center - np.array([0, 0, thickness_term, 0])

    def __post_init__(self):
        # Validate Q parameter ranges for both surfaces
        if self.anterior_Q > 1:
            print(f"Warning: anterior_Q = {self.anterior_Q} > 1 may represent unusual corneal geometry")
        if self.posterior_Q > 1:
            print(f"Warning: posterior_Q = {self.posterior_Q} > 1 may represent unusual corneal geometry")

        # Calculate p-values for reference
        anterior_p = self.anterior_Q + 1
        posterior_p = self.posterior_Q + 1
        if anterior_p <= 0:
            print(f"Warning: anterior p-value = {anterior_p} ≤ 0 may cause numerical issues in conic calculations")
        if posterior_p <= 0:
            print(f"Warning: posterior p-value = {posterior_p} ≤ 0 may cause numerical issues in conic calculations")

    def intersect(self, ray_origin: np.ndarray, ray_direction: np.ndarray) -> Optional[np.ndarray]:
        """Calculates intersection for the anterior conic surface.

        Returns the intersection point closer to the ray origin.
        """
        pos, _ = intersections.intersect_ray_conic(
            ray_origin, ray_direction, self.center, self.anterior_radius, self.anterior_Q
        )
        return pos

    def normal_at(self, point: np.ndarray) -> np.ndarray:
        """Calculates the normal vector for the anterior conic surface."""
        return intersections.conic_surface_normal(point, self.center, self.anterior_radius, self.anterior_Q)

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
        """Finds position of a glint on the anterior conic surface."""
        world_center = eye_transform @ self.center
        return reflections.find_reflection_conic(
            light_pos, camera_pos, world_center, self.anterior_radius, self.anterior_Q
        )

    def find_refraction(
        self,
        camera_pos: np.ndarray,
        object_pos: np.ndarray,
        n_outside: float,
        n_cornea: float,
        eye_transform: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Finds position where refraction occurs on the anterior conic surface."""
        world_center = eye_transform @ self.center
        return refractions.find_refraction_conic(
            camera_pos, object_pos, world_center, self.anterior_radius, self.anterior_Q, n_outside, n_cornea
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
        return SphericalCornea(
            center=center,
            anterior_radius=kwargs.get("anterior_radius", 7.98e-3),
            posterior_radius=kwargs.get("posterior_radius", 6.22e-3),
            thickness=kwargs.get("thickness", 0.55e-3),
            refractive_index=kwargs.get("refractive_index", 1.376),
            thickness_offset=kwargs.get("thickness_offset", 1.15e-3),
        )
    elif cornea_model_type == "conic":
        return ConicCornea(
            center=center,
            anterior_radius=kwargs.get("anterior_radius", 7.76e-3),
            anterior_Q=kwargs.get("anterior_Q", 0.10),
            posterior_radius=kwargs.get("posterior_radius", 6.52e-3),
            posterior_Q=kwargs.get("posterior_Q", 0.30),
            thickness=kwargs.get("thickness", 0.55e-3),
            refractive_index=kwargs.get("refractive_index", 1.376),
            thickness_offset=kwargs.get("thickness_offset", 1.15e-3),
        )
    else:
        raise ValueError(f"Unknown cornea model type: '{cornea_model_type}'")
