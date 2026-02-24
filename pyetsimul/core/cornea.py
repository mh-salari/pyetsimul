"""Cornea model definitions for eye tracking simulation.

Defines abstract and concrete cornea models (spherical, conic) for anatomical and optical simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyetsimul.log import info, table, warning

from ..geometry import intersections
from ..optics import reflections, refractions
from ..types import Direction3D, IntersectionResult, Point3D, Position3D, Ray, TransformationMatrix, Vector3D
from .default_configs import CorneaDefaults

if TYPE_CHECKING:
    from .eye import Eye


@dataclass
class Cornea(ABC):
    """Abstract base class for different corneal models.

    Defines common interface for corneal models to ensure interchangeability.
    Provides unified interface for intersection, reflection, and refraction calculations.
    """

    center_init: InitVar[Position3D | None] = None
    _center: Position3D | None = field(default=None, init=False)

    _cornea_depth_default: float = CorneaDefaults.CORNEA_DEPTH
    _cornea_center_to_rotation_center_default: float = CorneaDefaults.CENTER_TO_ROTATION

    def __post_init__(self, center_init: Position3D | None) -> None:
        """Initialize cornea center if provided."""
        if center_init is not None:
            self._center = center_init

    @property
    def center(self) -> Position3D:
        """Get the cornea center position."""
        if self._center is None:
            raise ValueError("Center has not been initialized. Did you call setup_eye_geometry() already?")
        return self._center

    @center.setter
    def center(self, value: Position3D) -> None:
        self._center = value

    @property
    def cornea_center_to_rotation_center_default(self) -> float:
        """Get the default cornea center to rotation center distance."""
        return self._cornea_center_to_rotation_center_default

    @property
    @abstractmethod
    def cornea_type(self) -> str:
        """Return the type name of this cornea model."""

    @abstractmethod
    def intersect(self, ray: Ray) -> IntersectionResult | None:
        """Calculates the intersection point of a light ray with the cornea."""

    @abstractmethod
    def normal_at(self, point: Point3D) -> Direction3D:
        """Calculates the normal vector at a given point on the cornea's surface."""

    def point_within_cornea(self, p: Position3D, eye: "Eye") -> bool:
        """Tests whether a point lies within the cornea boundaries.

        Uses projection distance calculation to determine if point is within corneal depth.

        Args:
            p: Point to test in local eye coordinates
            eye: Eye object containing apex position and cornea depth

        Returns:
            bool: True if point lies within cornea boundaries, False otherwise

        """
        # Calculate direction from apex to cornea center
        apex_pos = eye.cornea.get_apex_position()
        direction = Vector3D(self.center.x - apex_pos.x, self.center.y - apex_pos.y, self.center.z - apex_pos.z)
        diff = Vector3D(p.x - apex_pos.x, p.y - apex_pos.y, p.z - apex_pos.z)

        # Use dot product for projection distance calculation
        projection_distance = diff.dot(direction) / direction.magnitude()
        return projection_distance < eye.cornea.get_corneal_depth()

    @abstractmethod
    def find_reflection(
        self, light_pos: Position3D, camera_pos: Position3D, eye_transform: TransformationMatrix
    ) -> Point3D | None:
        """Finds position of a glint on the corneal surface."""

    @abstractmethod
    def find_refraction(
        self,
        camera_pos: Position3D,
        object_pos: Position3D,
        n_outside: float,
        n_cornea: float,
        eye_transform: TransformationMatrix,
    ) -> Point3D | None:
        """Finds position where refraction occurs on the corneal surface."""

    def __str__(self) -> str:
        """Basic string representation of the cornea."""
        center_str = f"({self.center.x:.1f}, {self.center.y:.1f}, {self.center.z:.1f})mm" if self.center else "unset"
        return f"{self.__class__.__name__}(center={center_str}, type={self.cornea_type})"

    def pprint(self) -> None:
        """Print detailed cornea parameters in a formatted table."""
        # Base parameters
        data = [
            ["Cornea type", self.cornea_type],
            [
                "Center (x,y,z) mm",
                f"({self.center.x:.3f}, {self.center.y:.3f}, {self.center.z:.3f})" if self.center else "unset",
            ],
        ]

        # SphericalCornea and ConicCornea parameters
        if isinstance(self, (SphericalCornea, ConicCornea)):
            data.extend([
                ["Anterior radius (mm)", f"{self.anterior_radius:.3f}"],
                ["Posterior radius (mm)", f"{self.posterior_radius:.3f}"],
                ["Refractive index", f"{self.refractive_index:.3f}"],
                ["Thickness offset (mm)", f"{self.thickness_offset:.3f}"],
                ["Corneal depth (mm)", f"{self.get_corneal_depth():.3f}"],
            ])

            if self.center:
                apex = self.get_apex_position()
                data.append([
                    "Apex position (x,y,z) mm",
                    f"({apex.x:.3f}, {apex.y:.3f}, {apex.z:.3f})",
                ])

        # ConicCornea-specific parameters
        if isinstance(self, ConicCornea):
            data.extend([
                ["Anterior k (conic)", f"{self.anterior_k:.3f}"],
                ["Posterior k (conic)", f"{self.posterior_k:.3f}"],
            ])

        headers = ["Parameter", "Value"]
        info(f"{self.__class__.__name__} Parameters:")
        table(data, headers=headers, tablefmt="grid")


@dataclass
class SphericalCornea(Cornea):
    """Represents a cornea with dual spherical surfaces.

    Uses anatomical scaling based on Boff and Lincoln [1988] parameters.
    Implements proportional scaling of all eye dimensions based on corneal radius.

    Attributes:
        anterior_radius (float): The radius of the anterior (outer) corneal surface.
        posterior_radius (float): The radius of the posterior (inner) corneal surface.
        thickness (float): Central corneal thickness.
        center (Point4D): Inherited from parent. If None, will be calculated by Eye based on anatomical scaling.

    """

    anterior_radius: float = CorneaDefaults.ANTERIOR_RADIUS
    refractive_index: float = CorneaDefaults.REFRACTIVE_INDEX

    # Reference values for scaling (from Boff and Lincoln [1988])
    _posterior_radius_default: float = CorneaDefaults.POSTERIOR_RADIUS
    _thickness_offset_default: float = CorneaDefaults.THICKNESS_OFFSET

    @property
    def cornea_type(self) -> str:
        """Return the type name of this cornea model."""
        return "spherical"

    # Sphere-specific constants from Boff and Lincoln [1988, Section 1.210]
    _r_cornea_default: float = CorneaDefaults.ANTERIOR_RADIUS

    @property
    def posterior_radius(self) -> float:
        """Calculate the scaled posterior corneal radius.

        Returns:
            Posterior corneal radius in mm (scaled from reference)

        """
        scale = self.get_scale_factor()
        return scale * self._posterior_radius_default

    @property
    def thickness_offset(self) -> float:
        """Calculate the scaled thickness offset.

        Returns:
            Thickness offset in mm (scaled from reference)

        """
        scale = self.get_scale_factor()
        return scale * self._thickness_offset_default

    @property
    def thickness(self) -> float:
        """Calculate the central corneal thickness based on radii and offset.

        For a dual-surface spherical cornea, the central thickness is the distance
        between the anterior and posterior surfaces along the optical axis.

        Returns:
            Central corneal thickness in mm

        """
        # Distance between surface centers along optical axis
        center_distance = abs(self.anterior_radius - self.posterior_radius - self.thickness_offset)
        return center_distance

    def get_posterior_center(self) -> Position3D:
        """Calculate the center of the posterior surface based on thickness."""
        thickness_term = self.anterior_radius - self.posterior_radius - self.thickness_offset
        return Position3D(self.center.x, self.center.y, self.center.z - thickness_term)

    @staticmethod
    def calculate_center_position(
        scale: float, axial_length: float, cornea_center_to_rotation_center: float
    ) -> Position3D:
        """Calculate the center position for spherical cornea based on anatomical parameters.

        This implements the original MATLAB/Eye logic for positioning the spherical cornea center.

        Args:
            scale: Scaling factor based on corneal radius
            axial_length: Total axial length of eye (mm)
            cornea_center_to_rotation_center: Distance from corneal center to rotation center (mm)

        Returns:
            Cornea center position

        """
        cornea_z_offset = axial_length - 2 * cornea_center_to_rotation_center
        return Position3D(0, 0, -scale * cornea_z_offset)

    def get_apex_position(self) -> Position3D:
        """Calculate the apex position for spherical cornea.

        For spherical cornea, apex is at center + [0, 0, -radius, 0]

        Returns:
            Corneal apex position

        """
        return Position3D(self.center.x, self.center.y, self.center.z - self.anterior_radius)

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
            Corneal depth in mm (scaled from reference depth)

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
        if self._center is None:
            self.center = self.calculate_center_position(
                scale, axial_length, self._cornea_center_to_rotation_center_default
            )

        return {
            "scale": scale,
            "corneal_depth": self.get_corneal_depth(),
            "apex_position": self.get_apex_position(),
            "cornea_center_to_rotation_center": self._cornea_center_to_rotation_center_default,
        }

    def intersect(self, ray: Ray) -> IntersectionResult | None:
        """Calculates intersection for a spherical cornea.

        Returns the intersection result closer to the ray origin.
        """
        intersection_result, _ = intersections.intersect_ray_sphere(ray, self.center, self.anterior_radius)
        return intersection_result

    def normal_at(self, point: Point3D) -> Direction3D:
        """Calculates the normal vector for a spherical surface."""
        # Calculate normal vector from center to point
        normal_vec = Direction3D(point.x - self.center.x, point.y - self.center.y, point.z - self.center.z)
        return normal_vec.normalize()

    def find_reflection(
        self, light_pos: Position3D, camera_pos: Position3D, eye_transform: TransformationMatrix
    ) -> Point3D | None:
        """Finds position of a glint on the spherical corneal surface."""
        world_center_homogeneous = eye_transform @ np.array(self.center)
        world_center = Position3D.from_array(world_center_homogeneous)
        return reflections.find_reflection_sphere(light_pos, camera_pos, world_center, self.anterior_radius)

    def find_refraction(
        self,
        camera_pos: Position3D,
        object_pos: Position3D,
        n_outside: float,
        n_cornea: float,
        eye_transform: TransformationMatrix,
    ) -> Point3D | None:
        """Finds position where refraction occurs on the spherical corneal surface."""
        world_center_homogeneous = eye_transform @ np.array(self.center)
        world_center = Position3D.from_array(world_center_homogeneous)
        return refractions.find_refraction_sphere(
            camera_pos, object_pos, world_center, self.anterior_radius, n_outside, n_cornea
        )

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "cornea_type": self.cornea_type,
            "center": self.center.serialize() if self.center else None,
            "anterior_radius": float(self.anterior_radius),
            "refractive_index": float(self.refractive_index),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "SphericalCornea":
        """Deserialize from dictionary representation."""
        cornea = cls(anterior_radius=data["anterior_radius"], refractive_index=data["refractive_index"])
        if data["center"]:
            cornea.center = Position3D.deserialize(data["center"])
        return cornea


@dataclass
class ConicCornea(Cornea):
    """Represents a cornea with dual conic surfaces.

    Uses conic section geometry with formula: (x-cx)² + (y-cy)² + (1+k)(z-cz)² - 2*R*(z-cz) = 0
    Implements absolute dimensions without scaling for mathematical consistency.

    Default parameters are 30-year-old values from Goncharov & Dainty (2007).

    Attributes:
        center (Point4D): The 4D homogeneous coordinate of the conic center.
        anterior_radius (float): Anterior surface radius of curvature at apex in mm.
        anterior_k (float): Anterior surface conic constant.
        posterior_radius (float): Posterior surface radius of curvature at apex in mm.
        posterior_k (float): Posterior surface conic constant.
        thickness (float): Central corneal thickness.
        refractive_index (float): Refractive index of cornea.
        thickness_offset (float): Corneal thickness offset.

        k-value meanings:
                  - k = 0: Perfect sphere
                  - k < 0: Prolate ellipsoid (typical cornea, flattens toward periphery)
                  - k > 0: Oblate ellipsoid (steepens toward periphery)

    """

    # Anterior surface (30-year defaults from Goncharov & Dainty 2007)
    anterior_radius: float = CorneaDefaults.CONIC_ANTERIOR_RADIUS
    anterior_k: float = CorneaDefaults.CONIC_ANTERIOR_K

    # Posterior surface (30-year defaults from Goncharov & Dainty 2007)
    posterior_radius: float = CorneaDefaults.CONIC_POSTERIOR_RADIUS
    posterior_k: float = CorneaDefaults.CONIC_POSTERIOR_K

    @property
    def cornea_type(self) -> str:
        """Return the type name of this cornea model."""
        return "conic"

    # Corneal properties
    thickness_offset: float = CorneaDefaults.CONIC_THICKNESS_OFFSET
    refractive_index: float = CorneaDefaults.REFRACTIVE_INDEX

    def get_posterior_center(self) -> Position3D:
        """Calculate the center of the posterior surface based on conic geometry and thickness."""
        # Calculate anterior apex position relative to anterior center
        anterior_apex_offset = -self.anterior_radius / (1 + self.anterior_k)

        # Calculate posterior apex position relative to posterior center
        posterior_apex_offset = -self.posterior_radius / (1 + self.posterior_k)

        # For cornea facing -z direction:
        # - Anterior apex is the foremost point (closest to +z)
        # - Posterior apex should be thickness_offset BEHIND anterior apex (more negative z)
        #
        # anterior_apex_z = anterior_center_z + anterior_apex_offset
        # posterior_apex_z = anterior_apex_z - thickness_offset  (more negative = behind)
        # posterior_center_z = posterior_apex_z - posterior_apex_offset

        anterior_apex_z = self.center.z + anterior_apex_offset
        posterior_apex_z = anterior_apex_z + self.thickness_offset  # Behind anterior apex
        posterior_center_z = posterior_apex_z - posterior_apex_offset  # Account for posterior apex offset

        return Position3D(self.center.x, self.center.y, posterior_center_z)

    def get_apex_position(self) -> Position3D:
        """Calculate the apex position for conic cornea.

        For conic cornea, apex is mathematically at z = -R/(1+k) from center.
        This is the foremost point along the -Z axis.

        Returns:
            Corneal apex position

        """
        # Mathematical apex position: z = -R/(1+k) from center
        apex_z = -self.anterior_radius / (1 + self.anterior_k)
        return Position3D(self.center.x, self.center.y, self.center.z + apex_z)

    def get_corneal_depth(self) -> float:
        """Calculate the corneal depth for conic cornea.

        For consistency with spherical cornea, we use the same reference depth.
        This ensures that point_within_cornea behaves consistently between models.

        Returns:
            Corneal depth in mm

        """
        return self._cornea_depth_default

    def get_scale_factor(self) -> float:  # noqa: PLR6301
        """Get scale factor for conic cornea.

        Conic cornea uses absolute dimensions, so scale factor is always 1.0.

        Returns:
            Scale factor of 1.0 (no scaling)

        """
        return 1.0

    @staticmethod
    def calculate_center_position(axial_length: float, cornea_center_to_rotation_center: float) -> Position3D:
        """Calculate the center position for conic cornea based on anatomical parameters (no scaling).

        Args:
            axial_length: Total axial length of eye (mm)
            cornea_center_to_rotation_center: Distance from corneal center to rotation center (mm)

        Returns:
            Cornea center position

        """
        cornea_z_offset = axial_length - 2 * cornea_center_to_rotation_center
        return Position3D(0, 0, -cornea_z_offset)

    def setup_eye_geometry(self, axial_length: float) -> dict:
        """Setup conic cornea geometry parameters.

        Unlike spherical cornea, conic cornea does not use scaling - it uses absolute dimensions.

        Args:
            axial_length: Total axial length of the eye (not used for conic cornea)

        Returns:
            Dictionary containing geometry parameters

        """
        # Set default center if not already set (no scaling applied)
        # For conic: position center at origin for mathematical consistency
        if self._center is None:
            # Use anatomical offset if desired, otherwise origin
            self.center = self.calculate_center_position(axial_length, self._cornea_center_to_rotation_center_default)

        return {
            "scale": 1.0,  # No scaling for conic cornea
            "corneal_depth": self.get_corneal_depth(),
            "apex_position": self.get_apex_position(),
            "cornea_center_to_rotation_center": 0.0,  # Not applicable for conic #CHECK
        }

    def __post_init__(self, center_init: Position3D | None) -> None:
        """Initialize conic cornea with validation of k parameters."""
        super().__post_init__(center_init)
        # Validate k parameter ranges for both surfaces
        if self.anterior_k < -1:
            warning(f"anterior_k = {self.anterior_k} < -1 may represent unusual corneal geometry")
        if self.posterior_k < -1:
            warning(f"posterior_k = {self.posterior_k} < -1 may represent unusual corneal geometry")

        # Calculate (1+k) values for reference
        anterior_1_plus_k = 1 + self.anterior_k
        posterior_1_plus_k = 1 + self.posterior_k
        if anterior_1_plus_k <= 0:
            warning(f"anterior (1+k) = {anterior_1_plus_k} ≤ 0 may cause numerical issues in conic calculations")
        if posterior_1_plus_k <= 0:
            warning(f"posterior (1+k) = {posterior_1_plus_k} ≤ 0 may cause numerical issues in conic calculations")

    def intersect(self, ray: Ray) -> IntersectionResult | None:
        """Calculates intersection for the anterior conic surface.

        Returns the intersection result closer to the ray origin.
        """
        intersection_result, _ = intersections.intersect_ray_conic(
            ray, self.center, self.anterior_radius, self.anterior_k
        )
        return intersection_result

    def normal_at(self, point: Point3D) -> Direction3D:
        """Calculates the normal vector for the anterior conic surface."""
        return intersections.conic_surface_normal(point, self.center, self.anterior_radius, self.anterior_k)

    def find_reflection(
        self, light_pos: Position3D, camera_pos: Position3D, eye_transform: TransformationMatrix
    ) -> Point3D | None:
        """Finds position of a glint on the anterior conic surface."""
        world_center_homogeneous = eye_transform @ np.array(self.center)
        world_center = Position3D.from_array(world_center_homogeneous)
        return reflections.find_reflection_conic(
            light_pos, camera_pos, world_center, self.anterior_radius, self.anterior_k
        )

    def find_refraction(
        self,
        camera_pos: Position3D,
        object_pos: Position3D,
        n_outside: float,
        n_cornea: float,
        eye_transform: TransformationMatrix,
    ) -> Point3D | None:
        """Finds position where refraction occurs on the anterior conic surface."""
        world_center_homogeneous = eye_transform @ np.array(self.center)
        world_center = Position3D.from_array(world_center_homogeneous)
        return refractions.find_refraction_conic(
            camera_pos, object_pos, world_center, self.anterior_radius, self.anterior_k, n_outside, n_cornea
        )

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "cornea_type": self.cornea_type,
            "center": self.center.serialize() if self.center else None,
            "anterior_radius": float(self.anterior_radius),
            "anterior_k": float(self.anterior_k),
            "posterior_radius": float(self.posterior_radius),
            "posterior_k": float(self.posterior_k),
            "refractive_index": float(self.refractive_index),
            "thickness_offset": float(self.thickness_offset),
        }

    @classmethod
    def deserialize(cls, data: dict) -> "ConicCornea":
        """Deserialize from dictionary representation."""
        cornea = cls(
            anterior_radius=data["anterior_radius"],
            anterior_k=data["anterior_k"],
            posterior_radius=data["posterior_radius"],
            posterior_k=data["posterior_k"],
            refractive_index=data["refractive_index"],
            thickness_offset=data["thickness_offset"],
        )
        if data["center"]:
            cornea.center = Position3D.deserialize(data["center"])
        return cornea


def create_cornea(cornea_model_type: str, center: Position3D, **kwargs: float) -> Cornea:
    """Factory function to create a cornea object of specified type.

    Provides unified interface for creating different corneal models.
    Supports both spherical and conic corneal geometries.

    Args:
        cornea_model_type (str): The type of cornea model to create.
                                 Supported types: "spherical", "conic".
        center (Point4D): The center of the cornea.
        **kwargs: Additional parameters required for the specific cornea model.
                  For "spherical": anterior_radius, refractive_index
                  For "conic": anterior_radius, anterior_k, posterior_radius, posterior_k, refractive_index, thickness_offset

    Returns:
        Cornea: An instance of the specified Cornea subclass.

    Raises:
        ValueError: If an unsupported cornea_model_type is provided.

    """
    if cornea_model_type == "spherical":
        return SphericalCornea(center_init=center, **kwargs)
    if cornea_model_type == "conic":
        return ConicCornea(center_init=center, **kwargs)
    raise ValueError(f"Unknown cornea model type: '{cornea_model_type}'")
