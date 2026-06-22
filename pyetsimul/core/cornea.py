"""Cornea model definitions for eye tracking simulation.

Defines abstract and concrete cornea models (spherical, conic, toric) for anatomical and optical simulation.
"""

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyetsimul.log import info, table, warning

from ..geometry import intersections
from ..optics import reflections, refraction_batch, refractions
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

    # When True, find_refraction models refraction at both corneal surfaces (posterior aqueous->cornea and
    # anterior cornea->air) instead of the anterior surface alone.
    use_posterior_surface: bool = False

    # When True, a thin tear film is modelled as a third refracting surface in front of the anterior cornea
    # (air -> tears -> cornea): the anterior shape shifted forward by ``tear_film_thickness`` mm, with
    # ``tear_refractive_index`` between air and the cornea (Werkmeister et al. 2013, ~5 um thick). Requires
    # use_posterior_surface. It lowers the anterior surface's outside index from the cornea to the tears,
    # reducing the corneal magnification of the pupil.
    use_tear_film: bool = False
    tear_film_thickness: float = 0.005
    tear_refractive_index: float = 1.336

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

    @abstractmethod
    def intersect_posterior(self, ray: Ray) -> IntersectionResult | None:
        """Calculates the intersection of a ray with the posterior corneal surface."""

    @abstractmethod
    def normal_at_posterior(self, point: Point3D) -> Direction3D:
        """Calculates the normal vector at a given point on the posterior corneal surface."""

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

    def find_refraction(
        self,
        camera_pos: Position3D,
        object_pos: Position3D,
        n_outside: float,
        n_cornea: float,
        eye_transform: TransformationMatrix,
        n_aqueous: float,
    ) -> Point3D | None:
        """Corneal point where ``object_pos`` refracts toward ``camera_pos``, or None.

        A single-point call into ``find_refraction_batch``, which is the one refraction implementation.
        """
        points, valid = self.find_refraction_batch(
            camera_pos,
            np.array([[object_pos.x, object_pos.y, object_pos.z]]),
            n_outside,
            n_cornea,
            eye_transform,
            n_aqueous,
        )
        if not valid[0]:
            return None
        return Point3D(points[0, 0], points[0, 1], points[0, 2])

    @abstractmethod
    def find_refraction_batch(
        self,
        camera_pos: Position3D,
        object_points: np.ndarray,
        n_outside: float,
        n_cornea: float,
        eye_transform: TransformationMatrix,
        n_aqueous: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Corneal points where ``(N, 3)`` object points refract toward ``camera_pos``; returns (points, valid)."""

    def refract_ray_inward(self, ray: Ray, n_outside: float, n_cornea: float, n_aqueous: float) -> Ray | None:
        """Trace a ray arriving from the air inward through the cornea.

        Inverse of find_refraction: a ray is followed from outside the eye, refracted at the anterior
        surface (air -> cornea) and, when use_posterior_surface is enabled, again at the posterior surface
        (cornea -> aqueous). The returned ray continues in the medium behind the last refracting surface.
        Works in eye-local coordinates, where the conic axis is aligned with the Z-axis.

        Args:
            ray: Incoming ray in eye-local coordinates, with its origin outside the eye.
            n_outside: Refractive index outside the cornea (air).
            n_cornea: Refractive index of the cornea.
            n_aqueous: Refractive index of the aqueous humor.

        Returns:
            Refracted ray in eye-local coordinates, or None if the ray does not pass through.

        """
        anterior_hit = self.intersect(ray)
        if anterior_hit is None or not anterior_hit.intersects:
            return None
        into_cornea = refractions.refract_direction(
            ray.direction, self.normal_at(anterior_hit.point), n_outside, n_cornea
        )
        if into_cornea is None:
            return None
        if not self.use_posterior_surface:
            return Ray(origin=anterior_hit.point, direction=into_cornea)

        posterior_hit = self.intersect_posterior(Ray(origin=anterior_hit.point, direction=into_cornea))
        if posterior_hit is None or not posterior_hit.intersects:
            return None
        into_aqueous = refractions.refract_direction(
            into_cornea, self.normal_at_posterior(posterior_hit.point), n_cornea, n_aqueous
        )
        if into_aqueous is None:
            return None
        return Ray(origin=posterior_hit.point, direction=into_aqueous)

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

    def intersect_posterior(self, ray: Ray) -> IntersectionResult | None:
        """Calculates intersection with the posterior spherical surface (closer root)."""
        result, _ = intersections.intersect_ray_sphere(ray, self.get_posterior_center(), self.posterior_radius)
        return result

    def normal_at_posterior(self, point: Point3D) -> Direction3D:
        """Calculates the normal vector on the posterior spherical surface."""
        center = self.get_posterior_center()
        return Direction3D(point.x - center.x, point.y - center.y, point.z - center.z).normalize()

    def find_reflection(
        self, light_pos: Position3D, camera_pos: Position3D, eye_transform: TransformationMatrix
    ) -> Point3D | None:
        """Finds position of a glint on the spherical corneal surface."""
        world_center_homogeneous = eye_transform @ np.array(self.center)
        world_center = Position3D.from_array(world_center_homogeneous)
        return reflections.find_reflection_sphere(light_pos, camera_pos, world_center, self.anterior_radius)

    def find_refraction_batch(
        self,
        camera_pos: Position3D,
        object_points: np.ndarray,
        n_outside: float,
        n_cornea: float,
        eye_transform: TransformationMatrix,
        n_aqueous: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Corneal points where ``(N, 3)`` object points refract toward the camera; returns (points, valid).

        Spherical surfaces are searched in world coordinates; invalid rows hold NaN.
        """
        et = np.asarray(eye_transform)
        camera = np.array([camera_pos.x, camera_pos.y, camera_pos.z])
        world_center = (et @ np.array(self.center))[:3]
        if self.use_posterior_surface:
            world_posterior = (et @ np.array(self.get_posterior_center()))[:3]
            return refraction_batch.find_refraction_dual_sphere_batch(
                camera,
                object_points,
                world_center,
                self.anterior_radius,
                world_posterior,
                self.posterior_radius,
                n_outside,
                n_cornea,
                n_aqueous,
            )
        return refraction_batch.find_refraction_sphere_batch(
            camera, object_points, world_center, self.anterior_radius, n_outside, n_cornea
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

    # Static tilt of the corneal surface relative to the eye optical axis, in degrees, pivoting about the apex.
    # tilt_x_deg rotates the corneal axis about the eye-local x-axis (a vertical tilt: the cornea points up/down);
    # tilt_y_deg rotates about the y-axis (a horizontal tilt). 0 = the corneal axis is aligned with the eye axis.
    # Models corneal misalignment (Navarro et al. 2013, "rotated about three meridians"; Atchison & Smith, Optics
    # of the Human Eye, 2nd ed.). The whole cornea (anterior and posterior surfaces) tilts as a rigid body, so it
    # reorients both the refraction of the pupil and the reflection of the glint.
    tilt_x_deg: float = 0.0
    tilt_y_deg: float = 0.0

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

    def intersect_posterior(self, ray: Ray) -> IntersectionResult | None:
        """Calculates intersection with the posterior conic surface (closer root)."""
        result, _ = intersections.intersect_ray_conic(
            ray, self.get_posterior_center(), self.posterior_radius, self.posterior_k
        )
        return result

    def normal_at_posterior(self, point: Point3D) -> Direction3D:
        """Calculates the normal vector on the posterior conic surface."""
        return intersections.conic_surface_normal(
            point, self.get_posterior_center(), self.posterior_radius, self.posterior_k
        )

    @staticmethod
    def _anterior_lateral_coeffs() -> tuple[float, float, float]:
        """Lateral quadric coefficients (axx, ayy, axy) of the anterior surface.

        The rotationally symmetric conic returns (1, 1, 0); ToricCornea overrides this so the anterior surface
        becomes astigmatic (two principal-meridian radii at an axis).
        """
        return 1.0, 1.0, 0.0

    def _tilt_active(self) -> bool:
        """True when the corneal surface is tilted relative to the eye axis."""
        return bool(self.tilt_x_deg or self.tilt_y_deg)

    def _tilt_rotation(self) -> np.ndarray:
        """Rotation R mapping the (un-tilted) cornea frame to eye-local; R.T maps eye-local into the cornea frame.

        The corneal axis is tilted by tilt_x_deg about the eye-local x-axis and tilt_y_deg about the y-axis. Both
        the reflection and the refraction solvers assume the conic axis is the eye-local Z-axis, so a tilt is
        applied by rotating the light/camera/object points into the un-tilted cornea frame (about the apex,
        through which the conic axis passes), solving, and rotating the result back.
        """
        ax, ay = np.radians(self.tilt_x_deg), np.radians(self.tilt_y_deg)
        rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(ax), -np.sin(ax)], [0.0, np.sin(ax), np.cos(ax)]])
        ry = np.array([[np.cos(ay), 0.0, np.sin(ay)], [0.0, 1.0, 0.0], [-np.sin(ay), 0.0, np.cos(ay)]])
        return rx @ ry

    def find_reflection(
        self, light_pos: Position3D, camera_pos: Position3D, eye_transform: TransformationMatrix
    ) -> Point3D | None:
        """Finds position of a glint on the anterior conic surface.

        Transforms light and camera positions into eye-local coordinates where
        the conic axis is aligned with the Z-axis, runs the reflection solver,
        then transforms the result back to world coordinates.
        """
        inv_transform = np.linalg.inv(eye_transform)
        local_light = Position3D.from_array(inv_transform @ np.array([light_pos.x, light_pos.y, light_pos.z, 1.0]))
        local_camera = Position3D.from_array(inv_transform @ np.array([camera_pos.x, camera_pos.y, camera_pos.z, 1.0]))

        if self._tilt_active():
            rot = self._tilt_rotation()
            apex = np.array([self.get_apex_position().x, self.get_apex_position().y, self.get_apex_position().z])
            local_light = Position3D(*((np.array([local_light.x, local_light.y, local_light.z]) - apex) @ rot + apex))
            local_camera = Position3D(
                *((np.array([local_camera.x, local_camera.y, local_camera.z]) - apex) @ rot + apex)
            )

        local_glint = reflections.find_reflection_conic(
            local_light, local_camera, self.center, self.anterior_radius, self.anterior_k
        )
        if local_glint is None:
            return None

        if self._tilt_active():
            local_glint = Point3D(*((np.array([local_glint.x, local_glint.y, local_glint.z]) - apex) @ rot.T + apex))

        world_glint = eye_transform @ np.array([local_glint.x, local_glint.y, local_glint.z, 1.0])
        return Point3D(world_glint[0], world_glint[1], world_glint[2])

    def find_refraction_batch(
        self,
        camera_pos: Position3D,
        object_points: np.ndarray,
        n_outside: float,
        n_cornea: float,
        eye_transform: TransformationMatrix,
        n_aqueous: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Corneal points where ``(N, 3)`` object points refract toward the camera; returns (points, valid).

        Camera and objects are taken to eye-local coordinates (conic axis on Z), all points are solved
        together, and the results are returned to world. Invalid rows hold NaN.
        """
        et = np.asarray(eye_transform)
        inv = np.linalg.inv(et)
        cam_local = (inv @ np.array([camera_pos.x, camera_pos.y, camera_pos.z, 1.0]))[:3]
        homog = np.column_stack([object_points, np.ones(len(object_points))])
        obj_local = (homog @ inv.T)[:, :3]
        tilt = self._tilt_active()
        if tilt:
            rot = self._tilt_rotation()
            apex = np.array([self.get_apex_position().x, self.get_apex_position().y, self.get_apex_position().z])
            cam_local = (cam_local - apex) @ rot + apex
            obj_local = (obj_local - apex) @ rot + apex
        center = np.array([self.center.x, self.center.y, self.center.z])
        if self.use_posterior_surface:
            pc = self.get_posterior_center()
            posterior = np.array([pc.x, pc.y, pc.z])
            axx, ayy, axy = self._anterior_lateral_coeffs()
            if self.use_tear_film:
                tear_center = center.copy()
                tear_center[2] -= (
                    self.tear_film_thickness
                )  # tear surface = anterior shape shifted forward (toward the apex)
                local_points, valid = refraction_batch.find_refraction_triple_conic_batch(
                    cam_local,
                    obj_local,
                    center,
                    self.anterior_radius,
                    self.anterior_k,
                    posterior,
                    self.posterior_radius,
                    self.posterior_k,
                    tear_center,
                    n_outside,
                    n_cornea,
                    n_aqueous,
                    self.tear_refractive_index,
                    ant_axx=axx,
                    ant_ayy=ayy,
                    ant_axy=axy,
                )
            else:
                local_points, valid = refraction_batch.find_refraction_dual_conic_batch(
                    cam_local,
                    obj_local,
                    center,
                    self.anterior_radius,
                    self.anterior_k,
                    posterior,
                    self.posterior_radius,
                    self.posterior_k,
                    n_outside,
                    n_cornea,
                    n_aqueous,
                    ant_axx=axx,
                    ant_ayy=ayy,
                    ant_axy=axy,
                )
        else:
            local_points, valid = refraction_batch.find_refraction_conic_batch(
                cam_local, obj_local, center, self.anterior_radius, self.anterior_k, n_outside, n_cornea
            )
        if tilt:
            local_points = (local_points - apex) @ rot.T + apex
        homog_out = np.column_stack([local_points, np.ones(len(local_points))])
        world = (homog_out @ et.T)[:, :3]
        return np.where(valid[:, None], world, np.nan), valid

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


@dataclass
class ToricCornea(ConicCornea):
    """Conic cornea with a toric (astigmatic) anterior surface.

    The anterior surface has two principal-meridian apex radii differing by ``anterior_astigmatism_mm`` about the
    conic ``anterior_radius``, with the steeper meridian oriented at ``anterior_astigmatism_axis_deg``. The
    posterior surface stays rotationally symmetric, and all other geometry (asphericity, posterior surface,
    thickness) is the conic base.

    Defaults are population means for young adults from Atchison & Smith, Optics of the Human Eye, 2nd ed., 2023,
    section 2.2.4: anterior corneal astigmatism is predominantly with-the-rule (steeper vertical meridian, axis
    ~90 deg) and about 0.5-1 D (about three quarters of eyes have >= 0.5 D; Hoffmann & Hutz 2010), which is
    ~0.1 mm of meridian-radius difference at the 7.76 mm anterior radius. Mirror symmetry of the axis between
    fellow eyes is common (Read et al. 2017). Set the fields like any other cornea to model a specific eye.

    References:
        Atchison DA & Smith G. Optics of the Human Eye, 2nd ed., 2023, section 2.2.4 (corneal astigmatism).
        Hoffmann PC & Hutz WW. J Cataract Refract Surg 2010, 36(9):1479-1485 (astigmatism prevalence).
        Read SA et al. 2017 (mirror symmetry of fellow-eye corneal-astigmatism axes), cited therein.

    """

    anterior_astigmatism_mm: float = 0.10  # radius difference between principal meridians (~0.6 D at 7.76 mm)
    anterior_astigmatism_axis_deg: float = 90.0  # steep-meridian orientation; 90 deg = with-the-rule

    @property
    def cornea_type(self) -> str:
        """Return the type name of this cornea model."""
        return "toric"

    def _anterior_lateral_coeffs(self) -> tuple[float, float, float]:
        """Anterior lateral quadric coefficients (axx, ayy, axy) for the two principal meridians at the axis."""
        r = self.anterior_radius
        flat = r / (r + self.anterior_astigmatism_mm / 2.0)  # flatter (larger-radius) meridian coefficient
        steep = r / (r - self.anterior_astigmatism_mm / 2.0)  # steeper (smaller-radius) meridian coefficient
        phi = np.radians(self.anterior_astigmatism_axis_deg)  # orientation of the steep meridian
        cos2, sin2, cs = np.cos(phi) ** 2, np.sin(phi) ** 2, np.sin(phi) * np.cos(phi)
        return float(steep * cos2 + flat * sin2), float(steep * sin2 + flat * cos2), float((steep - flat) * cs)

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        data = super().serialize()
        data["anterior_astigmatism_mm"] = float(self.anterior_astigmatism_mm)
        data["anterior_astigmatism_axis_deg"] = float(self.anterior_astigmatism_axis_deg)
        return data

    @classmethod
    def deserialize(cls, data: dict) -> "ToricCornea":
        """Deserialize from dictionary representation."""
        cornea = cls(
            anterior_radius=data["anterior_radius"],
            anterior_k=data["anterior_k"],
            posterior_radius=data["posterior_radius"],
            posterior_k=data["posterior_k"],
            refractive_index=data["refractive_index"],
            thickness_offset=data["thickness_offset"],
            anterior_astigmatism_mm=data.get("anterior_astigmatism_mm", 0.0),
            anterior_astigmatism_axis_deg=data.get("anterior_astigmatism_axis_deg", 90.0),
        )
        if data.get("center"):
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
    if cornea_model_type == "toric":
        return ToricCornea(center_init=center, **kwargs)
    raise ValueError(f"Unknown cornea model type: '{cornea_model_type}'")
