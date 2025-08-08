"""Main eye model for eye tracking simulation.

Defines the Eye class, integrating cornea, pupil, fovea displacement, and gaze mechanics for simulation and analysis.
"""

import numpy as np
import warnings
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING


from ..types import Position3D, Direction3D, TransformationMatrix, RotationMatrix, PupilData
from .pupil import Pupil, create_pupil
from .cornea import SphericalCornea
from .eye_operations import look_at_target
from ..optics.reflections import find_corneal_reflection
from ..optics.refractions import find_refraction_point
from ..optics.pupil_imaging import calculate_pupil_center_from_boundary

if TYPE_CHECKING:
    from .camera import Camera
    from .light import Light


@dataclass
class Eye:
    """Eye model for eye tracking simulation.

    Implements spherical eye model with optical axis along negative z-axis.
    Includes cornea, pupil, fovea displacement, and Listing's law mechanics.
    Based on Böhme et al. 2008 with nodal point at cornea center.

    Eye Model Components:
    - Cornea: Spherical surface with configurable radius and refractive index
    - Pupil: Elliptical aperture with configurable size and position
    - Fovea displacement: Visual axis offset from optical axis
    - Listing's law: Eye rotation mechanics for realistic torsion
    """

    # Instance parameters
    cornea: Optional[SphericalCornea] = None  # Spherical cornea object
    fovea_displacement: bool = True
    fovea_alpha_deg: float = 6.0  # Horizontal fovea displacement (degrees)
    fovea_beta_deg: float = 2.0  # Vertical fovea displacement (degrees)
    pupil_type: str = "elliptical"  # Pupil type: "elliptical" (default), "realistic"
    pupil_boundary_points: Optional[int] = None  # Number of points for pupil boundary (uses pupil default if None)
    pupil_random_seed: Optional[int] = None  # Random seed for realistic pupil (None = random, int = deterministic)

    # These fields are calculated in __post_init__
    trans: TransformationMatrix = field(init=False)
    _rest_orientation: RotationMatrix = field(init=False)
    axial_length: float = field(init=False)  # Total axial length of eye (m)
    n_aqueous_humor: float = field(init=False)
    pupil: Pupil = field(init=False)  # Pupil object that handles all pupil calculations

    def __post_init__(self) -> None:
        """Initializes the eye's anatomical properties based on constructor parameters.

        Sets up cornea geometry, transformation matrices, and pupil object.
        Scales pupil size based on corneal scaling factor.
        """
        # General eye constants (not cornea-specific)
        pupil_radius_default = 3e-3  # Default pupil radius (m)
        n_aqueous_humor_default = 1.336  # Refractive index of aqueous humor
        axial_length_default = 24.75e-3  # Default total axial length of eye (m)

        # Create default cornea if none provided
        if self.cornea is None:
            self.cornea = SphericalCornea()

        # Setup cornea-specific geometry (this handles all sphere-specific scaling)
        self.cornea.setup_eye_geometry(axial_length_default)

        # Initialize transformation matrix (identity at rest position)
        self.trans = np.eye(4)
        self._rest_orientation = np.eye(3)
        self.trans[:3, :3] = self._rest_orientation

        # Set general anatomical parameters
        self.axial_length = axial_length_default

        # Refractive indices
        self.n_aqueous_humor = n_aqueous_humor_default

        # Create pupil object - calculate pupil position and scale radius
        pupil_position = self.get_pupil_position()
        # Scale pupil radius based on corneal scaling factor
        scale = self.cornea.get_scale_factor()
        scaled_pupil_radius = pupil_radius_default * scale
        x_pupil = Direction3D(scaled_pupil_radius, 0, 0)
        y_pupil = Direction3D(0, scaled_pupil_radius, 0)

        # Create pupil with optional N parameter and random seed
        pupil_kwargs = {}
        if self.pupil_boundary_points is not None:
            pupil_kwargs["N"] = self.pupil_boundary_points

        # For realistic pupils, create params with random seed if specified
        if self.pupil_type == "realistic" and self.pupil_random_seed is not None:
            from .pupil import RealisticPupilParams

            pupil_params = RealisticPupilParams(random_seed=self.pupil_random_seed)
            pupil_kwargs["params"] = pupil_params

        self.pupil = create_pupil(
            pupil_type=self.pupil_type, pos_pupil=pupil_position, x_pupil=x_pupil, y_pupil=y_pupil, **pupil_kwargs
        )

    @property
    def orientation(self) -> RotationMatrix:
        """Get/set the eye's current orientation (3x3 rotation matrix)."""
        return self.trans[:3, :3]

    @orientation.setter
    def orientation(self, value: RotationMatrix) -> None:
        """Set the eye's current orientation and update transformation matrix."""
        self.trans[:3, :3] = value

    def set_rest_orientation(self, value: RotationMatrix) -> None:
        """Set the rest orientation and initialize current orientation to match.

        Establishes reference orientation for eye rotation calculations.
        Validates right-handed rotation matrix with determinant = +1.

        Args:
            value: 3x3 rotation matrix (must be right-handed with determinant = +1)

        Raises:
            ValueError: If the matrix is not right-handed (det ≠ +1)
        """
        # RotationMatrix type already validates during construction

        self._rest_orientation = value.copy()
        self.trans[:3, :3] = value

    @property
    def rest_orientation(self) -> RotationMatrix:
        """Get the rest orientation (read-only).

        Returns reference orientation for eye rotation calculations.
        """
        return self._rest_orientation.copy()

    @property
    def position(self) -> Position3D:
        """Get/set the eye's position in world coordinates."""
        return Position3D.from_array(self.trans[:, 3])

    @position.setter
    def position(self, value: Position3D) -> None:
        """Set the eye's position and update transformation matrix."""
        self.trans[:, 3] = np.array(value)

    def point_within_cornea(self, p: Position3D) -> bool:
        """Check if a point lies within the corneal boundaries.

        Transforms point to eye coordinates and validates against corneal geometry.

        Args:
            p: Point to test

        Returns:
            True if point lies within cornea boundaries
        """
        # Transform point to local eye coordinates
        p_homogeneous = np.array(p)
        p_local_homogeneous = np.linalg.solve(self.trans, p_homogeneous)
        p_local = Position3D.from_array(p_local_homogeneous[:3])

        # Use cornea object's point_within_cornea method
        return self.cornea.point_within_cornea(p_local, self)

    def find_cr(self, light: "Light", camera: "Camera") -> Optional[Position3D]:
        """Finds the position of a corneal reflex.

        Delegates to reflections module for corneal reflection calculation.

        Args:
            light: Light source object
            camera: Camera object

        Returns:
            Position of corneal reflex, or None if not within cornea
        """
        return find_corneal_reflection(self, light, camera)

    def look_at(self, target_position: Position3D) -> None:
        """Rotates an eye to look at a given position in space.

        Delegates to eye_operations module for gaze control.

        Args:
            target_position: Position in world coordinates to look at
        """
        look_at_target(self, target_position)

    def get_pupil(self) -> PupilData:
        """Get pupil boundary points in world coordinates.

        Retrieves boundary points from pupil object and transforms to world coordinates.

        Returns:
            PupilData object with boundary_points attribute set
        """
        # Get pupil boundary points from pupil object
        pupil_points = self.pupil.get_boundary_points()

        # Transform to world coordinates
        pupil_world = self.trans @ pupil_points

        return PupilData(boundary_points=pupil_world)

    def get_pupil_position(self) -> Position3D:
        """Get pupil center position.

        Calculates pupil position based on corneal apex and depth.

        Returns:
            Pupil center position
        """
        apex = self.cornea.get_apex_position()
        corneal_depth = self.cornea.get_corneal_depth()
        return Position3D(apex.x, apex.y, apex.z + corneal_depth)

    def get_pupil_radii(self) -> tuple[float, float]:
        """Get current pupil radii.

        Delegates to pupil object for radius information.

        Returns:
            Tuple of (x_radius, y_radius) in meters
        """
        return self.pupil.get_radii()

    def set_pupil_radii(self, x_radius: float = None, y_radius: float = None) -> None:
        """Set pupil radii.

        Delegates to pupil object for radius modification.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)
        """
        self.pupil.set_radii(x_radius, y_radius)

    def get_pupil_center_in_world(self) -> Position3D:
        """Get pupil center in world coordinates.

        Delegates to pupil object for world coordinate transformation.

        Returns:
            Pupil center position in world coordinates
        """
        return self.pupil.get_center_world_coords(self.trans)

    def find_refracted_position(
        self, camera_position: Position3D, object_position: Position3D
    ) -> Optional[Position3D]:
        """Find where an intraocular object appears due to corneal refraction.

        Calculates where camera observes intraocular object through corneal refraction,
        including boundary checking to ensure the refraction point lies within the cornea.

        Args:
            camera_position: Camera position (Position3D)
            object_position: Object position inside eye (Position3D)

        Returns:
            Position3D on corneal surface where refraction occurs, or None if no valid solution
        """
        # Call pure refraction function
        refraction_point = find_refraction_point(self.cornea, self.trans, camera_position, object_position)

        # Check if point is within corneal boundaries
        if refraction_point is not None:
            if not self.point_within_cornea(refraction_point.to_position3d()):
                refraction_point = None

        return refraction_point

    @property
    def fovea_position(self) -> Position3D:
        """Calculate the 3D position of the fovea on the retinal surface.

        Uses spherical eye model with fovea displacement angles.
        Positions fovea at axial_length/2 distance from rotation center.

        Returns:
            Fovea position in eye coordinate system
        """
        # Convert displacement angles to radians
        alpha = self.fovea_alpha_deg * np.pi / 180.0  # Horizontal (temporal) displacement
        beta = self.fovea_beta_deg * np.pi / 180.0  # Vertical (upward) displacement

        # Retina distance from rotation center (from our eye model)
        retina_distance = self.axial_length / 2

        # Calculate fovea position in eye coordinate system using spherical coordinates
        fovea_x = retina_distance * np.sin(alpha) * np.cos(beta)  # Temporal displacement
        fovea_y = retina_distance * np.sin(beta)  # Vertical displacement
        fovea_z = retina_distance * np.cos(alpha) * np.cos(beta)  # Along optical axis

        return Position3D(fovea_x, fovea_y, fovea_z)

    @property
    def angle_kappa(self) -> float:
        """Calculate angle kappa - the angle between optical and visual axes.

        Measures angle between optical axis (-Z) and visual axis (to fovea).
        Important for realistic gaze modeling and eye tracking accuracy.

        Returns:
            Angle kappa in degrees
        """
        # Get fovea position
        fovea_pos = self.fovea_position

        # Calculate visual axis direction (normalized)
        visual_axis = Direction3D(fovea_pos.x, fovea_pos.y, fovea_pos.z).normalize()

        # Optical axis points along -Z direction in eye coordinates
        optical_axis = Direction3D(0, 0, -1)

        # Calculate angle between visual and optical axes
        dot_product = visual_axis.dot(optical_axis)
        # Use abs() to get acute angle and clip to avoid numerical errors
        angle_kappa_rad = np.arccos(np.clip(np.abs(dot_product), 0, 1))

        # Convert to degrees
        return angle_kappa_rad * 180.0 / np.pi

    def get_pupil_in_camera_image(self, camera: "Camera", use_refraction: bool = True, center_method: str = "ellipse"):
        """Projects pupil boundary points to camera image coordinates.

        Handles corneal refraction effects and camera projection.
        Supports both refracted and direct projection modes.

        Args:
            camera: Camera object to project into
            use_refraction: Whether to apply refraction effects (default True)
            center_method: Method to use for pupil center detection (default "ellipse")
                          Options: "ellipse", "center_of_mass"

        Returns:
            Tuple of (pupil_boundary, pupil_center) where:
            - pupil_boundary: numpy array of boundary points (2xN)
            - pupil_center: Point2D object of pupil center, or None if invalid
        """

        # Get pupil boundary points in world coordinates
        pupil_data = self.get_pupil()
        pupil_world = pupil_data.boundary_points

        if use_refraction:
            # Apply refraction: for each pupil point, find where it appears due to corneal refraction
            refracted_points = []
            for i in range(pupil_world.shape[1]):
                pupil_point = Position3D.from_array(pupil_world[:, i])
                refracted_point = self.find_refracted_position(camera.position, pupil_point)
                if refracted_point is not None:
                    # Convert Point3D result to Position3D for camera projection
                    refracted_points.append(refracted_point.to_position3d())

            # Project refracted points to camera image coordinates
            if refracted_points:
                projection_result = camera.project(refracted_points)

                # Convert valid boundary points to a 2xN numpy array
                valid_pupil_points = []
                for i in range(projection_result.image_points.shape[1]):
                    if projection_result.valid_mask[i] and not np.any(np.isnan(projection_result.image_points[:, i])):
                        point_2d = np.array(
                            [float(projection_result.image_points[0, i]), float(projection_result.image_points[1, i])]
                        )
                        valid_pupil_points.append(point_2d)

                pupil_boundary_array = None
                if valid_pupil_points:
                    pupil_boundary_array = np.array(valid_pupil_points).T
                else:
                    warnings.warn(
                        "No valid pupil points found in camera image (with refraction). Check camera-eye setup.",
                        UserWarning,
                    )
            else:
                warnings.warn("No refracted pupil points could be computed. Check camera-eye setup.", UserWarning)
                pupil_boundary_array = None
        else:
            # Direct projection without refraction
            projection_result = camera.project(
                [Position3D.from_array(pupil_world[:, i]) for i in range(pupil_world.shape[1])]
            )

            # Convert valid boundary points to a 2xN numpy array
            valid_pupil_points = []
            for i in range(projection_result.image_points.shape[1]):
                if projection_result.valid_mask[i] and not np.any(np.isnan(projection_result.image_points[:, i])):
                    point_2d = np.array(
                        [float(projection_result.image_points[0, i]), float(projection_result.image_points[1, i])]
                    )
                    valid_pupil_points.append(point_2d)

            pupil_boundary_array = None
            if valid_pupil_points:
                pupil_boundary_array = np.array(valid_pupil_points).T
            else:
                warnings.warn(
                    "No valid pupil points found in camera image (without refraction). Check camera-eye setup.",
                    UserWarning,
                )

        # Calculate pupil center using specified method
        pupil_center = None
        if pupil_boundary_array is not None:
            pupil_center = calculate_pupil_center_from_boundary(
                pupil_boundary_array, camera.camera_matrix.resolution, center_method
            )

        return pupil_boundary_array, pupil_center
