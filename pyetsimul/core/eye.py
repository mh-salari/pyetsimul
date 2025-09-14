"""Main eye model for eye tracking simulation.

Defines the Eye class, integrating cornea, pupil, fovea displacement, and gaze mechanics for simulation and analysis.
"""

import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
from tabulate import tabulate

from ..optics.pupil_imaging import calculate_pupil_center_from_boundary
from ..optics.reflections import find_corneal_reflection
from ..optics.refractions import find_refraction_point
from ..types import Direction3D, Point2D, Position3D, PupilData, RotationMatrix, TransformationMatrix
from .cornea import ConicCornea, SphericalCornea
from .default_configs import EyeAnatomyDefaults
from .eye_operations import look_at_target, look_at_target_optical_then_kappa
from .eyelid import Eyelid, create_eyelid
from .pupil import EllipticalPupil, Pupil, RealisticPupilParams, create_pupil
from .pupil_decentration import PupilDecentrationConfig, PupilDecentrationRegistry

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
    cornea: SphericalCornea | ConicCornea = field(
        default_factory=SphericalCornea
    )  # Spherical cornea object by default
    fovea_displacement: bool = True
    fovea_alpha_deg: float = EyeAnatomyDefaults.FOVEA_ALPHA_DEG
    fovea_beta_deg: float = EyeAnatomyDefaults.FOVEA_BETA_DEG
    pupil_type: str = "elliptical"  # Pupil type: "elliptical" (default), "realistic"
    pupil_boundary_points: int | None = None  # Number of points for pupil boundary (uses pupil default if None)
    pupil_random_seed: int | None = None  # Random seed for realistic pupil (None = random, int = deterministic)

    # Eyelid configuration (enabled off by default to avoid behavior changes)
    eyelid_enabled: bool = False

    # Pupil decentration configuration
    decentration_config: PupilDecentrationConfig = field(default_factory=PupilDecentrationConfig)

    # These fields are calculated in __post_init__
    trans: TransformationMatrix = field(init=False)
    # Eyelid transform (local→world): follows eye position but keeps a fixed orientation
    eyelid_trans: TransformationMatrix = field(init=False, repr=False)
    _rest_orientation: RotationMatrix = field(init=False)
    _current_target_point: Position3D | None = field(init=False, default=None)  # Updated by look_at()
    axial_length: float = field(init=False)  # Total axial length of eye (m)
    n_aqueous_humor: float = field(init=False)
    pupil: Pupil = field(init=False)  # Pupil object that handles all pupil calculations
    eyelid: Eyelid | None = field(init=False, default=None)

    def __post_init__(self) -> None:
        """Initializes the eye's anatomical properties based on constructor parameters.

        Sets up cornea geometry, transformation matrices, and pupil object.
        Scales pupil size based on corneal scaling factor.
        """
        pupil_radius_default = EyeAnatomyDefaults.PUPIL_RADIUS
        n_aqueous_humor_default = EyeAnatomyDefaults.N_AQUEOUS_HUMOR
        axial_length_default = EyeAnatomyDefaults.AXIAL_LENGTH

        # Create default cornea if none provided
        if self.cornea is None:
            self.cornea = SphericalCornea()

        # Setup cornea-specific geometry (this handles all sphere-specific scaling)
        self.cornea.setup_eye_geometry(axial_length_default)

        # Initialize transformation matrix (identity at rest position)
        self._rest_orientation = RotationMatrix.identity()
        self.trans = TransformationMatrix.from_rotation(self._rest_orientation)

        # Initialize eyelid transform (same position as eye, orientation = rest)
        self.eyelid_trans = TransformationMatrix.from_translation_and_rotation(
            self.trans.get_translation(), self._rest_orientation
        )

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

        # Create pupil with optional n parameter and random seed
        pupil_kwargs = {}
        if self.pupil_boundary_points is not None:
            pupil_kwargs["n"] = self.pupil_boundary_points

        # For realistic pupils, create params with random seed if specified
        if self.pupil_type == "realistic" and self.pupil_random_seed is not None:
            pupil_params = RealisticPupilParams(random_seed=self.pupil_random_seed)
            pupil_kwargs["params"] = pupil_params

        self.pupil = create_pupil(
            pupil_type=self.pupil_type, pos_pupil=pupil_position, x_pupil=x_pupil, y_pupil=y_pupil, **pupil_kwargs
        )

        # Create eyelid if enabled: positioned at eye center, sphere radius = axial_length/2,
        # phi_max derived from limbus z position so that the footprint matches corneal boundary.
        if self.eyelid_enabled:
            sphere_radius = self.axial_length / 2.0
            apex_pos = self.cornea.get_apex_position()
            limbus_z_local = apex_pos.z + self.cornea.get_corneal_depth()
            # phi from apex normal (-Z): cos(phi) = n·(r̂) = -z/sphere_radius  -> phi = arccos(-z/sphere_radius)
            ratio = np.clip(-limbus_z_local / sphere_radius, -1.0, 1.0)
            phi_max = float(np.arccos(ratio))

            self.eyelid = create_eyelid(
                center=Position3D(0.0, 0.0, 0.0),
                sphere_radius=sphere_radius,
                phi_max=phi_max,
                openness=EyeAnatomyDefaults.EYELID_OPENNESS,
            )
            # Keep eyelid orientation locked to rest orientation (fixed to face)
            self.eyelid_trans[:3, :3] = self._rest_orientation

        # Initialize pupil decentration if enabled
        if self.decentration_config.enabled:
            # Set baseline diameter to current pupil diameter
            if self.decentration_config.baseline_diameter is None:
                self.decentration_config.baseline_diameter = self.get_pupil_diameter()
            # Apply initial decentration
            self._update_pupil_position_with_decentration()

    @property
    def orientation(self) -> RotationMatrix:
        """Get/set the eye's current orientation (3x3 rotation matrix)."""
        return self.trans.get_rotation()

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

        # Keep eyelid orientation aligned to rest orientation (stationary relative to eye rotation)
        if self.eyelid_enabled:
            self.eyelid_trans[:3, :3] = value

    @property
    def rest_orientation(self) -> RotationMatrix:
        """Get the rest orientation (read-only).

        Returns reference orientation for eye rotation calculations.
        """
        return self._rest_orientation.copy()

    @property
    def current_target_point(self) -> Position3D | None:
        """Get the current target point (read-only).

        Returns the target position that was last used with look_at(), or None if
        the eye has not been oriented toward any target yet.
        """
        return self._current_target_point

    def set_rest_orientation_at_target(self, target_position: Position3D) -> None:
        """Set rest orientation so the VISUAL axis points to the target.

        Aligns the eye-local visual axis (derived from fovea angles when enabled,
        or equals the optical axis when disabled) with the world-space direction
        from eye position to target. Keeps +Y approximately aligned with world up.
        """
        # Eye position and target direction (world)
        eye_pos = np.array([self.trans[0, 3], self.trans[1, 3], self.trans[2, 3]], dtype=float)
        target_vec = np.array([target_position.x, target_position.y, target_position.z], dtype=float) - eye_pos
        norm = np.linalg.norm(target_vec)
        if norm < 1e-12:
            return
        z_world = target_vec / norm

        # Eye-local visual axis direction (unit, pointing outward toward cornea)
        if self.fovea_displacement:
            alpha = self.fovea_alpha_deg * np.pi / 180.0
            beta = self.fovea_beta_deg * np.pi / 180.0
            v_local = np.array(
                [
                    np.sin(alpha) * np.cos(beta),
                    np.sin(beta),
                    np.cos(alpha) * np.cos(beta),
                ],
                dtype=float,
            )
            v_local = -v_local / (np.linalg.norm(v_local) + 1e-12)
        else:
            v_local = np.array([0.0, 0.0, -1.0], dtype=float)

        # Build local visual basis
        y_local_pref = np.array([0.0, 1.0, 0.0], dtype=float)
        y_local = y_local_pref - np.dot(y_local_pref, v_local) * v_local
        if np.linalg.norm(y_local) < 1e-12:
            y_local = np.array([1.0, 0.0, 0.0], dtype=float)
            y_local -= np.dot(y_local, v_local) * v_local
        y_local /= np.linalg.norm(y_local)
        x_local = np.cross(y_local, v_local)
        x_local /= np.linalg.norm(x_local)

        local_rotation_matrix = np.column_stack([x_local, y_local, v_local])  # maps basis to canonical

        # Build world visual basis (target dir as z)
        world_up = np.array([0.0, 1.0, 0.0], dtype=float)
        y_world = world_up - np.dot(world_up, z_world) * z_world
        if np.linalg.norm(y_world) < 1e-12:
            world_up = np.array([1.0, 0.0, 0.0], dtype=float)
            y_world = world_up - np.dot(world_up, z_world) * z_world
        y_world /= np.linalg.norm(y_world)
        x_world = np.cross(y_world, z_world)
        x_world /= np.linalg.norm(x_world)
        world_rotation_matrix = np.column_stack([x_world, y_world, z_world])

        # Rotation mapping local visual basis to world visual basis
        rest_orientation = world_rotation_matrix @ local_rotation_matrix.T
        self.set_rest_orientation(RotationMatrix(rest_orientation))

    @property
    def position(self) -> Position3D:
        """Get/set the eye's position in world coordinates."""
        return Position3D.from_array(self.trans[:, 3])

    @position.setter
    def position(self, value: Position3D) -> None:
        """Set the eye's position and update transformation matrix."""
        self.trans[:, 3] = np.array(value)
        # Eyelid follows eye translation but not rotation
        if self.eyelid_enabled:
            self.eyelid_trans[0, 3] = value.x
            self.eyelid_trans[1, 3] = value.y
            self.eyelid_trans[2, 3] = value.z

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

    def find_cr(self, light: "Light", camera: "Camera") -> Position3D | None:
        """Finds the position of a corneal reflex.

        Delegates to reflections module for corneal reflection calculation.

        Args:
            light: Light source object
            camera: Camera object

        Returns:
            Position of corneal reflex, or None if not within cornea

        """
        return find_corneal_reflection(self, light, camera)

    def look_at(self, target_position: Position3D, legacy: bool = False) -> None:
        """Rotates an eye to look at a given position in space.

        Delegates to eye_operations module for gaze control.
        Updates current_target_point to track the target position.

        Args:
            target_position: Position in world coordinates to look at
            legacy: If True, uses optical-then-kappa method for backward compatibility

        """
        # Update current target point
        self._current_target_point = target_position

        if legacy:
            look_at_target_optical_then_kappa(self, target_position)
        else:
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

    def get_pupil_diameter(self) -> float:
        """Get current pupil diameter.

        Delegates to pupil object for diameter information.

        Returns:
            Pupil diameter in meters

        """
        return self.pupil.get_diameter()

    def set_pupil_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update decentration if enabled.

        Delegates to pupil object for radius modification, then applies
        decentration based on the new average diameter if decentration is enabled.

        Args:
            x_radius: Pupil radius in X direction (meters)
            y_radius: Pupil radius in Y direction (meters)

        """
        self.pupil.set_radii(x_radius, y_radius)
        self._update_pupil_position_with_decentration()

    def set_pupil_diameter(self, diameter: float) -> None:
        """Set pupil diameter and update decentration if enabled.

        Delegates to pupil object for diameter modification, then applies
        decentration based on the new diameter if decentration is enabled.

        Args:
            diameter: Pupil diameter in meters

        """
        self.pupil.set_diameter(diameter)
        self._update_pupil_position_with_decentration()

    def move_pupil_position(self, dx: float, dy: float, dz: float) -> None:
        """Move pupil position by given offset.

        Args:
            dx: X offset in meters
            dy: Y offset in meters
            dz: Z offset in meters

        """
        current_pos = self.pupil.pos_pupil
        new_pos = Position3D(current_pos.x + dx, current_pos.y + dy, current_pos.z + dz)
        self.pupil.pos_pupil = new_pos

    def set_pupil_position(self, x: float, y: float, z: float) -> None:
        """Set pupil position to absolute coordinates.

        Args:
            x: Absolute X position in meters
            y: Absolute Y position in meters
            z: Absolute Z position in meters

        """
        self.pupil.pos_pupil = Position3D(x, y, z)

    def get_pupil_center_in_world(self) -> Position3D:
        """Get pupil center in world coordinates.

        Delegates to pupil object for world coordinate transformation.

        Returns:
            Pupil center position in world coordinates

        """
        return self.pupil.get_center_world_coords(self.trans)

    def _calculate_decentration_offset(self, current_diameter: float) -> Position3D:
        """Calculate pupil decentration offset based on current diameter.

        Args:
            current_diameter: Current pupil diameter in meters

        Returns:
            Position3D offset for pupil decentration

        """
        if not self.decentration_config.enabled:
            return Position3D(0.0, 0.0, 0.0)

        # Auto-set baseline diameter if not specified
        if self.decentration_config.baseline_diameter is None:
            self.decentration_config.baseline_diameter = current_diameter

        # Get the decentration model
        model = PupilDecentrationRegistry.get_model(self.decentration_config.model_name)

        # Calculate offset using the model
        return model.calculate_offset(
            current_diameter=current_diameter,
            baseline_diameter=self.decentration_config.baseline_diameter,
            **self.decentration_config.get_model_params(),
        )

    def _update_pupil_position_with_decentration(self) -> None:
        """Update pupil position based on current size and decentration config."""
        if not self.decentration_config.enabled:
            return

        # Get base position from corneal geometry
        base_position = self.get_pupil_position()

        # Calculate decentration offset
        current_diameter = self.get_pupil_diameter()
        offset = self._calculate_decentration_offset(current_diameter)

        # Apply offset to base position
        new_position = Position3D(base_position.x + offset.x, base_position.y + offset.y, base_position.z + offset.z)

        self.pupil.pos_pupil = new_position

    def find_refracted_position(self, camera_position: Position3D, object_position: Position3D) -> Position3D | None:
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

        # Check if point is on visible cornea (within boundaries and not occluded by eyelid)
        if refraction_point is not None and not self.point_on_visible_cornea(refraction_point):
            refraction_point = None

        return refraction_point

    def point_within_eyelid(self, p: Position3D) -> bool:
        """Check if a point lies on eyelid skin (not the opening).

        Transforms the point to canonical eye coordinates (undo rest orientation)
        so that the eyelid remains fixed to the face regardless of eye rotation.
        Returns False if eyelid is not enabled or not present.
        """
        if self.eyelid is None:
            return False

        # Transform world point to eye position
        p_relative_to_eye = np.array(p) - np.array(self.position)

        # Transform to canonical eye coordinates (undo rest orientation)
        rest_inv = np.linalg.inv(self._rest_orientation)
        p_canonical = Position3D.from_array(rest_inv @ p_relative_to_eye[:3])

        # Do eyelid calculation in canonical space where +Y is anatomical up
        return self.eyelid.point_within_eyelid(p_canonical)

    def point_on_visible_cornea(self, p: Position3D) -> bool:
        """True if point lies within cornea and is not occluded by eyelid."""
        if not self.point_within_cornea(p):
            return False
        if self.eyelid is None:
            return True
        return not self.point_within_eyelid(p)

    @property
    def fovea_position(self) -> Position3D:
        """Calculate the 3D position of the fovea on the retinal surface.

        Uses spherical eye model with optional fovea displacement angles.
        Positions fovea at axial_length/2 distance from rotation center.

        Returns:
            Fovea position in eye coordinate system

        """
        # Retina distance from rotation center (from our eye model)
        retina_distance = self.axial_length / 2

        if self.fovea_displacement:
            # Convert displacement angles to radians
            alpha = self.fovea_alpha_deg * np.pi / 180.0  # Horizontal (temporal) displacement
            beta = self.fovea_beta_deg * np.pi / 180.0  # Vertical (upward) displacement

            # Calculate fovea position with displacement using spherical coordinates
            fovea_x = retina_distance * np.sin(alpha) * np.cos(beta)  # Temporal displacement
            fovea_y = retina_distance * np.sin(beta)  # Vertical displacement
            fovea_z = retina_distance * np.cos(alpha) * np.cos(beta)  # Along optical axis
        else:
            # Fovea at optical axis center (no displacement)
            fovea_x = 0.0
            fovea_y = 0.0
            fovea_z = retina_distance

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

    def get_pupil_in_camera_image(
        self, camera: "Camera", use_refraction: bool = True, center_method: str = "ellipse"
    ) -> tuple[np.ndarray | None, Point2D | None]:
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
            refracted_points: list[Position3D] = []
            for i in range(pupil_world.shape[1]):
                pupil_point = Position3D.from_array(pupil_world[:, i])
                refracted_point = self.find_refracted_position(camera.position, pupil_point)
                if refracted_point is not None:
                    # Convert Point3D result to Position3D for camera projection
                    refracted_points.append(refracted_point)

            # Project refracted points to camera image coordinates
            if refracted_points:
                projection_result = camera.project(refracted_points)

                # Convert valid boundary points to structured Point2D list
                valid_pupil_points = []
                for i in range(projection_result.image_points.shape[1]):
                    if projection_result.valid_mask[i] and not np.any(np.isnan(projection_result.image_points[:, i])):
                        point_2d = Point2D(
                            x=float(projection_result.image_points[0, i]),
                            y=float(projection_result.image_points[1, i]),
                        )
                        valid_pupil_points.append(point_2d)

                pupil_boundary_points = valid_pupil_points or None

                if not valid_pupil_points:
                    warnings.warn(
                        "No valid pupil points found in camera image (with refraction). Check camera-eye setup.",
                        UserWarning,
                        stacklevel=2,
                    )
            else:
                warnings.warn(
                    "No refracted pupil points could be computed. Check camera-eye setup.", UserWarning, stacklevel=2
                )
                pupil_boundary_points = None
        else:
            # Direct projection without refraction
            projection_result = camera.project([
                Position3D.from_array(pupil_world[:, i]) for i in range(pupil_world.shape[1])
            ])

            # Convert valid boundary points to structured Point2D list
            valid_pupil_points = []
            for i in range(projection_result.image_points.shape[1]):
                if projection_result.valid_mask[i] and not np.any(np.isnan(projection_result.image_points[:, i])):
                    point_2d = Point2D(
                        x=float(projection_result.image_points[0, i]), y=float(projection_result.image_points[1, i])
                    )
                    valid_pupil_points.append(point_2d)

            pupil_boundary_points = valid_pupil_points or None

            if not valid_pupil_points:
                warnings.warn(
                    "No valid pupil points found in camera image (without refraction). Check camera-eye setup.",
                    UserWarning,
                    stacklevel=2,
                )

        # Calculate pupil center using specified method
        pupil_center = None
        if pupil_boundary_points is not None:
            pupil_center = calculate_pupil_center_from_boundary(
                pupil_boundary_points, camera.camera_matrix.resolution, center_method
            )

        return pupil_boundary_points, pupil_center

    def __str__(self) -> str:
        """Basic string representation of the eye."""
        pos = self.position
        return f"Eye(pos=({pos.x * 1000:.1f}, {pos.y * 1000:.1f}, {pos.z * 1000:.1f})mm, axial_length={self.axial_length * 1000:.1f}mm)"

    def pprint(self) -> None:
        """Print detailed eye anatomy parameters in a formatted table."""
        cornea_center = self.cornea.center
        apex_pos = self.cornea.get_apex_position()
        pupil_pos = self.pupil.pos_pupil
        x_radius, _ = self.get_pupil_radii()

        data = [
            ["Anterior corneal radius R_a (mm)", f"{self.cornea.anterior_radius * 1000:.3f}"],
            ["Posterior corneal radius R_p (mm)", f"{self.cornea.posterior_radius * 1000:.3f}"],
            ["Axial length L (mm)", f"{self.axial_length * 1000:.3f}"],
            [
                "Cornea center to rotation center (mm)",
                f"{self.cornea.cornea_center_to_rotation_center_default * 1000:.3f}",
            ],
            ["Thickness offset t_offset (mm)", f"{self.cornea.thickness_offset * 1000:.3f}"],
            ["Corneal depth d_c (mm)", f"{self.cornea.get_corneal_depth() * 1000:.3f}"],
            ["Refractive index n_cornea", f"{self.cornea.refractive_index:.3f}"],
            ["Refractive index n_aqueous", f"{self.n_aqueous_humor:.3f}"],
            ["Fovea α (deg)", f"{self.fovea_alpha_deg:.1f}"],
            ["Fovea β (deg)", f"{self.fovea_beta_deg:.1f}"],
            ["Angle κ (deg)", f"{self.angle_kappa:.3f}"],
            [
                "Cornea center (x,y,z) mm",
                f"({cornea_center.x * 1000:.3f}, {cornea_center.y * 1000:.3f}, {cornea_center.z * 1000:.3f})",
            ],
            [
                "Anterior apex (x,y,z) mm",
                f"({apex_pos.x * 1000:.3f}, {apex_pos.y * 1000:.3f}, {apex_pos.z * 1000:.3f})",
            ],
            [
                "Pupil center (x,y,z) mm",
                f"({pupil_pos.x * 1000:.3f}, {pupil_pos.y * 1000:.3f}, {pupil_pos.z * 1000:.3f})",
            ],
            ["Pupil radius r_p (mm)", f"{x_radius * 1000:.3f}"],
        ]

        headers = ["Parameter", "Value"]
        print("Eye Anatomy Parameters:")
        print(tabulate(data, headers=headers, tablefmt="grid"))

    def serialize(self) -> dict:
        """Serialize eye state to a dictionary.

        Returns complete eye state that can be used to perfectly reconstruct
        the eye object in its exact current state.

        Returns:
            Dictionary containing all eye parameters and current state

        """
        # Core anatomical parameters
        data = {
            # Position and orientation
            "position": self.position.serialize() if self.position else None,
            "transformation_matrix": self.trans.tolist(),
            "rest_orientation": self._rest_orientation.tolist(),
            "current_target_point": self._current_target_point.serialize() if self._current_target_point else None,
            # Anatomical parameters
            "axial_length": float(self.axial_length),
            "n_aqueous_humor": float(self.n_aqueous_humor),
            "fovea_displacement": bool(self.fovea_displacement),
            "fovea_alpha_deg": float(self.fovea_alpha_deg),
            "fovea_beta_deg": float(self.fovea_beta_deg),
            # Pupil configuration
            "pupil_type": self.pupil_type,
            "pupil_boundary_points": self.pupil_boundary_points,
            "pupil_random_seed": self.pupil_random_seed,
            # Eyelid configuration
            "eyelid_enabled": bool(self.eyelid_enabled),
            "eyelid_transformation_matrix": self.eyelid_trans.tolist(),
        }

        # Serialize cornea
        if self.cornea:
            data["cornea"] = self.cornea.serialize()

        # Serialize pupil state
        if self.pupil:
            data["pupil"] = self.pupil.serialize()

        # Serialize eyelid if enabled
        if self.eyelid and self.eyelid_enabled:
            data["eyelid"] = self.eyelid.serialize()

        return data

    @classmethod
    def deserialize(cls, data: dict) -> "Eye":
        """Reconstruct eye from serialized data.

        Creates a new Eye instance and restores it to the exact state
        captured in the serialized data.

        Args:
            data: Dictionary from serialize() method

        Returns:
            Eye instance in the exact state when serialized

        """
        # Create new eye with basic configuration
        eye = cls(
            fovea_displacement=data["fovea_displacement"],
            fovea_alpha_deg=data["fovea_alpha_deg"],
            fovea_beta_deg=data["fovea_beta_deg"],
            pupil_type=data["pupil_type"],
            pupil_boundary_points=data["pupil_boundary_points"],
            pupil_random_seed=data["pupil_random_seed"],
            eyelid_enabled=data["eyelid_enabled"],
        )

        # Restore position and orientation
        if data["position"]:
            eye.position = Position3D.deserialize(data["position"])

        eye.trans = TransformationMatrix(np.array(data["transformation_matrix"]))
        eye._rest_orientation = RotationMatrix.deserialize(data["rest_orientation"])
        eye.eyelid_trans = TransformationMatrix(np.array(data["eyelid_transformation_matrix"]))

        if data["current_target_point"]:
            eye._current_target_point = Position3D.deserialize(data["current_target_point"])

        # Restore anatomical parameters
        eye.axial_length = data["axial_length"]
        eye.n_aqueous_humor = data["n_aqueous_humor"]

        # Restore cornea
        if data.get("cornea"):
            eye.cornea = SphericalCornea.deserialize(data["cornea"])

        # Restore pupil
        if data.get("pupil"):
            eye.pupil = EllipticalPupil.deserialize(data["pupil"])

        # Restore eyelid if enabled
        if "eyelid" in data and data["eyelid"] and eye.eyelid_enabled:
            eye.eyelid = Eyelid.deserialize(data["eyelid"])

        return eye
