"""Main eye model for eye tracking simulation.

Defines the Eye class, integrating cornea, pupil, fovea displacement, and gaze mechanics for simulation and analysis.
"""

import copy
import warnings
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from pyetsimul.log import info, table

from ..optics.pupil_imaging import calculate_pupil_center_from_boundary, calculate_pupil_diameter_from_boundary
from ..optics.reflections import find_corneal_reflection
from ..optics.refractions import find_refraction_point
from ..types import Direction3D, Point2D, Position3D, PupilData, Ray, RotationMatrix, TransformationMatrix
from .cornea import ConicCornea, SphericalCornea
from .default_configs import EyeAnatomyDefaults
from .eye_model import EyeModel, get_eye_model
from .eye_operations import look_at_target, look_at_target_line_of_sight, look_at_target_optical_then_kappa
from .eyelid import Eyelid, create_eyelid
from .off_axis_pupil import OffAxisPupilConfig
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

    # The eye-model specification. Accepts an EyeModel or a registered model name (case-insensitive,
    # e.g. "PyEtSimul"); a name is resolved to its EyeModel in __post_init__. Read its parameters as
    # ``self.model.<field>``.
    model: str | EyeModel = field(default_factory=EyeModel)
    # "left" or "right": identity, not a model property. The fovea is temporal in both eyes, so the
    # horizontal kappa flips sign between them and the pupil-decentration coefficients are mirrored;
    # which_eye applies both signs to the model's unsigned magnitudes.
    which_eye: str = "right"

    # Per-eye working instances, built in __post_init__ from the (shared, frozen) model spec. The model
    # holds the spec; these are this eye's mutable working copies -- the cornea is set up for the axial
    # length, the pupil-centre configs are resolved per side and accrue a baseline, the pupil and eyelid
    # are live. ``self.cornea`` is this eye's working cornea; ``self.model.cornea`` is the pristine spec.
    cornea: SphericalCornea | ConicCornea = field(init=False)
    decentration_config: PupilDecentrationConfig = field(init=False)
    off_axis_pupil: OffAxisPupilConfig = field(init=False)
    pupil: Pupil = field(init=False)  # live pupil object; its size and shape are dynamic state
    eyelid: Eyelid | None = field(init=False, default=None)

    # Placement and dynamic state, calculated in __post_init__.
    trans: TransformationMatrix = field(init=False)
    # Eyelid transform (local→world): follows eye position but keeps a fixed orientation
    eyelid_trans: TransformationMatrix = field(init=False, repr=False)
    _rest_orientation: RotationMatrix = field(init=False)
    _rest_orientation_explicitly_set: bool = field(init=False, default=False)  # Track if user set rest orientation
    _current_target_point: Position3D | None = field(init=False, default=None)  # Updated by look_at()
    # Rest globe-centre placement; the gaze-dependent rotation centre re-pivots about it.
    _placement: Position3D = field(init=False)
    # {pupil diameter (mm): physical eye-local boundary (N x 2)} registered via register_pupil_shape;
    # set_pupil_diameter swaps to the matching shape when called with a registered size.
    _size_shapes: dict[float, np.ndarray] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        """Builds this eye's working state from its model specification.

        The frozen model is the shared spec; the eye builds its own mutable working copies of the cornea
        and the pupil-centre configs (set up / resolved for this eye), plus a live pupil and eyelid, so two
        eyes built from one model never mutate each other's geometry. Pupil size is scaled by the corneal
        scaling factor.
        """
        # Resolve a registered model name (case-insensitive) to its EyeModel.
        if isinstance(self.model, str):
            self.model = get_eye_model(self.model)
        model = self.model

        # Per-eye working copies of the mutable model components; the model spec stays pristine and shared.
        self.cornea = copy.deepcopy(model.cornea)
        self.decentration_config = copy.deepcopy(model.decentration_config)
        self.off_axis_pupil = copy.deepcopy(model.off_axis_pupil)

        # Setup cornea-specific geometry (this handles all sphere-specific scaling)
        self.cornea.setup_eye_geometry(model.axial_length)

        # Initialize transformation matrix (identity at rest position)
        self._rest_orientation = RotationMatrix.identity()
        self.trans = TransformationMatrix.from_rotation(self._rest_orientation)

        # Initialize eyelid transform (same position as eye, orientation = rest)
        self.eyelid_trans = TransformationMatrix.from_translation_and_rotation(
            self.trans.get_translation(), self._rest_orientation
        )

        # Create pupil object - calculate pupil position and scale radius. The model's default (nominal)
        # diameter sets the starting size; set_pupil_diameter changes it later.
        pupil_position = self.get_pupil_position()
        # Scale pupil radius based on corneal scaling factor
        scale = self.cornea.get_scale_factor()
        scaled_pupil_radius = (model.default_pupil_diameter / 2.0) * scale
        x_pupil = Direction3D(scaled_pupil_radius, 0, 0)
        y_pupil = Direction3D(0, scaled_pupil_radius, 0)

        # Create pupil with optional n parameter and random seed
        pupil_kwargs = {}
        if model.pupil_boundary_points is not None:
            pupil_kwargs["n"] = model.pupil_boundary_points

        # For realistic pupils, use provided params or create from random seed
        if model.pupil_type == "realistic":
            if model.realistic_pupil_params is not None:
                pupil_kwargs["params"] = model.realistic_pupil_params
            elif model.pupil_random_seed is not None:
                pupil_kwargs["params"] = RealisticPupilParams(random_seed=model.pupil_random_seed)

        self.pupil = create_pupil(
            pupil_type=model.pupil_type, pos_pupil=pupil_position, x_pupil=x_pupil, y_pupil=y_pupil, **pupil_kwargs
        )

        # Create eyelid if enabled: the eyeball globe (radius axial_length/2) centred on the eyeball
        # centre (axial_length/2 behind the corneal apex), phi_max set so its front-cap rim meets the
        # corneal limbus.
        if model.eyelid_enabled:
            sphere_radius = model.axial_length / 2.0
            apex_pos = self.cornea.get_apex_position()
            eyeball_center_z = self.eyeball_center_z
            # Limbus z relative to the eyeball centre; phi from the apex pole (-Z from the centre):
            # cos(phi) = -limbus_z_rel / sphere_radius -> phi = arccos(...).
            limbus_z_rel = apex_pos.z + self.cornea.get_corneal_depth() - eyeball_center_z
            ratio = np.clip(-limbus_z_rel / sphere_radius, -1.0, 1.0)
            phi_max = float(np.arccos(ratio))

            self.eyelid = create_eyelid(
                center=Position3D(0.0, 0.0, eyeball_center_z),
                sphere_radius=sphere_radius,
                phi_max=phi_max,
                openness=EyeAnatomyDefaults.EYELID_OPENNESS,
            )
            # Keep eyelid orientation locked to rest orientation (fixed to face)
            self.eyelid_trans[:3, :3] = self._rest_orientation

        # Resolve the population defaults for the pupil-centre models (no-ops when the values were given
        # or the model is disabled), then position the pupil from whichever models are enabled.
        self.off_axis_pupil.resolve_defaults()
        if self.decentration_config.enabled:
            self.decentration_config.resolve_for_eye(self.which_eye)
            if self.decentration_config.baseline_diameter is None:
                self.decentration_config.baseline_diameter = self.get_pupil_diameter()
        if self.decentration_config.enabled or self.off_axis_pupil.enabled:
            self._update_pupil_position()

        # Capture the rest placement (globe centre) that the gaze-dependent rotation centre pivots about.
        self._placement = self.position

    @property
    def _signed_alpha_rad(self) -> float:
        """Horizontal fovea angle in radians, signed by eye side.

        The fovea is temporal in both eyes, so the angle points the opposite horizontal way for the left
        eye; ``fovea_alpha_deg`` carries the magnitude.
        """
        return np.radians((-1.0 if self.which_eye == "left" else 1.0) * self.model.fovea_alpha_deg)

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
        self._rest_orientation_explicitly_set = True
        self.trans[:3, :3] = value

        # Keep eyelid orientation aligned to rest orientation (stationary relative to eye rotation)
        if self.model.eyelid_enabled:
            self.eyelid_trans[:3, :3] = value

    @property
    def rest_orientation(self) -> RotationMatrix:
        """Get the rest orientation (read-only).

        Returns reference orientation for eye rotation calculations.
        """
        return self._rest_orientation.copy()

    @property
    def placement(self) -> Position3D:
        """Get the commanded eye position (read-only).

        The pivot anchor for the gaze-direction-dependent rotation centre. It differs from
        :attr:`position` after a re-pivot: ``position`` is the globe centre in world coordinates, while
        ``placement`` is the position the eye was commanded to and about which it re-pivots when turning
        to off-axis targets.
        """
        return self._placement

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
        if self.model.fovea_displacement:
            alpha = self._signed_alpha_rad
            beta = self.model.fovea_beta_deg * np.pi / 180.0
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
        """Get/set the world position of the eye-local origin.

        The origin is the corneal apex under the "apex" placement convention and the eyeball centre
        under the "center" convention.
        """
        return Position3D.from_array(self.trans[:, 3])

    @position.setter
    def position(self, value: Position3D) -> None:
        """Set the eye's position and update transformation matrix."""
        self.trans[:, 3] = np.array(value)
        # Record the placement as the rest origin so look_at can re-pivot about a gaze-dependent
        # rotation centre relative to it (look_at writes trans directly, not here).
        self._placement = value
        # Eyelid follows eye translation but not rotation
        if self.model.eyelid_enabled:
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

    def look_at(self, target_position: Position3D, method: str | None = None) -> None:
        """Rotates an eye to look at a given position in space.

        Delegates to eye_operations module for gaze control.
        Updates current_target_point to track the target position.

        Args:
            target_position: Position in world coordinates to look at
            method: Look-at method for this call, overriding the eye model's default ``method``.
                "visual_axis" (fovea through the eye centre), "line_of_sight" (fovea through the pupil
                centre), or "optical_then_kappa" (optical axis then a kappa post-rotation; the Böhme 2008
                et_simul construction). None uses the eye model's default.

        """
        # Update current target point
        self._current_target_point = target_position

        method = method if method is not None else self.model.look_at_method
        if method == "line_of_sight":
            look_at_target_line_of_sight(self, target_position)
        elif method == "optical_then_kappa":
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

        # Tilt the pupil plane (z-shear about the pupil centre) before the world transform, so the disc is
        # eye-fixed and rotates with gaze. A flat plane keeps every point at pos_pupil.z; the shear gives
        # each point a z-offset linear in its in-plane offset from the centre.
        if self.model.pupil_tilt_x_deg or self.model.pupil_tilt_y_deg:
            cx, cy = self.pupil.pos_pupil.x, self.pupil.pos_pupil.y
            dx = pupil_points[0, :] - cx
            dy = pupil_points[1, :] - cy
            pupil_points[2, :] += dx * np.tan(np.radians(self.model.pupil_tilt_x_deg)) + dy * np.tan(
                np.radians(self.model.pupil_tilt_y_deg)
            )

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
            Tuple of (x_radius, y_radius) in mm

        """
        return self.pupil.get_radii()

    def get_pupil_diameter(self) -> float:
        """Get current pupil diameter.

        Delegates to pupil object for diameter information.

        Returns:
            Pupil diameter in mm

        """
        return self.pupil.get_diameter()

    def set_pupil_radii(self, x_radius: float, y_radius: float) -> None:
        """Set pupil radii and update decentration if enabled.

        Delegates to pupil object for radius modification, then applies
        decentration based on the new average diameter if decentration is enabled.

        Args:
            x_radius: Pupil radius in X direction (mm)
            y_radius: Pupil radius in Y direction (mm)

        """
        self.pupil.set_radii(x_radius, y_radius)
        self._update_pupil_position()

    def corneal_magnification(self, camera: "Camera") -> float:
        """Ratio of the apparent (refracted) to the physical pupil size, for this eye's cornea.

        The pupil boundary is projected to ``camera`` with and without refraction and the size ratio
        returned. Dimensionless, set by the cornea's radius and conic constant.

        Args:
            camera: Camera the pupil is viewed through.

        Returns:
            Apparent / physical pupil-size ratio.

        Raises:
            ValueError: If the pupil does not project into the camera image with and without refraction.

        """
        refracted, _ = self.get_pupil_in_camera_image(camera, use_refraction=True)
        direct, _ = self.get_pupil_in_camera_image(camera, use_refraction=False)
        if not refracted or not direct:
            raise ValueError("corneal_magnification: pupil did not project into the camera image.")

        def span(points: list[Point2D]) -> float:
            xs = [p.x for p in points]
            ys = [p.y for p in points]
            return ((max(xs) - min(xs)) + (max(ys) - min(ys))) / 2.0

        return span(refracted) / span(direct)

    def _physical_diameter_for_apparent(
        self, apparent_diameter: float, camera: "Camera", max_iter: int = 8, tol: float = 1e-4
    ) -> float:
        """Physical pupil diameter whose refracted projection has ``apparent_diameter`` in ``camera``.

        The corneal magnification varies with pupil size -- the conic cornea does not scale the pupil by a
        single constant -- so a one-shot ``apparent_diameter / corneal_magnification`` under-sets the physical
        pupil (the magnification is measured at the apparent size, not at the smaller physical size it should
        be measured at). This iterates the fixed point ``physical = apparent_diameter /
        corneal_magnification(physical)``, re-measuring the magnification at each candidate physical size until
        the diameter settles to within ``tol`` mm. The magnification changes slowly with size, so it converges
        in a few iterations.

        Args:
            apparent_diameter: Target apparent (camera) pupil diameter in mm.
            camera: Camera the apparent size is measured through.
            max_iter: Maximum fixed-point iterations.
            tol: Convergence tolerance on the physical diameter in mm.

        Returns:
            Physical pupil diameter in mm.

        """
        physical = float(apparent_diameter)
        for _ in range(max_iter):
            self.pupil.set_diameter(physical)
            self._update_pupil_position()
            updated = apparent_diameter / self.corneal_magnification(camera)
            if abs(updated - physical) <= tol:
                return updated
            physical = updated
        return physical

    def set_pupil_diameter(self, diameter: float, apparent: bool = False, camera: "Camera | None" = None) -> None:
        """Set pupil diameter and update decentration if enabled.

        Delegates to pupil object for diameter modification, then applies
        decentration based on the new diameter if decentration is enabled.

        Args:
            diameter: Pupil diameter in mm. By default this is the physical pupil behind the cornea.
            apparent: If True, ``diameter`` is the apparent (camera) pupil size seen *through* the cornea;
                the physical pupil that the cornea refracts back up to ``diameter`` is recovered exactly by
                ``_physical_diameter_for_apparent`` (iterating the size-dependent corneal magnification), so
                the rendered apparent pupil matches ``diameter``. Requires ``camera``.
            camera: Camera used to measure the corneal magnification when ``apparent`` is True.

        Raises:
            ValueError: If ``apparent`` is True but no camera is given.

        """
        # A pupil's shape can vary with its size; if a boundary is registered for this diameter
        # (register_pupil_shape), adopt it before scaling so the shape matches the size.
        if diameter in self._size_shapes:
            self.pupil = create_pupil(
                "contour", self.get_pupil_position(), None, None, contour=self._size_shapes[diameter]
            )
        if apparent:
            if camera is None:
                raise ValueError("set_pupil_diameter(apparent=True) requires a camera to un-refract the size.")
            diameter = self._physical_diameter_for_apparent(diameter, camera)
        self.pupil.set_diameter(diameter)
        self._update_pupil_position()

    def _pupil_centre_and_diameter(self, camera: "Camera", center_method: str) -> tuple[np.ndarray, float]:
        """The refracted pupil's ``center_method`` centre (px) and mean-axis ellipse diameter (px) in ``camera``."""
        boundary, _ = self.get_pupil_in_camera_image(camera, use_refraction=True)
        if not boundary:
            raise ValueError("solve_decentration_from_apparent: pupil did not project into the camera image.")
        centre = calculate_pupil_center_from_boundary(boundary, camera.camera_matrix.resolution, center_method)
        diameter = calculate_pupil_diameter_from_boundary(boundary)
        return np.array([centre.x, centre.y]), float(diameter)

    def solve_decentration_from_apparent(
        self,
        image_cx: float,
        image_cy: float,
        apparent_con: float,
        apparent_dil: float,
        camera: "Camera",
        target: Position3D | None = None,
        center_method: str = "ellipse",
        max_iter: int = 12,
        tol: float = 1e-5,
    ) -> tuple[float, float]:
        """Eye-local decentration coefficients whose apparent render reproduces a measured image decentration.

        The pupil centre shifts as the pupil changes size; in the camera image that is a centre displacement
        per unit of apparent pupil-size change. ``image_cx, image_cy`` are that measured shift normalised by
        the apparent pupil-diameter change (centre-shift px / mean-ellipse-axis-change px, so dimensionless),
        at the ``apparent_con`` -> ``apparent_dil`` transition. The decentration is applied as an eye-local
        pupil offset proportional to the physical diameter change; the image projection of that offset is
        non-linear over the sub-millimetre shift, so a single linear inversion mis-scales it by several
        percent. This renders the pupil with apparent sizing (adopting any registered per-size shape) at the
        two diameters, measures the ``center_method`` pupil-centre shift over the mean-ellipse-axis diameter
        change, and Newton-iterates the eye-local coefficients until the rendered ratio matches
        ``(image_cx, image_cy)`` -- a 1:1 round-trip, converging in a few iterations.

        ``center_method`` must match the pupil-centre method used downstream (e.g. by the gaze model), so the
        inversion is consistent with how the centre is later read. The returned coefficients are NOT set on the
        eye -- the caller assigns them to a ``PupilDecentrationConfig`` (this eye's ``decentration_config`` or a
        per-target one), since the solve leaves the config at an intermediate probe value.

        Args:
            image_cx: Measured image decentration x (centre-shift px / pupil-diameter-change px).
            image_cy: Measured image decentration y.
            apparent_con: First (constricted) apparent pupil diameter in mm.
            apparent_dil: Second (dilated) apparent pupil diameter in mm.
            camera: Camera the apparent decentration was measured in.
            target: Gaze the decentration was measured at; defaults to the world origin.
            center_method: Pupil-centre method ("ellipse", "convex_hull", "center_of_mass").
            max_iter: Maximum Newton iterations.
            tol: Convergence tolerance on the image-frame ratio.

        Returns:
            ``(x_coeff, y_coeff)`` eye-local decentration coefficients.

        """
        self.look_at(target if target is not None else Position3D(0.0, 0.0, 0.0))
        self.decentration_config.enabled = True

        def rendered_ratio(ex: float, ey: float) -> np.ndarray:
            self.decentration_config.x_coeff, self.decentration_config.y_coeff = ex, ey
            self.set_pupil_diameter(apparent_con, apparent=True, camera=camera)
            con_centre, con_diameter = self._pupil_centre_and_diameter(camera, center_method)
            self.set_pupil_diameter(apparent_dil, apparent=True, camera=camera)
            dil_centre, dil_diameter = self._pupil_centre_and_diameter(camera, center_method)
            return (dil_centre - con_centre) / (dil_diameter - con_diameter)

        measured = np.array([image_cx, image_cy])
        ex, ey = image_cx, image_cy  # the image-frame value is a good first coefficient
        eps = 0.005
        for _ in range(max_iter):
            current = rendered_ratio(ex, ey)
            residual = measured - current
            if float(np.hypot(*residual)) < tol:
                break
            jacobian = np.column_stack([
                (rendered_ratio(ex + eps, ey) - current) / eps,
                (rendered_ratio(ex, ey + eps) - current) / eps,
            ])
            step = np.linalg.solve(jacobian, residual)
            ex, ey = ex + float(step[0]), ey + float(step[1])
        return float(ex), float(ey)

    def move_pupil_position(self, dx: float, dy: float, dz: float) -> None:
        """Move pupil position by given offset.

        Args:
            dx: X offset in mm
            dy: Y offset in mm
            dz: Z offset in mm

        """
        current_pos = self.pupil.pos_pupil
        new_pos = Position3D(current_pos.x + dx, current_pos.y + dy, current_pos.z + dz)
        self.pupil.pos_pupil = new_pos

    def set_pupil_position(self, x: float, y: float, z: float) -> None:
        """Set pupil position to absolute coordinates.

        Args:
            x: Absolute X position in mm
            y: Absolute Y position in mm
            z: Absolute Z position in mm

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
            current_diameter: Current pupil diameter in mm

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

    def _update_pupil_position(self) -> None:
        """Position the pupil centre.

        The position is the optical-axis base plus the static off-axis offset and the size-dependent
        decentration, each applied only when its config is enabled.
        """
        base = self.get_pupil_position()
        offset = self.off_axis_pupil.offset_for_eye(self.which_eye)
        if self.decentration_config.enabled:
            decentration = self._calculate_decentration_offset(self.get_pupil_diameter())
            offset = Position3D(offset.x + decentration.x, offset.y + decentration.y, offset.z + decentration.z)
        self.pupil.pos_pupil = Position3D(base.x + offset.x, base.y + offset.y, base.z + offset.z)

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
        refraction_point = find_refraction_point(
            self.cornea, self.trans, camera_position, object_position, self.model.n_aqueous_humor
        )

        # Check if point is on visible cornea (within boundaries and not occluded by eyelid)
        if refraction_point is not None and not self.point_on_visible_cornea(refraction_point):
            refraction_point = None

        return refraction_point

    def _backproject_image_point(self, image_point: Point2D, camera: "Camera") -> tuple[float, float] | None:
        """Eye-local (x, y) on the pupil plane that a camera image point maps to, through the cornea.

        Inverse of the corneal projection used by get_pupil_in_camera_image for a single point: the image
        point is unprojected to a 3D ray, refracted inward through the cornea, and intersected with the
        pupil plane. The ray is carried in eye-local coordinates, where the conic axis is aligned with the
        Z-axis, so the returned coordinates are directly usable as pupil-plane offsets.

        Args:
            image_point: Boundary point in the camera image (Point2D, center-origin).
            camera: Camera that produced the image point.

        Returns:
            Eye-local (x, y) on the pupil plane, or None if the back-projected ray does not pass through.

        """
        inv_transform = np.linalg.inv(self.trans)
        far_world = camera.unproject(image_point, 1.0)  # any positive distance; only the ray direction matters
        camera_local = Position3D.from_array(
            inv_transform @ np.array([camera.position.x, camera.position.y, camera.position.z, 1.0])
        )
        far_local = inv_transform @ np.array([far_world.x, far_world.y, far_world.z, 1.0])
        direction = Direction3D(
            far_local[0] - camera_local.x, far_local[1] - camera_local.y, far_local[2] - camera_local.z
        ).normalize()
        ray = Ray(origin=camera_local.to_point3d(), direction=direction)

        refracted = self.cornea.refract_ray_inward(ray, 1.0, self.cornea.refractive_index, self.model.n_aqueous_humor)
        if refracted is None:
            return None

        pupil_z = self.pupil.pos_pupil.z
        if abs(refracted.direction.z) < 1e-9:
            return None
        t = (pupil_z - refracted.origin.z) / refracted.direction.z
        point = refracted.point_at(t)
        return point.x, point.y

    def backproject_contour(
        self, boundary: "list[Point2D] | np.ndarray", camera: "Camera | None" = None, apparent: bool = True
    ) -> np.ndarray:
        """Physical eye-local pupil-plane boundary (N x 2) for a measured pupil boundary.

        When apparent is True, each image point is back-projected through this eye's cornea (the inverse
        of the corneal projection) at the eye's current orientation; when False, the boundary is already
        eye-local pupil-plane points and is returned unchanged. Raises if any apparent point does not
        pass back through the cornea (no silent dropping).

        Args:
            boundary: A list of Point2D in the camera image (apparent), or eye-local (x, y) points (N x 2).
            camera: Camera that produced the boundary, required when apparent is True.
            apparent: Whether boundary is in the camera image (True) or already eye-local (False).

        """
        if not apparent:
            return np.asarray(boundary, dtype=float).reshape(-1, 2)
        local = [self._backproject_image_point(p, camera) for p in boundary]
        if any(point is None for point in local):
            raise ValueError("Pupil contour back-projection failed for one or more boundary points.")
        return np.array(local, dtype=float)

    def set_pupil_from_contour(
        self, boundary: "list[Point2D] | np.ndarray", camera: "Camera | None" = None, apparent: bool = True
    ) -> None:
        """Replace the pupil with a ContourPupil built from a measured boundary.

        The shape (boundary relative to its center) is taken from the measurement; pupil size and
        decentration remain governed by set_pupil_diameter and the decentration config, so a later
        set_pupil_diameter rescales this contour without changing its shape.
        """
        contour = self.backproject_contour(boundary, camera, apparent=apparent)
        self.pupil = create_pupil("contour", self.get_pupil_position(), None, None, contour=contour)

    def register_pupil_shape(
        self,
        diameter: float,
        boundary: "list[Point2D] | np.ndarray",
        camera: "Camera | None" = None,
        apparent: bool = False,
    ) -> None:
        """Register a per-size physical pupil shape adopted automatically by set_pupil_diameter(diameter).

        The boundary is back-projected to physical eye-local coordinates (when apparent, at the eye's
        current orientation) and stored under ``diameter`` -- the size set_pupil_diameter is later called
        with. When the pupil is set to that size, its shape is taken from this boundary; the shape is
        eye-fixed and rotates with the eye toward any target.
        """
        self._size_shapes[float(diameter)] = self.backproject_contour(boundary, camera, apparent=apparent)

    def point_within_eyelid(self, p: Position3D) -> bool:
        """Check if a point lies on eyelid skin (not the opening).

        Transforms the point to canonical eye coordinates (undo rest orientation)
        so that the eyelid remains fixed to the face regardless of eye rotation.
        Returns False if eyelid is not enabled or not present.
        """
        if self.eyelid is None:
            return False

        # Fail if rest orientation hasn't been set - eyelid checks will be incorrect
        if not self._rest_orientation_explicitly_set:
            raise ValueError(
                "Eyelid occlusion check requires rest orientation to be set. "
                "Call set_rest_orientation_at_target(target) before using eyelid features."
            )

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

    def point_within_cornea_batch(self, points: np.ndarray) -> np.ndarray:
        """Whether each of ``(N, 3)`` world points lies within the corneal boundary; returns an ``(N,)`` mask.

        Carries every point into eye coordinates at once and checks its projected depth from the apex along
        the apex-to-centre axis against the corneal depth. NaN rows (no refraction) fall through as False.
        """
        inv = np.linalg.inv(np.asarray(self.trans))
        local = (np.column_stack([points, np.ones(len(points))]) @ inv.T)[:, :3]
        apex = self.cornea.get_apex_position()
        center = self.cornea.center
        axis = np.array([center.x - apex.x, center.y - apex.y, center.z - apex.z])
        diff = local - np.array([apex.x, apex.y, apex.z])
        projection = diff @ axis / np.linalg.norm(axis)
        return projection < self.cornea.get_corneal_depth()

    def point_on_visible_cornea_batch(self, points: np.ndarray) -> np.ndarray:
        """Whether each of ``(N, 3)`` world points is within the cornea and not occluded by the eyelid."""
        within = self.point_within_cornea_batch(points)
        if self.eyelid is None:
            return within
        # Eyelid present (rare): check occlusion only for the within-cornea points.
        for i in np.nonzero(within)[0]:
            if self.point_within_eyelid(Position3D(points[i, 0], points[i, 1], points[i, 2])):
                within[i] = False
        return within

    @property
    def eyeball_center_z(self) -> float:
        """Eye-local z of the eyeball (globe) centre, axial_length/2 behind the corneal apex.

        Derived from the live apex position, so it is convention-agnostic: 0 when the eye-local
        origin is the eyeball centre ("center" convention) and axial_length/2 when the origin is the
        corneal apex ("apex" convention).
        """
        return self.cornea.get_apex_position().z + self.model.axial_length / 2.0

    @property
    def fovea_position(self) -> Position3D:
        """Calculate the 3D position of the fovea on the retinal surface.

        The fovea sits on the retina, axial_length/2 behind the eyeball centre, with the x/y offset
        set by the optional fovea-displacement (kappa) angles.

        Returns:
            Fovea position in eye coordinate system

        """
        # Retina radius (from the eyeball centre) and the eyeball centre in eye-local coordinates.
        retina_distance = self.model.axial_length / 2
        eyeball_center_z = self.eyeball_center_z

        if self.model.fovea_displacement:
            # Convert displacement angles to radians
            alpha = self._signed_alpha_rad  # horizontal (temporal); sign set by eye side
            beta = self.model.fovea_beta_deg * np.pi / 180.0  # Vertical (upward) displacement

            # Calculate fovea position with displacement using spherical coordinates
            fovea_x = retina_distance * np.sin(alpha) * np.cos(beta)  # Temporal displacement
            fovea_y = retina_distance * np.sin(beta)  # Vertical displacement
            fovea_z = eyeball_center_z + retina_distance * np.cos(alpha) * np.cos(beta)  # Along optical axis
        else:
            # Fovea at optical axis center (no displacement)
            fovea_x = 0.0
            fovea_y = 0.0
            fovea_z = eyeball_center_z + retina_distance

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

        # Visual axis = the direction from the eyeball centre (the nodal point) to the fovea, which
        # depends only on the kappa angles, independent of where the eye-local origin sits.
        eyeball_center_z = self.eyeball_center_z
        visual_axis = Direction3D(fovea_pos.x, fovea_pos.y, fovea_pos.z - eyeball_center_z).normalize()

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
            # Refract the whole pupil boundary at once (vectorised), then keep the points that land on the
            # visible cornea -- equivalent to per-point find_refracted_position, without the Python loop.
            objects = pupil_world[:3, :].T  # (N, 3) world pupil-boundary points
            refracted_world, valid = self.cornea.find_refraction_batch(
                camera.position, objects, 1.0, self.cornea.refractive_index, self.trans, self.model.n_aqueous_humor
            )
            keep = valid & self.point_on_visible_cornea_batch(refracted_world)
            refracted_points: list[Position3D] = [
                Position3D(refracted_world[i, 0], refracted_world[i, 1], refracted_world[i, 2])
                for i in np.nonzero(keep)[0]
            ]

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
        return f"Eye(pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})mm, axial_length={self.model.axial_length:.1f}mm)"

    def pprint(self) -> None:
        """Print detailed eye anatomy parameters in a formatted table."""
        cornea_center = self.cornea.center
        apex_pos = self.cornea.get_apex_position()
        pupil_pos = self.pupil.pos_pupil
        x_radius, _ = self.get_pupil_radii()

        data = [
            ["Anterior corneal radius R_a (mm)", f"{self.cornea.anterior_radius:.3f}"],
            ["Posterior corneal radius R_p (mm)", f"{self.cornea.posterior_radius:.3f}"],
            ["Axial length L (mm)", f"{self.model.axial_length:.3f}"],
            [
                "Cornea center to rotation center (mm)",
                f"{self.cornea.cornea_center_to_rotation_center_default:.3f}",
            ],
            ["Thickness offset t_offset (mm)", f"{self.cornea.thickness_offset:.3f}"],
            ["Corneal depth d_c (mm)", f"{self.cornea.get_corneal_depth():.3f}"],
            ["Refractive index n_cornea", f"{self.cornea.refractive_index:.3f}"],
            ["Refractive index n_aqueous", f"{self.model.n_aqueous_humor:.3f}"],
            ["Fovea α (deg)", f"{self.model.fovea_alpha_deg:.1f}"],
            ["Fovea β (deg)", f"{self.model.fovea_beta_deg:.1f}"],
            ["Angle κ (deg)", f"{self.angle_kappa:.3f}"],
            [
                "Cornea center (x,y,z) mm",
                f"({cornea_center.x:.3f}, {cornea_center.y:.3f}, {cornea_center.z:.3f})",
            ],
            [
                "Anterior apex (x,y,z) mm",
                f"({apex_pos.x:.3f}, {apex_pos.y:.3f}, {apex_pos.z:.3f})",
            ],
            [
                "Pupil center (x,y,z) mm",
                f"({pupil_pos.x:.3f}, {pupil_pos.y:.3f}, {pupil_pos.z:.3f})",
            ],
            ["Pupil radius r_p (mm)", f"{x_radius:.3f}"],
        ]

        headers = ["Parameter", "Value"]
        info("Eye Anatomy Parameters:")
        table(data, headers=headers, tablefmt="grid")

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
            "axial_length": float(self.model.axial_length),
            "n_aqueous_humor": float(self.model.n_aqueous_humor),
            "fovea_displacement": bool(self.model.fovea_displacement),
            "fovea_alpha_deg": float(self.model.fovea_alpha_deg),
            "fovea_beta_deg": float(self.model.fovea_beta_deg),
            "which_eye": self.which_eye,
            # Pupil configuration
            "pupil_type": self.model.pupil_type,
            "pupil_boundary_points": self.model.pupil_boundary_points,
            "pupil_random_seed": self.model.pupil_random_seed,
            # Eyelid configuration
            "eyelid_enabled": bool(self.model.eyelid_enabled),
            "eyelid_transformation_matrix": self.eyelid_trans.tolist(),
        }

        # Serialize cornea
        if self.cornea:
            data["cornea"] = self.cornea.serialize()

        # Serialize pupil state
        if self.pupil:
            data["pupil"] = self.pupil.serialize()

        # Serialize eyelid if enabled
        if self.eyelid and self.model.eyelid_enabled:
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
        # Build the model from the stored specification, then construct the eye.
        model = EyeModel(
            axial_length=data["axial_length"],
            n_aqueous_humor=data["n_aqueous_humor"],
            fovea_displacement=data["fovea_displacement"],
            fovea_alpha_deg=data["fovea_alpha_deg"],
            fovea_beta_deg=data["fovea_beta_deg"],
            pupil_type=data["pupil_type"],
            pupil_boundary_points=data["pupil_boundary_points"],
            pupil_random_seed=data["pupil_random_seed"],
            eyelid_enabled=data["eyelid_enabled"],
        )
        eye = cls(model=model, which_eye=data.get("which_eye", "right"))

        # Restore position and orientation
        if data["position"]:
            eye.position = Position3D.deserialize(data["position"])

        eye.trans = TransformationMatrix(np.array(data["transformation_matrix"]))
        eye._rest_orientation = RotationMatrix.deserialize(data["rest_orientation"])
        eye.eyelid_trans = TransformationMatrix(np.array(data["eyelid_transformation_matrix"]))

        if data["current_target_point"]:
            eye._current_target_point = Position3D.deserialize(data["current_target_point"])

        # Restore the working cornea, live pupil and eyelid to their exact serialized state.
        if data.get("cornea"):
            eye.cornea = SphericalCornea.deserialize(data["cornea"])

        if data.get("pupil"):
            eye.pupil = EllipticalPupil.deserialize(data["pupil"])

        if "eyelid" in data and data["eyelid"] and eye.model.eyelid_enabled:
            eye.eyelid = Eyelid.deserialize(data["eyelid"])

        return eye
