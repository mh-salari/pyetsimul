"""Pinhole camera model for eye tracking simulation.

Implements camera projection, pan-tilt, and image capture for synthetic eye tracking experiments.
Supports both simple pinhole cameras and realistic cameras with distortion from OpenCV calibration.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from pyetsimul.log import info, table

from ..camera_noise import GlintNoiseConfig, apply_glint_noise
from ..optics.glint_size import compute_glint_diameter
from ..types import (
    CameraImage,
    CameraMatrix,
    Point2D,
    Position3D,
    ProjectionResult,
    RotationMatrix,
    TransformationMatrix,
)
from .default_configs import CameraDefaults
from .light import Light

if TYPE_CHECKING:
    from .eye import Eye


# Rows map a world-frame offset (x = right, y = depth away from the screen, z = up) into the reference
# frame the FickRotation projection uses (x = right, y = up, z = forward).
_REFERENCE_AXES = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])


def _fick_rotation_angles(camera_position: np.ndarray, target: np.ndarray) -> list[float]:
    """``[elevation, azimuth, torsion]`` degrees that aim a camera at ``camera_position`` toward ``target``.

    Both points are in the projection's reference frame. The rotation is the shortest arc turning the
    camera's initial facing (``-Z``) onto the target direction, decomposed into the three sequential angle
    rotations the FickRotation projection consumes. A camera already facing the target needs no turn.
    """
    initial = np.array([0.0, 0.0, -1.0])
    direction = np.asarray(target, float) - np.asarray(camera_position, float)
    direction /= np.linalg.norm(direction)
    cos_angle = float(np.clip(initial @ direction, -1.0, 1.0))
    axis = np.cross(initial, direction)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:  # already aligned or anti-aligned: no usable turn axis
        return [0.0, 0.0, 0.0]
    axis /= axis_norm
    skew = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    angle = np.arccos(cos_angle)
    rotation = np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)
    pitch = np.arcsin(np.clip(-rotation[2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    torsion = np.arctan2(rotation[2, 1], rotation[2, 2])
    return [float(np.degrees(-pitch)), float(np.degrees(-torsion)), float(np.degrees(yaw))]


@dataclass
class Camera:
    """Pinhole camera model for eye tracking.

    Key components:
    - trans: Camera to world transformation matrix
    - rest_trans: Rest position transformation (for pan-tilt cameras)
    - camera_matrix: CameraMatrix with focal_length and resolution properties
    - dist_coeffs: OpenCV distortion coefficients (default: no distortion)
    - err: Random error amount (default: 0.0)
    - err_type: Error distribution type ('gaussian' or 'uniform')
    - glint_noise_config: GlintNoiseConfig for corneal reflection detection noise (default: None)

    Usage:
    - Default pinhole: Camera()
    - Custom pinhole: c = Camera(); c.camera_matrix.focal_length = 1000
    - Realistic camera: Camera(camera_matrix=CameraMatrix(matrix), dist_coeffs=coeffs)
    """

    camera_matrix: CameraMatrix = field(default_factory=CameraMatrix)
    dist_coeffs: np.ndarray | None = None
    err: float = CameraDefaults.MEASUREMENT_ERROR
    err_type: str = "gaussian"
    glint_noise_config: GlintNoiseConfig | None = None
    name: str | None = None
    trans: TransformationMatrix = field(default_factory=TransformationMatrix.identity)
    rest_trans: TransformationMatrix = field(init=False)

    # Internal field to track where camera is pointing (set by point_at method)
    _pointing_at: Position3D | None = field(default=None, init=False)

    # World->pixel matrix when the camera is switched to the FickRotation projection; None = default pinhole.
    _fick_projection: np.ndarray | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """Initialize camera with default values."""
        # Store rest position for pan-tilt operations
        self.rest_trans = self.trans.copy()

        if self.dist_coeffs is None:
            self.dist_coeffs = np.zeros(5)
        else:
            self.dist_coeffs = np.asarray(self.dist_coeffs, dtype=np.float64)

    @property
    def orientation(self) -> RotationMatrix:
        """Get/set the camera's orientation (3x3 rotation matrix)."""
        return self.trans.get_rotation()

    @orientation.setter
    def orientation(self, value: RotationMatrix) -> None:
        """Set the camera's orientation matrix.

        Args:
            value: 3x3 rotation matrix (must be right-handed with determinant = +1)

        Raises:
            ValueError: If the matrix is not right-handed (det ≠ +1)

        """
        # RotationMatrix type already validates during construction

        self.trans[:3, :3] = value

    @property
    def position(self) -> Position3D:
        """Get/set the camera's position (3D vector)."""
        return Position3D.from_array(self.trans[:3, 3])

    @position.setter
    def position(self, value: Position3D) -> None:
        self.trans[:3, 3] = np.array(value)[:3]  # Extract x,y,z from homogeneous coordinates

    @property
    def pointing_at(self) -> Position3D | None:
        """Get the position that the camera is currently pointing at.

        Returns None if point_at() has never been called.
        """
        return self._pointing_at

    def project(self, pos: Position3D | list[Position3D]) -> ProjectionResult:
        """Projects points in space onto the camera's image plane.

        Transforms 3D positions to camera coordinates and projects to image plane.
        Adds random error based on camera settings and validates image bounds.
        Uses OpenCV for realistic camera projection when pinhole_mode=False.

        Args:
            pos: 3D positions to project. Can be:
                - Single Position3D object
                - list of Position3D objects

        Returns:
            ProjectionResult containing:
            - image_points: 2xn matrix of image coordinates (NaN for invalid points)
            - distances: 1xn array of distances from camera along optical axis
            - valid_mask: 1xn boolean array indicating points within image bounds

        """
        if self._fick_projection is not None:
            return self._project_fick(pos)

        # Convert input to homogeneous coordinates matrix
        if isinstance(pos, Position3D):
            # Single position
            pos_homogeneous = np.array(pos).reshape(-1, 1)
        elif isinstance(pos, list) and all(isinstance(p, Position3D) for p in pos):
            # list of positions
            pos_homogeneous = np.column_stack([np.array(p) for p in pos])
        else:
            raise ValueError(f"Position must be Position3D or list of Position3D objects, got: {type(pos)}")

        # Transform to camera coordinates
        pos_camera = np.linalg.solve(self.trans, pos_homogeneous)

        # Calculate distances along optical axis
        dist = -pos_camera[2, :]

        points_3d = pos_camera[:3, :].T
        points_3d_opencv = points_3d.copy()
        points_3d_opencv[:, 2] = -points_3d_opencv[:, 2]  # Flip Z coordinate system
        points_3d_opencv = points_3d_opencv.reshape(-1, 1, 3).astype(np.float64)

        points_2d, _ = cv2.projectPoints(
            points_3d_opencv, np.zeros(3), np.zeros(3), self.camera_matrix.matrix, self.dist_coeffs
        )

        # Convert to center-origin coordinate system
        cx = self.camera_matrix.matrix[0, 2]
        cy = self.camera_matrix.matrix[1, 2]
        x = points_2d.reshape(-1, 2).T
        x[0, :] -= cx
        x[1, :] -= cy

        # Add error based on error type
        rng = np.random.default_rng()
        if self.err_type == "uniform":
            x += self.err * (2 * rng.random((2, pos_camera.shape[1])) - 1)
        elif self.err_type == "gaussian":
            x += self.err * rng.normal(0, 1, (pos_camera.shape[1], 2)).T
        else:
            raise ValueError(f"Unknown error type: {self.err_type}")

        # Check which points are within image bounds and in front of camera
        # Small tolerance for floating-point boundary comparisons
        resolution = self.camera_matrix.resolution
        boundary_tol = 1e-10
        condition = (
            (x[0, :] >= -resolution.x / 2 - boundary_tol)
            & (x[0, :] <= resolution.x / 2 + boundary_tol)
            & (x[1, :] >= -resolution.y / 2 - boundary_tol)
            & (x[1, :] <= resolution.y / 2 + boundary_tol)
            & (dist > 0)  # Points must be in front of camera
        )

        # Set out-of-bounds points to NaN
        invalid_mask = ~condition
        x[:, invalid_mask] = np.nan

        return ProjectionResult(image_points=x, distances=dist, valid_mask=condition)

    def _set_fick_projection(
        self,
        reference_origin: Position3D,
        rotation_deg: list[float],
        translation: list[float],
        intrinsic: np.ndarray,
    ) -> None:
        """Build and store the FickRotation projection from explicit angles and translation.

        The world-to-camera rotation is composed by applying, in order, a torsion rotation about the
        optical axis, an elevation rotation about the horizontal axis and an azimuth rotation about the
        vertical axis. The azimuth term carries matching (both-negative) off-diagonal signs, so it is not a
        pure rotation: it scales the image as well as turning it, magnifying points more the further they
        fall from the centre. Everything else is an ordinary pinhole; lens distortion is not applied.

        Args:
            reference_origin: World point that is the origin of the projection's reference frame.
            rotation_deg: ``[elevation, azimuth, torsion]`` angles in degrees.
            translation: Camera position in the reference frame as ``[right, up, forward]``.
            intrinsic: 3x3 matrix of focal lengths and principal point in pixels.

        """
        origin = np.asarray(reference_origin, float)[:3]
        r1, r2, r3 = np.radians(rotation_deg)
        c1, s1 = np.cos(r1), np.sin(r1)
        c2, s2 = np.cos(r2), np.sin(r2)
        c3, s3 = np.cos(r3), np.sin(r3)
        torsion = np.array([[c3, -s3, 0.0], [s3, c3, 0.0], [0.0, 0.0, 1.0]])
        elevation = np.array([[c1, 0.0, -s1], [0.0, 1.0, 0.0], [s1, 0.0, c1]])
        azimuth = np.array([[1.0, 0.0, 0.0], [0.0, c2, -s2], [0.0, -s2, c2]])
        rotation = (torsion @ elevation @ azimuth) @ np.array([[1.0, 0, 0], [0, -1.0, 0], [0, 0, -1.0]])

        translation = np.asarray(translation, float)
        rotation_t_axes = rotation.T @ _REFERENCE_AXES
        # Fold the world->reference-frame change into the extrinsic so callers project world points directly.
        offset = -(rotation_t_axes @ origin) - rotation.T @ translation
        self._fick_projection = np.asarray(intrinsic, float) @ np.hstack([rotation_t_axes, offset.reshape(3, 1)])
        # Keep the world position consistent with the projection for the optics that read it.
        self.position = Position3D(*(origin + _REFERENCE_AXES.T @ translation))

    def _project_fick(self, pos: Position3D | list[Position3D]) -> ProjectionResult:
        """Project world points to pixels using the configured FickRotation projection."""
        if isinstance(pos, Position3D):
            world_homogeneous = np.array(pos).reshape(-1, 1)
        else:
            world_homogeneous = np.column_stack([np.array(p) for p in pos])
        projected = self._fick_projection @ world_homogeneous
        depth = projected[2, :]
        return ProjectionResult(image_points=projected[:2, :] / depth, distances=depth, valid_mask=depth > 0)

    def _aim_fick(self, eye: "Eye") -> None:
        """Configure the FickRotation projection aimed at ``eye``'s rotation centre.

        The rotation and the reference-frame translation are derived from this camera's own world position
        and the eye -- nothing precomputed is supplied -- and the camera's intrinsic is used as-is.
        """
        origin = np.array([eye.position.x, eye.position.y, eye.position.z], dtype=float)
        camera = np.array([self.position.x, self.position.y, self.position.z], dtype=float)
        centre = eye.rotation_centre
        target = np.array([centre.x, centre.y, centre.z], dtype=float)
        translation = _REFERENCE_AXES @ (camera - origin)
        rotation = _fick_rotation_angles(translation, _REFERENCE_AXES @ (target - origin))
        self._set_fick_projection(eye.position, rotation, list(translation), self.camera_matrix.matrix)

    def unproject(
        self, image_points: Point2D | list[Point2D], distance: float | np.ndarray
    ) -> Position3D | list[Position3D]:
        """Unprojects points on the image plane back into 3D space.

        Reconstructs 3D positions from 2D image points at specified distance.
        Uses inverse projection to map image coordinates to world coordinates.
        Uses OpenCV for realistic camera unprojection when pinhole_mode=False.

        Args:
            image_points: 2D image points. Can be:
                - Single Point2D object
                - list of Point2D objects
            distance: Distance from camera along optical axis

        Returns:
            Position3D object(s) in world coordinates

        """
        # Convert input to numpy array format
        if isinstance(image_points, Point2D):
            # Single point
            x = np.array([[image_points.x], [image_points.y]])
            single_point = True
        elif isinstance(image_points, list) and all(isinstance(p, Point2D) for p in image_points):
            # list of points
            x = np.array([[p.x for p in image_points], [p.y for p in image_points]])
            single_point = False
        else:
            raise ValueError(f"Image points must be Point2D or list of Point2D objects, got: {type(image_points)}")

        n = x.shape[1]

        # Convert distance to numpy array if needed
        d = np.full(n, distance) if isinstance(distance, (int, float)) else np.asarray(distance)

        # Convert from center-origin to top-left-origin coordinate system
        cx = self.camera_matrix.matrix[0, 2]
        cy = self.camera_matrix.matrix[1, 2]
        x_opencv = x.copy().astype(np.float64)
        x_opencv[0, :] += cx
        x_opencv[1, :] += cy

        points_2d_normalized = cv2.undistortPoints(
            x_opencv.T.reshape(-1, 1, 2).astype(np.float64), self.camera_matrix.matrix, self.dist_coeffs
        )

        points_3d = points_2d_normalized.reshape(-1, 2) * d.reshape(-1, 1)

        camera_coords = np.column_stack([
            points_3d[:, 0],
            points_3d[:, 1],
            -d,  # Camera faces -Z axis
            np.ones(len(points_3d)),
        ]).T

        # Transform to world coordinates
        world_coords = self.trans @ camera_coords

        # Convert result back to Position3D objects
        if single_point:
            return Position3D.from_array(world_coords[:, 0])
        return [Position3D.from_array(world_coords[:, i]) for i in range(n)]

    def pan_tilt(self, look_at: Position3D, world_frame: RotationMatrix | None = None) -> None:
        """Pans and tilts a camera towards a certain location.

        Orients camera to look directly at specified point in world coordinates.
        Modifies transformation matrix around camera's coordinate system origin.

        Args:
            look_at: Point to look at in world coordinates
            world_frame: Optional world coordinate frame for camera orientation

        """
        # Convert to homogeneous coordinates for transformation
        look_at_homogeneous = np.array(look_at)

        # Transform look_at point to camera's rest coordinate system
        axis_homogeneous = np.linalg.solve(self.rest_trans, look_at_homogeneous)

        # Extract and normalize the 3D direction vector
        axis = axis_homogeneous[:3] / np.linalg.norm(axis_homogeneous[:3])

        # Calculate pan and tilt angles
        # Handle special case where both axis[2] and axis[0] are zero
        alpha = 0.0 if abs(axis[2]) < 1e-10 and abs(axis[0]) < 1e-10 else np.pi / 2 - np.arctan2(-axis[2], axis[0])
        beta = np.arcsin(axis[1])

        # Construct pan matrix (rotation around Y axis)
        pan_matrix = np.array([
            [np.cos(alpha), 0, -np.sin(alpha), 0],
            [0, 1, 0, 0],
            [np.sin(alpha), 0, np.cos(alpha), 0],
            [0, 0, 0, 1],
        ])

        # Construct tilt matrix (rotation around X axis)
        tilt_matrix = np.array([
            [1, 0, 0, 0],
            [0, np.cos(beta), -np.sin(beta), 0],
            [0, np.sin(beta), np.cos(beta), 0],
            [0, 0, 0, 1],
        ])

        # Apply pan and tilt transformations
        self.trans = TransformationMatrix(self.rest_trans @ pan_matrix @ tilt_matrix)

        # If world_frame is specified, align camera with world coordinate frame
        if world_frame is not None:
            self._align_with_world_frame(world_frame)

    def _align_with_world_frame(self, world_frame: RotationMatrix) -> None:
        """Align camera orientation with world coordinate frame while preserving viewing direction."""
        # Get current camera viewing direction (where camera is pointing)
        current_rotation = self.trans[:3, :3]
        viewing_direction = -current_rotation[:, 2]  # Camera looks along -Z
        viewing_direction /= np.linalg.norm(viewing_direction)

        # Use the world's up vector as a reference to define the camera's orientation
        world_up = -world_frame[:, 2]

        # Create the camera's x-axis (perpendicular to viewing direction and world up)
        camera_x = np.cross(viewing_direction, world_up)

        # If viewing direction is parallel to world up, use world x-axis as fallback
        if np.linalg.norm(camera_x) < 1e-6:
            world_x = world_frame[:, 0]
            camera_x = np.cross(viewing_direction, world_x)

        camera_x /= np.linalg.norm(camera_x)

        # Compute camera y-axis orthogonal to x-axis and viewing direction
        camera_y = np.cross(camera_x, viewing_direction)

        # Construct orthogonal rotation matrix: [x, y, -viewing] as columns
        camera_rotation = np.column_stack((camera_x, camera_y, -viewing_direction))

        # Update camera's orientation
        self.trans[:3, :3] = camera_rotation

    def point_at(
        self, target: "Position3D | Eye", world_frame: RotationMatrix | None = None, mode: str | None = None
    ) -> None:
        """Point the camera at a target.

        By default the camera is oriented (pan-tilt) toward a 3D ``target`` point, which the ordinary
        projection then uses. With ``mode='fick'`` the camera is instead given the FickRotation projection
        aimed at an ``Eye``'s rotation centre, derived from the camera's own position and intrinsic.

        Args:
            target: A ``Position3D`` to look at (default), or an ``Eye`` to aim at when ``mode='fick'``.
            world_frame: Optional world coordinate frame for the default orientation.
            mode: ``None`` for the ordinary pan-tilt orientation, ``'fick'`` for the FickRotation projection.

        Raises:
            TypeError: If ``target`` does not match ``mode``.
            ValueError: If ``mode`` is neither ``None`` nor ``'fick'``.

        """
        if mode == "fick":
            from .eye import Eye  # noqa: PLC0415  (deferred to avoid a circular import with eye.py)

            if not isinstance(target, Eye):
                raise TypeError(f"point_at(mode='fick') expects an Eye, got {type(target).__name__}")
            self._aim_fick(target)
            return
        if mode is not None:
            raise ValueError(f"unknown point_at mode {mode!r}; expected None or 'fick'")
        if not isinstance(target, Position3D):
            raise TypeError(f"point_at expects a Position3D target, got {type(target).__name__}")

        self._pointing_at = target
        self.rest_trans = self.trans.copy()
        self.pan_tilt(target, world_frame)
        self.rest_trans = self.trans.copy()

    def point_at_binocular(self, left_eye_pos: Position3D, right_eye_pos: Position3D) -> None:
        """Point camera at the midpoint between two eyes, rolled so the inter-eye axis is horizontal.

        Computes the correct camera roll from the inter-eye vector so that both eyes
        appear horizontally aligned in the camera image. Delegates to point_at().

        Args:
            left_eye_pos: Position of the left eye in world coordinates.
            right_eye_pos: Position of the right eye in world coordinates.

        Raises:
            ValueError: If the camera lies on the inter-eye line (roll is undefined).

        """
        midpoint = left_eye_pos + (right_eye_pos - left_eye_pos) * 0.5

        # Inter-eye vector defines the camera's horizontal axis
        inter_eye = np.array(right_eye_pos)[:3] - np.array(left_eye_pos)[:3]
        inter_eye /= np.linalg.norm(inter_eye)

        # Viewing direction: camera to midpoint
        view_dir = np.array(midpoint)[:3] - np.array(self.position)[:3]
        view_dir /= np.linalg.norm(view_dir)

        # Camera "up" is perpendicular to both inter-eye axis and viewing direction
        up = np.cross(inter_eye, view_dir)
        up_norm = np.linalg.norm(up)
        if up_norm < 1e-10:
            msg = "Camera lies on the inter-eye line — roll is undefined for point_at_binocular."
            raise ValueError(msg)
        up /= up_norm

        # Build world frame: inter-eye as x, derived forward as y, -up as z
        forward = np.cross(-up, inter_eye)
        forward /= np.linalg.norm(forward)
        world_frame = RotationMatrix(np.column_stack([inter_eye, forward, -up]))

        self.point_at(midpoint, world_frame)

    def take_image(
        self,
        eye: "Eye",
        lights: list[Light] | None = None,
        use_refraction: bool = True,
        center_method: str = "ellipse",
    ) -> CameraImage:
        """Computes the image of an eye seen by a camera.

        Generates synthetic eye image with corneal reflections and pupil detection.
        Uses light sources to create corneal reflections (CRs) on the cornea.

        Args:
            eye: Eye object
            lights: list of light source objects (optional, if None no CRs are computed)
            use_refraction: Whether to use refraction model for pupil (default True)
            center_method: Method to use for pupil center detection (default "ellipse")
                          Options: "ellipse", "center_of_mass"

        Returns:
            CameraImage object containing corneal reflections, pupil boundary, and pupil center

        """
        # Find the corneal reflections for each light (if lights provided)
        corneal_reflections: list[Point2D | None] = []
        glint_sizes_px: list[float | None] = []
        has_any_glint_size = False
        if lights is not None:
            for light in lights:
                # Find 3D corneal reflection position
                cr_3d = eye.find_cr(light, self)

                if cr_3d is None:
                    corneal_reflections.append(None)
                    glint_sizes_px.append(None)
                else:
                    # Project to camera image coordinates using refactored interface
                    projection_result = self.project(cr_3d)
                    if np.any(np.isnan(projection_result.image_points)):
                        corneal_reflections.append(None)
                        glint_sizes_px.append(None)
                    else:
                        # Convert to Point2D
                        cr_2d = Point2D(
                            x=float(projection_result.image_points[0, 0]),
                            y=float(projection_result.image_points[1, 0]),
                        )
                        # Add noise to glint position
                        if self.glint_noise_config is not None:
                            cr_2d_noisy = apply_glint_noise(cr_2d, self.glint_noise_config)
                        else:
                            cr_2d_noisy = cr_2d
                        corneal_reflections.append(cr_2d_noisy)

                        # Compute glint size if the light has a physical diameter
                        if light.diameter is not None:
                            glint_diameter_3d = compute_glint_diameter(
                                light.position, cr_3d, eye.cornea, eye.trans, light.diameter
                            )
                            distance_to_cr = float(projection_result.distances[0])
                            glint_size = self.camera_matrix.focal_length * glint_diameter_3d / distance_to_cr
                            glint_sizes_px.append(glint_size)
                            has_any_glint_size = True
                        else:
                            glint_sizes_px.append(None)

        # Get pupil boundary and center
        pupil_boundary, pupil_center = eye.get_pupil_in_camera_image(
            self, use_refraction=use_refraction, center_method=center_method
        )

        return CameraImage(
            corneal_reflections=corneal_reflections,
            pupil_boundary=pupil_boundary,
            pupil_center=pupil_center,
            resolution=self.camera_matrix.resolution,
            glint_sizes_px=glint_sizes_px if has_any_glint_size else None,
        )

    def __str__(self) -> str:
        """Basic string representation of the camera."""
        res = self.camera_matrix.resolution
        pos = self.position
        return f"Camera(pos=({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})mm, f={self.camera_matrix.focal_length:.0f}px, {res.x}x{res.y})"

    def pprint(self) -> None:
        """Print detailed camera parameters in a formatted table."""
        pos = self.position
        matrix = self.camera_matrix.matrix
        res = self.camera_matrix.resolution

        # Prepare distortion coefficients display
        if self.dist_coeffs is not None:
            dist_info = f"k1={self.dist_coeffs[0]:.3f}"
            if len(self.dist_coeffs) > 1:
                dist_info += f", k2={self.dist_coeffs[1]:.3f}"
            if len(self.dist_coeffs) > 4:
                dist_info += f", p1={self.dist_coeffs[3]:.3f}, p2={self.dist_coeffs[4]:.3f}"
        else:
            dist_info = "None (pinhole)"

        data = [
            ["Focal length (px)", f"{self.camera_matrix.focal_length:.1f}"],
            ["Resolution", f"{res.x} x {res.y}"],
            ["Principal point (px)", f"({matrix[0, 2]:.1f}, {matrix[1, 2]:.1f})"],
            ["Position (x,y,z) mm", f"({pos.x:.1f}, {pos.y:.1f}, {pos.z:.1f})"],
            ["Distortion coefficients", dist_info],
            ["Measurement error", f"{self.err:.4f}"],
            ["Error type", self.err_type],
            ["Glint noise", "Enabled" if self.glint_noise_config else "Disabled"],
        ]

        headers = ["Parameter", "Value"]
        info("Camera Parameters:")
        table(data, headers=headers, tablefmt="grid")

    def serialize(self) -> dict:
        """Serialize to dictionary representation."""
        return {
            "position": self.position.serialize(),
            "orientation": self.orientation.tolist(),
            "focal_length": float(self.camera_matrix.focal_length),
            "resolution": self.camera_matrix.resolution.serialize(),
            "camera_matrix": self.camera_matrix.matrix.tolist(),
            "distortion_coefficients": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "measurement_error": float(self.err),
            "error_type": self.err_type,
            "name": self.name,
            "pointing_at": self._pointing_at.serialize() if self._pointing_at else None,
            "rest_transformation": self.rest_trans.tolist(),
            "glint_noise_config": self.glint_noise_config.serialize() if self.glint_noise_config else None,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "Camera":
        """Deserialize from dictionary representation."""
        # Create camera with basic parameters
        camera = cls(err=data["measurement_error"], err_type=data["error_type"], name=data["name"])

        # Restore position and orientation
        camera.position = Position3D.deserialize(data["position"])
        camera.orientation = RotationMatrix(np.array(data["orientation"]))

        # Restore camera matrix
        camera.camera_matrix = CameraMatrix(np.array(data["camera_matrix"]))

        # Restore distortion coefficients
        camera.dist_coeffs = np.array(data["distortion_coefficients"])

        # Restore rest transformation
        camera.rest_trans = TransformationMatrix(np.array(data["rest_transformation"]))

        # Restore pointing direction
        if data["pointing_at"]:
            camera._pointing_at = Position3D.deserialize(data["pointing_at"])

        # Restore glint noise config
        if data["glint_noise_config"]:
            camera.glint_noise_config = GlintNoiseConfig.deserialize(data["glint_noise_config"])

        return camera
