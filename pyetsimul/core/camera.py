"""Pinhole camera model for eye tracking simulation.

Implements camera projection, pan-tilt, and image capture for synthetic eye tracking experiments.
Supports both simple pinhole cameras and realistic cameras with distortion from OpenCV calibration.
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Union, List, TYPE_CHECKING, Optional

from .light import Light
from ..types import (
    TransformationMatrix,
    RotationMatrix,
    Point2D,
    Position3D,
    CameraImage,
    CameraMatrix,
    ProjectionResult,
)
from ..camera_noise import apply_glint_noise, GlintNoiseConfig
from .default_configs import CameraDefaults

if TYPE_CHECKING:
    from .eye import Eye


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
    dist_coeffs: Optional[np.ndarray] = None
    err: float = CameraDefaults.MEASUREMENT_ERROR
    err_type: str = "gaussian"
    glint_noise_config: Optional[GlintNoiseConfig] = None
    name: Optional[str] = None
    trans: TransformationMatrix = field(default_factory=lambda: np.eye(4))
    rest_trans: TransformationMatrix = field(init=False)

    # Internal field to track where camera is pointing (set by point_at method)
    _pointing_at: Optional[Position3D] = field(default=None, init=False)

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
        return self.trans[:3, :3]

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
    def pointing_at(self) -> Optional[Position3D]:
        """Get the position that the camera is currently pointing at.

        Returns None if point_at() has never been called.
        """
        return self._pointing_at

    def project(self, pos: Union[Position3D, List[Position3D]]) -> ProjectionResult:
        """Projects points in space onto the camera's image plane.

        Transforms 3D positions to camera coordinates and projects to image plane.
        Adds random error based on camera settings and validates image bounds.
        Uses OpenCV for realistic camera projection when pinhole_mode=False.

        Args:
            pos: 3D positions to project. Can be:
                - Single Position3D object
                - List of Position3D objects

        Returns:
            ProjectionResult containing:
            - image_points: 2×n matrix of image coordinates (NaN for invalid points)
            - distances: 1×n array of distances from camera along optical axis
            - valid_mask: 1×n boolean array indicating points within image bounds
        """
        # Convert input to homogeneous coordinates matrix
        if isinstance(pos, Position3D):
            # Single position
            pos_homogeneous = np.array(pos).reshape(-1, 1)
        elif isinstance(pos, list) and all(isinstance(p, Position3D) for p in pos):
            # List of positions
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
        if self.err_type == "uniform":
            x = x + self.err * (2 * np.random.rand(2, pos_camera.shape[1]) - 1)
        elif self.err_type == "gaussian":
            x = x + self.err * np.random.normal(0, 1, (pos_camera.shape[1], 2)).T
        else:
            raise ValueError(f"Unknown error type: {self.err_type}")

        # Check which points are within image bounds and in front of camera
        resolution = self.camera_matrix.resolution
        condition = (
            (x[0, :] >= -resolution.x / 2)
            & (x[0, :] <= resolution.x / 2)
            & (x[1, :] >= -resolution.y / 2)
            & (x[1, :] <= resolution.y / 2)
            & (dist > 0)  # Points must be in front of camera
        )

        # Set out-of-bounds points to NaN
        invalid_mask = ~condition
        x[:, invalid_mask] = np.nan

        return ProjectionResult(image_points=x, distances=dist, valid_mask=condition)

    def unproject(
        self, image_points: Union[Point2D, List[Point2D]], distance: Union[float, np.ndarray]
    ) -> Union[Position3D, List[Position3D]]:
        """Unprojects points on the image plane back into 3D space.

        Reconstructs 3D positions from 2D image points at specified distance.
        Uses inverse projection to map image coordinates to world coordinates.
        Uses OpenCV for realistic camera unprojection when pinhole_mode=False.

        Args:
            image_points: 2D image points. Can be:
                - Single Point2D object
                - List of Point2D objects
            distance: Distance from camera along optical axis

        Returns:
            Position3D object(s) in world coordinates
        """
        # Convert input to numpy array format
        if isinstance(image_points, Point2D):
            # Single point
            X = np.array([[image_points.x], [image_points.y]])
            single_point = True
        elif isinstance(image_points, list) and all(isinstance(p, Point2D) for p in image_points):
            # List of points
            X = np.array([[p.x for p in image_points], [p.y for p in image_points]])
            single_point = False
        else:
            raise ValueError(f"Image points must be Point2D or list of Point2D objects, got: {type(image_points)}")

        n = X.shape[1]

        # Convert distance to numpy array if needed
        if isinstance(distance, (int, float)):
            d = np.full(n, distance)
        else:
            d = np.asarray(distance)

        # Convert from center-origin to top-left-origin coordinate system
        cx = self.camera_matrix.matrix[0, 2]
        cy = self.camera_matrix.matrix[1, 2]
        X_opencv = X.copy().astype(np.float64)
        X_opencv[0, :] += cx
        X_opencv[1, :] += cy

        points_2d_normalized = cv2.undistortPoints(
            X_opencv.T.reshape(-1, 1, 2).astype(np.float64), self.camera_matrix.matrix, self.dist_coeffs
        )

        points_3d = points_2d_normalized.reshape(-1, 2) * d.reshape(-1, 1)

        camera_coords = np.column_stack(
            [
                points_3d[:, 0],
                points_3d[:, 1],
                -d,  # Camera faces -Z axis
                np.ones(len(points_3d)),
            ]
        ).T

        # Transform to world coordinates
        world_coords = self.trans @ camera_coords

        # Convert result back to Position3D objects
        if single_point:
            return Position3D.from_array(world_coords[:, 0])
        else:
            return [Position3D.from_array(world_coords[:, i]) for i in range(n)]

    def pan_tilt(self, look_at: Position3D, world_frame: Optional[RotationMatrix] = None) -> None:
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
        if abs(axis[2]) < 1e-10 and abs(axis[0]) < 1e-10:
            # Target is directly in front/behind camera, no pan needed
            alpha = 0.0
        else:
            alpha = np.pi / 2 - np.arctan2(-axis[2], axis[0])
        beta = np.arcsin(axis[1])

        # Construct pan matrix (rotation around Y axis)
        pan_matrix = np.array(
            [
                [np.cos(alpha), 0, -np.sin(alpha), 0],
                [0, 1, 0, 0],
                [np.sin(alpha), 0, np.cos(alpha), 0],
                [0, 0, 0, 1],
            ]
        )

        # Construct tilt matrix (rotation around X axis)
        tilt_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(beta), -np.sin(beta), 0],
                [0, np.sin(beta), np.cos(beta), 0],
                [0, 0, 0, 1],
            ]
        )

        # Apply pan and tilt transformations
        self.trans = self.rest_trans @ pan_matrix @ tilt_matrix

        # If world_frame is specified, align camera with world coordinate frame
        if world_frame is not None:
            self._align_with_world_frame(world_frame)

    def _align_with_world_frame(self, world_frame: RotationMatrix) -> None:
        """Align camera orientation with world coordinate frame while preserving viewing direction."""
        # Get current camera viewing direction (where camera is pointing)
        current_rotation = self.trans[:3, :3]
        viewing_direction = -current_rotation[:, 2]  # Camera looks along -Z
        viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)

        # Use the world's up vector as a reference to define the camera's orientation
        world_up = -world_frame[:, 2]

        # Create the camera's x-axis (perpendicular to viewing direction and world up)
        camera_x = np.cross(viewing_direction, world_up)

        # If viewing direction is parallel to world up, use world x-axis as fallback
        if np.linalg.norm(camera_x) < 1e-6:
            world_x = world_frame[:, 0]
            camera_x = np.cross(viewing_direction, world_x)

        camera_x = camera_x / np.linalg.norm(camera_x)

        # Compute camera y-axis orthogonal to x-axis and viewing direction
        camera_y = np.cross(camera_x, viewing_direction)

        # Construct orthogonal rotation matrix: [x, y, -viewing] as columns
        camera_rotation = np.column_stack((camera_x, camera_y, -viewing_direction))

        # Update camera's orientation
        self.trans[:3, :3] = camera_rotation

    def point_at(self, target_point: Position3D, world_frame: Optional[RotationMatrix] = None) -> None:
        """Points camera towards a certain location.

        Changes camera's rest position to point at specified target.
        Updates both rest_trans and trans matrices accordingly.
        Differs from pan_tilt() by modifying the rest position.

        Args:
            target_point: Point to point at in world coordinates
            world_frame: Optional world coordinate frame for camera orientation
        """
        # Store the target point for later reference
        self._pointing_at = target_point

        # Store current transformation as rest position
        self.rest_trans = self.trans.copy()

        # Pan and tilt towards the target point
        self.pan_tilt(target_point, world_frame)

        # Update rest position to the new orientation
        self.rest_trans = self.trans.copy()

    def take_image(
        self,
        eye: "Eye",
        lights: Optional[List[Light]] = None,
        use_refraction: bool = True,
        center_method: str = "ellipse",
    ) -> CameraImage:
        """Computes the image of an eye seen by a camera.

        Generates synthetic eye image with corneal reflections and pupil detection.
        Uses light sources to create corneal reflections (CRs) on the cornea.

        Args:
            eye: Eye object
            lights: List of light source objects (optional, if None no CRs are computed)
            use_refraction: Whether to use refraction model for pupil (default True)
            center_method: Method to use for pupil center detection (default "ellipse")
                          Options: "ellipse", "center_of_mass"

        Returns:
            CameraImage object containing corneal reflections, pupil boundary, and pupil center
        """
        # Find the corneal reflections for each light (if lights provided)
        corneal_reflections = []
        if lights is not None:
            for light in lights:
                # Find 3D corneal reflection position
                cr_3d = eye.find_cr(light, self)

                if cr_3d is None:
                    corneal_reflections.append(None)
                else:
                    # Project to camera image coordinates using refactored interface
                    projection_result = self.project(cr_3d)
                    if np.any(np.isnan(projection_result.image_points)):
                        corneal_reflections.append(None)
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

        # Get pupil boundary and center
        pupil_boundary, pupil_center = eye.get_pupil_in_camera_image(
            self, use_refraction=use_refraction, center_method=center_method
        )

        return CameraImage(
            corneal_reflections=corneal_reflections,
            pupil_boundary=pupil_boundary,
            pupil_center=pupil_center,
            resolution=self.camera_matrix.resolution,
        )
