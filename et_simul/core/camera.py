"""Pinhole camera model for eye tracking simulation.

Implements camera projection, pan-tilt, and image capture for synthetic eye tracking experiments.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Union, List, TYPE_CHECKING

from .light import Light
from ..types import TransformationMatrix, RotationMatrix, Point2D, Point3D, Position3D, CameraImage, ProjectionResult

if TYPE_CHECKING:
    from .eye import Eye


@dataclass
class Camera:
    """Pinhole camera model for eye tracking.

    Implements pinhole camera with optical axis along negative z-axis.
    Provides projection/unprojection, pan-tilt control, and image capture.
    Supports random error simulation for realistic measurements.

    Key components:
    - trans: Camera to world transformation matrix
    - rest_trans: Rest position transformation (for pan-tilt cameras)
    - focal_length: Focal length in pixels (default: 2880)
    - resolution: Image resolution as Point2D (default: [1280, 1024])
    - err: Random error amount (default: 0.0)
    - err_type: Error distribution type ('gaussian' or 'uniform')
    """

    focal_length: float = 2880
    resolution: Point2D = field(default_factory=lambda: Point2D(x=1280, y=1024))
    err: float = 0.0
    err_type: str = "gaussian"
    trans: TransformationMatrix = field(default_factory=lambda: np.eye(4))
    rest_trans: TransformationMatrix = field(init=False)

    def __post_init__(self) -> None:
        """Initialize camera with default values."""
        # Line 70: c.rest_trans=c.trans;
        self.rest_trans = self.trans.copy()

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
    def position(self) -> Point3D:
        """Get/set the camera's position (3D vector)."""
        return Point3D.from_array(self.trans[:3, 3])

    @position.setter
    def position(self, value: Point3D) -> None:
        self.trans[:3, 3] = np.array(value)

    def project(self, pos: Union[Position3D, List[Position3D], np.ndarray]) -> ProjectionResult:
        """Projects points in space onto the camera's image plane.

        Transforms 3D positions to camera coordinates and projects to image plane.
        Adds random error based on camera settings and validates image bounds.

        Args:
            pos: 3D positions to project. Can be:
                - Single Position3D object
                - List of Position3D objects
                - 4×n numpy array of homogeneous coordinates (legacy support)

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
        elif isinstance(pos, np.ndarray):
            # Legacy numpy array support
            pos_homogeneous = pos.copy()
            if pos_homogeneous.ndim == 1:
                pos_homogeneous = pos_homogeneous.reshape(-1, 1)
        else:
            raise ValueError(f"Unsupported position type: {type(pos)}")

        # Transform to camera coordinates
        pos_camera = np.linalg.solve(self.trans, pos_homogeneous)

        # Calculate distances along optical axis
        dist = -pos_camera[2, :]

        # Project to image plane
        x = self.focal_length * pos_camera[:2, :] / dist

        # Add error based on error type
        if self.err_type == "uniform":
            x = x + self.err * (2 * np.random.rand(2, pos_camera.shape[1]) - 1)
        elif self.err_type == "gaussian":
            x = x + self.err * np.random.normal(0, 1, (pos_camera.shape[1], 2)).T
        else:
            raise ValueError(f"Unknown error type: {self.err_type}")

        # Check which points are within image bounds and in front of camera
        condition = (
            (x[0, :] >= -self.resolution.x / 2)
            & (x[0, :] <= self.resolution.x / 2)
            & (x[1, :] >= -self.resolution.y / 2)
            & (x[1, :] <= self.resolution.y / 2)
            & (dist > 0)  # Points must be in front of camera
        )

        # Set out-of-bounds points to NaN
        invalid_mask = ~condition
        x[:, invalid_mask] = np.nan

        return ProjectionResult(image_points=x, distances=dist, valid_mask=condition)

    def unproject(
        self, image_points: Union[Point2D, List[Point2D], np.ndarray], distance: Union[float, np.ndarray]
    ) -> Union[Position3D, List[Position3D]]:
        """Unprojects points on the image plane back into 3D space.

        Reconstructs 3D positions from 2D image points at specified distance.
        Uses inverse projection to map image coordinates to world coordinates.

        Args:
            image_points: 2D image points. Can be:
                - Single Point2D object
                - List of Point2D objects
                - 2×n numpy array (legacy support)
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
        elif isinstance(image_points, np.ndarray):
            # Legacy numpy array support
            X = image_points.copy()
            if X.ndim == 1:
                X = X.reshape(-1, 1)
            single_point = X.shape[1] == 1
        else:
            raise ValueError(f"Unsupported image_points type: {type(image_points)}")

        n = X.shape[1]

        # Convert distance to numpy array if needed
        if isinstance(distance, (int, float)):
            d = np.full(n, distance)
        else:
            d = np.asarray(distance)

        # Create camera coordinates
        camera_coords = np.array(
            [
                X[0, :] / self.focal_length * d,
                X[1, :] / self.focal_length * d,
                -d,  # Negative because camera looks down -Z axis
                np.ones(n),
            ]
        )

        # Transform to world coordinates
        world_coords = self.trans @ camera_coords

        # Convert result back to Position3D objects
        if single_point:
            return Position3D.from_array(world_coords[:, 0])
        else:
            return [Position3D.from_array(world_coords[:, i]) for i in range(n)]

    def pan_tilt(self, look_at: Position3D) -> None:
        """Pans and tilts a camera towards a certain location.

        Orients camera to look directly at specified point in world coordinates.
        Modifies transformation matrix around camera's coordinate system origin.

        Args:
            look_at: Point to look at in world coordinates
        """
        # Convert to homogeneous coordinates for transformation
        look_at_homogeneous = np.array(look_at)

        # Transform look_at point to camera's rest coordinate system
        axis_homogeneous = np.linalg.solve(self.rest_trans, look_at_homogeneous)

        # Extract and normalize the 3D direction vector
        axis = axis_homogeneous[:3] / np.linalg.norm(axis_homogeneous[:3])

        # Calculate pan and tilt angles
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

    def point_at(self, target_point: Position3D) -> None:
        """Points camera towards a certain location.

        Changes camera's rest position to point at specified target.
        Updates both rest_trans and trans matrices accordingly.
        Differs from pan_tilt() by modifying the rest position.

        Args:
            target_point: Point to point at in world coordinates
        """
        # Store current transformation as rest position
        self.rest_trans = self.trans.copy()

        # Pan and tilt towards the target point
        self.pan_tilt(target_point)

        # Update rest position to the new orientation
        self.rest_trans = self.trans.copy()

    def take_image(self, eye: "Eye", lights: List[Light], use_refraction: bool = True) -> CameraImage:
        """Computes the image of an eye seen by a camera.

        Generates synthetic eye image with corneal reflections and pupil detection.
        Uses light sources to create corneal reflections (CRs) on the cornea.

        Args:
            eye: Eye object
            lights: List of light source objects
            use_refraction: Whether to use refraction model for pupil (default True)

        Returns:
            CameraImage object containing corneal reflections, pupil boundary, and pupil center
        """
        # Find the corneal reflections for each light
        corneal_reflections = []
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
                        x=float(projection_result.image_points[0, 0]), y=float(projection_result.image_points[1, 0])
                    )
                    corneal_reflections.append(cr_2d)

        # Get pupil boundary and center
        pupil_boundary, pupil_center = eye.get_pupil_in_camera_image(self, use_refraction=use_refraction)

        return CameraImage(
            corneal_reflections=corneal_reflections,
            pupil_boundary=pupil_boundary,
            pupil_center=pupil_center,
            resolution=self.resolution,
        )
