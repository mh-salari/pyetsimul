"""Pupil imaging functions extracted from the Eye class.

This module contains pupil imaging operations that were previously
part of the Eye class, extracted for better modularity and testability.
"""

from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from skimage.draw import polygon
from skimage.measure import EllipseModel

from ..core.camera import Camera
from ..types.geometry import Point2D, Position3D
from ..types.imaging import PupilData

if TYPE_CHECKING:
    from ..core.eye import Eye


def get_pupil_boundary_image(eye: "Eye", camera: Camera, use_refraction: bool = True) -> PupilData:
    """Computes image of pupil boundary.

    Projects pupil boundary to camera image with corneal refraction.
    Accounts for camera error and visibility constraints.

    Args:
        eye: Eye object
        camera: Camera object
        use_refraction: Whether to apply corneal refraction (default True)

    Returns:
        PupilData object with boundary points in camera image

    """
    # Use the new eye method that already handles refraction and projection
    boundary_points, _ = eye.get_pupil_in_camera_image(camera, use_refraction=use_refraction)

    return PupilData(boundary_points=boundary_points)


def get_pupil_ellipse_image(eye: "Eye", camera: Camera, use_refraction: bool = True) -> PupilData:
    """Determines pupil ellipse in camera image.

    Fits ellipse to pupil boundary points to find center.
    Uses least-squares ellipse fitting for robust center estimation.

    Args:
        eye: Eye object
        camera: Camera object
        use_refraction: Whether to use refraction model (default True)

    Returns:
        PupilData object containing boundary points and ellipse center

    """
    # Get pupil boundary points
    pupil_data = get_pupil_boundary_image(eye, camera, use_refraction=use_refraction)

    if not pupil_data.boundary_points:
        return PupilData.empty()

    # Fit ellipse to find center; route through the helper that converts the Point2D list to the
    # 2xN array the fitter expects.
    pupil_center = calculate_pupil_center_from_boundary(
        pupil_data.boundary_points, camera.camera_matrix.resolution, center_method="ellipse"
    )

    return PupilData(
        boundary_points=pupil_data.boundary_points,
        center=pupil_center,
        ellipse_params=None,  # Could be extended to return full ellipse parameters
    )


def get_pupil_center_mass_image(eye: "Eye", camera: Camera, use_refraction: bool = True) -> PupilData:
    """Determines pupil center using center of mass calculation.

    Creates binary mask from boundary points and calculates centroid.
    Provides alternative to ellipse fitting for center estimation.

    Args:
        eye: Eye object
        camera: Camera object
        use_refraction: Whether to use refraction model (default True)

    Returns:
        PupilData object containing boundary points and center of mass

    """
    # Get pupil boundary points
    pupil_data = get_pupil_boundary_image(eye, camera, use_refraction=use_refraction)

    if not pupil_data.boundary_points:
        return PupilData.empty()

    # Calculate center of mass; route through the helper that converts the Point2D list to the
    # 2xN array the helper expects.
    pupil_center = calculate_pupil_center_from_boundary(
        pupil_data.boundary_points, camera.camera_matrix.resolution, center_method="center_of_mass"
    )

    return PupilData(boundary_points=pupil_data.boundary_points, center=pupil_center)


def calculate_pupil_center_from_boundary(
    boundary_points: list[Point2D], camera_resolution: Point2D, center_method: str = "ellipse"
) -> Point2D | None:
    """Calculate pupil center from boundary points using specified method.

    Args:
        boundary_points: List of Point2D boundary points in image coordinates
        camera_resolution: Camera resolution as Point2D
        center_method: Method to use for center detection ("ellipse" or "center_of_mass")

    Returns:
        Point2D with pupil center coordinates, or None if calculation fails

    Raises:
        ValueError: If center_method is not recognized

    """
    if boundary_points is None or len(boundary_points) < 3:
        return None

    # Convert to numpy array format for existing helper functions
    boundary_array = np.array([[p.x for p in boundary_points], [p.y for p in boundary_points]])

    if center_method == "ellipse":
        return _fit_ellipse_center(boundary_array)
    if center_method == "convex_hull":
        return _fit_convex_hull_center(boundary_array)
    if center_method == "center_of_mass":
        return _calculate_center_of_mass(boundary_array, camera_resolution)
    raise ValueError(f"Unknown center_method '{center_method}'. Use 'ellipse', 'convex_hull', or 'center_of_mass'")


def calculate_pupil_center_methods(
    eye: "Eye", camera: Camera, use_refraction: bool = True, center_method: str = "ellipse"
) -> PupilData:
    """Gets pupil boundary and center in camera image using specified method.

    Provides unified interface for different pupil center detection methods.
    Supports ellipse fitting and center of mass calculations.

    Args:
        eye: Eye object
        camera: Camera object
        use_refraction: Whether to use refraction model (default True)
        center_method: Method to use for pupil center detection (default "ellipse")
                      Options: "ellipse", "center_of_mass"

    Returns:
        PupilData object containing boundary points and center using specified method

    Raises:
        ValueError: If center_method is not recognized

    """
    if center_method == "ellipse":
        return get_pupil_ellipse_image(eye, camera, use_refraction)
    if center_method == "center_of_mass":
        return get_pupil_center_mass_image(eye, camera, use_refraction)
    raise ValueError(f"Unknown center_method '{center_method}'. Use 'ellipse' or 'center_of_mass'")


def project_point_to_image(
    eye: "Eye", camera: Camera, point: Position3D, use_refraction: bool = True
) -> Point2D | None:
    """Projects a single intraocular point to camera image coordinates.

    Forward of what a camera observes for one point inside the eye: the point is imaged through corneal
    refraction (the apparent, entrance-side position) and then projected to the image plane. With
    use_refraction disabled the point is projected directly, ignoring the cornea.

    Args:
        eye: Eye object providing the corneal refraction.
        camera: Camera object to project into.
        point: Intraocular point in world coordinates.
        use_refraction: Whether to apply corneal refraction (default True).

    Returns:
        Point2D image coordinates, or None if the point does not refract to the camera or projects to NaN.

    """
    observed = eye.find_refracted_position(camera.position, point) if use_refraction else point
    if observed is None:
        return None
    image_points = camera.project([observed]).image_points
    if np.any(np.isnan(image_points[:, 0])):
        return None
    return Point2D(x=float(image_points[0, 0]), y=float(image_points[1, 0]))


def reverse_project_to_pupil_plane(eye: "Eye", camera: Camera, image_points: list[Point2D]) -> np.ndarray:
    """Recovers eye-frame points in the pupil plane that image to the given camera points.

    Inverse of project_point_to_image for points constrained to the pupil plane at the eye's current
    orientation. For each image point it searches the in-plane offset from the pupil center whose refracted
    projection matches the target (Nelder-Mead), then expresses the result in eye-frame coordinates. The
    search is confined to the pupil plane because a single camera cannot resolve the depth of an unconstrained
    point; recovering the axial component requires combining two cameras (see image_jacobian).

    Args:
        eye: Eye object providing the corneal refraction and current orientation.
        camera: Camera object the points were imaged in.
        image_points: Camera-image points to invert.

    Returns:
        Nx3 array of eye-frame coordinates, one row per input point.

    """
    transform = np.asarray(eye.trans)
    x_axis, y_axis = transform[:3, 0], transform[:3, 1]
    pupil_center = np.asarray(eye.get_pupil().boundary_points)[:3].mean(axis=1)

    def residual(offset: np.ndarray, target: np.ndarray) -> float:
        world = pupil_center + offset[0] * x_axis + offset[1] * y_axis
        projected = project_point_to_image(eye, camera, Position3D(float(world[0]), float(world[1]), float(world[2])))
        if projected is None:
            return 1e9
        return float((projected.x - target[0]) ** 2 + (projected.y - target[1]) ** 2)

    eye_frame_points = []
    for image_point in image_points:
        target = np.array([image_point.x, image_point.y])
        solution = minimize(
            residual, [0.0, 0.0], args=(target,), method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-10}
        )
        world = pupil_center + solution.x[0] * x_axis + solution.x[1] * y_axis
        eye_frame_points.append((np.linalg.inv(transform) @ np.array([world[0], world[1], world[2], 1.0]))[:3])
    return np.array(eye_frame_points)


def image_jacobian(
    eye: "Eye", camera: Camera, eye_frame_point: np.ndarray, epsilon: float = 0.05
) -> np.ndarray | None:
    """Central-difference Jacobian d(image px)/d(eye-frame mm) of the refracted projection at a point.

    Evaluated at the eye's current orientation by displacing the point along each eye-frame axis and
    re-projecting. The third (axial) column is the depth sensitivity a single camera cannot resolve but
    that two cameras can jointly invert when their Jacobians are stacked.

    Args:
        eye: Eye object providing the corneal refraction and current orientation.
        camera: Camera object to project into.
        eye_frame_point: Point in eye-frame coordinates at which to evaluate the Jacobian.
        epsilon: Central-difference step in millimetres (default 0.05).

    Returns:
        2x3 Jacobian array, or None if any perturbed point fails to project.

    """
    transform = np.asarray(eye.trans)
    columns = []
    for axis in range(3):
        step = np.zeros(3)
        step[axis] = epsilon
        plus_world = (transform @ np.append(eye_frame_point + step, 1.0))[:3]
        minus_world = (transform @ np.append(eye_frame_point - step, 1.0))[:3]
        plus = project_point_to_image(
            eye, camera, Position3D(float(plus_world[0]), float(plus_world[1]), float(plus_world[2]))
        )
        minus = project_point_to_image(
            eye, camera, Position3D(float(minus_world[0]), float(minus_world[1]), float(minus_world[2]))
        )
        if plus is None or minus is None:
            return None
        columns.append((np.array([plus.x, plus.y]) - np.array([minus.x, minus.y])) / (2 * epsilon))
    return np.array(columns).T


def _fit_ellipse_center(pupil_boundary: np.ndarray) -> Point2D | None:
    """Fit ellipse to pupil boundary points and return center.

    Uses least-squares ellipse fitting for robust center estimation.
    Falls back to centroid if scikit-image not available.

    Args:
        pupil_boundary: 2xN numpy array representing pupil boundary points

    Returns:
        Point2D with center coordinates, or None if fitting fails

    """
    if pupil_boundary.shape[1] < 5:
        return None

    # Convert 2xN array to Nx2 array for ellipse fitting
    points = pupil_boundary.T
    ellipse = EllipseModel.from_estimate(points)

    if ellipse:
        # Extract center coordinates
        center_x, center_y = ellipse.center
        return Point2D(x=float(center_x), y=float(center_y))
    # Fallback to simple centroid if ellipse fitting fails
    center_x = np.mean(pupil_boundary[0, :])
    center_y = np.mean(pupil_boundary[1, :])
    return Point2D(x=float(center_x), y=float(center_y))

    return None


def _fit_convex_hull_center(pupil_boundary: np.ndarray) -> Point2D | None:
    """Fit an ellipse to the convex hull of the boundary points and return its center.

    The hull fills concavities before the ellipse is fit, so a notch in the boundary (such as one a
    corneal reflection cuts into a detected pupil contour) does not pull the center toward it. For a
    smooth, already convex boundary the result matches the plain ellipse-center method.

    Args:
        pupil_boundary: 2xN numpy array representing pupil boundary points.

    Returns:
        Point2D with center coordinates, or None if fitting fails.

    """
    if pupil_boundary.shape[1] < 5:
        return None
    points = pupil_boundary.T
    hull_points = points[ConvexHull(points).vertices]
    if len(hull_points) < 5:
        return None
    ellipse = EllipseModel.from_estimate(hull_points)
    if ellipse:
        center_x, center_y = ellipse.center
        return Point2D(x=float(center_x), y=float(center_y))
    return Point2D(x=float(np.mean(hull_points[:, 0])), y=float(np.mean(hull_points[:, 1])))


def _calculate_center_of_mass(pupil_boundary: np.ndarray, camera_resolution: Point2D) -> Point2D | None:
    """Calculate center of mass from pupil boundary points using binary mask.

    Creates binary mask from boundary polygon and calculates centroid.
    Falls back to simple centroid if required packages not available.

    Args:
        pupil_boundary: 2xN numpy array representing pupil boundary points
        camera_resolution: Point2D with camera width (x) and height (y)

    Returns:
        Point2D with center of mass coordinates, or None if calculation fails

    """
    if pupil_boundary.shape[1] < 3:
        return None

    # Use polygon and ndimage for center of mass calculation

    # Convert camera coordinates to image array coordinates
    width, height = int(camera_resolution.x), int(camera_resolution.y)

    # Convert pupil points to array coordinates
    pupil_array_x = pupil_boundary[0, :] + width // 2
    pupil_array_y = pupil_boundary[1, :] + height // 2

    # Clip to valid image bounds
    pupil_array_x = np.clip(pupil_array_x, 0, width - 1)
    pupil_array_y = np.clip(pupil_array_y, 0, height - 1)

    # Create binary mask
    mask = np.zeros((height, width), dtype=bool)

    # Fill polygon defined by pupil boundary
    rr, cc = polygon(pupil_array_y, pupil_array_x, shape=(height, width))
    mask[rr, cc] = True

    if not np.any(mask):
        return None

    # Calculate center of mass
    y_center, x_center = ndimage.center_of_mass(mask.astype(float))

    # Convert back to camera coordinates
    x_camera = x_center - width // 2
    y_camera = y_center - height // 2

    return Point2D(x=float(x_camera), y=float(y_camera))
