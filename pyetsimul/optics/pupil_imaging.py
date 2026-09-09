"""Pupil imaging functions extracted from the Eye class.

This module contains pupil imaging operations that were previously
part of the Eye class, extracted for better modularity and testability.
"""

import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
from skimage.draw import polygon

from ..core.camera import Camera
from ..types.geometry import Point2D, Position3D
from ..types.imaging import PupilData

if TYPE_CHECKING:
    from ..core.eye import Eye

MIN_ELLIPSE_POINTS = 5  # fewest points that determine an ellipse


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


def calculate_pupil_diameter_from_boundary(boundary_points: list[Point2D]) -> float | None:
    """Mean-axis diameter (px) of the ellipse fitted to the boundary: (major + minor) / 2.

    Matches the ``(axis1 + axis2) / 2`` diameter a contour detector reports, so a simulated boundary and a
    detected one are sized the same way. Returns None for fewer than five points; falls back to the boundary
    bounding-box extent only when the ellipse fit is degenerate.
    """
    if boundary_points is None or len(boundary_points) < MIN_ELLIPSE_POINTS:
        return None
    points = np.array([[p.x, p.y] for p in boundary_points])
    fitted = fit_ellipse(points)
    if fitted is not None:
        return (fitted[2] + fitted[3]) / 2.0
    warnings.warn(
        f"Ellipse fit failed for {len(points)} boundary points; reporting the mean bounding-box "
        f"extent instead, which is not the same quantity.",
        RuntimeWarning,
        stacklevel=2,
    )
    extent = (points[:, 0].max() - points[:, 0].min()) + (points[:, 1].max() - points[:, 1].min())
    return float(extent / 2.0)


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


def fit_ellipse(points: np.ndarray) -> tuple[float, float, float, float, float] | None:
    """(centre x, centre y, major axis, minor axis, angle in degrees) of the least-squares ellipse.

    Direct conic fit under the constraint that keeps the solution an ellipse, so any point set that
    determines one is fitted without iteration. The coordinates are centred and scaled first, because
    the conic design matrix on raw pixel values is ill-conditioned.

    Returns None only when the points determine no ellipse: fewer than five of them, or a singular
    system, which near-collinear points produce.
    """
    points = np.asarray(points, dtype=float)
    if points.shape[0] < MIN_ELLIPSE_POINTS:
        return None
    origin = points.mean(axis=0)
    centred = points - origin
    scale = float(np.sqrt((centred**2).sum(axis=1)).max())
    if not np.isfinite(scale) or scale <= 0:
        return None
    x, y = (centred / scale).T

    quadratic = np.column_stack([x * x, x * y, y * y])
    linear = np.column_stack([x, y, np.ones_like(x)])
    try:
        linear_terms = -np.linalg.solve(linear.T @ linear, (quadratic.T @ linear).T)
    except np.linalg.LinAlgError:
        return None
    reduced = quadratic.T @ quadratic + (quadratic.T @ linear) @ linear_terms
    # The ellipse constraint 4ac - b^2 = 1, inverted and applied to the reduced scatter matrix.
    constrained = np.array([reduced[2] / 2.0, -reduced[1], reduced[0] / 2.0])

    _eigenvalues, eigenvectors = np.linalg.eig(constrained)
    elliptical = np.where(4.0 * eigenvectors[0] * eigenvectors[2] - eigenvectors[1] ** 2 > 0)[0]
    if elliptical.size == 0:
        return None
    quadratic_terms = np.real(eigenvectors[:, elliptical[0]])
    a, b, c = quadratic_terms
    d, e, f = linear_terms @ quadratic_terms

    discriminant = b * b - 4.0 * a * c
    if not np.isfinite(discriminant) or discriminant == 0:
        return None
    centre_x = (2.0 * c * d - b * e) / discriminant
    centre_y = (2.0 * a * e - b * d) / discriminant

    # In the eigenbasis of the quadratic form the conic is curvature[i] * u[i]^2 = -constant, so each
    # eigenvector carries a semi-axis of sqrt(-constant / curvature[i]) and the major axis is simply the
    # longer of the two. Reading both from one decomposition keeps the axes and the angle consistent
    # whichever overall sign the conic solution came back with.
    constant = a * centre_x**2 + b * centre_x * centre_y + c * centre_y**2 + d * centre_x + e * centre_y + f
    curvature, directions = np.linalg.eigh(np.array([[a, b / 2.0], [b / 2.0, c]]))
    squared = -constant / curvature
    if np.any(squared <= 0) or not np.all(np.isfinite(squared)):
        return None
    semi_axes = np.sqrt(squared)
    major = int(np.argmax(semi_axes))
    if not (np.isfinite(centre_x) and np.isfinite(centre_y)):
        return None
    return (
        float(origin[0] + scale * centre_x),
        float(origin[1] + scale * centre_y),
        float(2.0 * scale * semi_axes[major]),
        float(2.0 * scale * semi_axes[1 - major]),
        float(np.degrees(np.arctan2(directions[1, major], directions[0, major])) % 180.0),
    )


def _safe_ellipse_center(points: np.ndarray) -> tuple[float, float] | None:
    """Centre (x, y) of the least-squares ellipse, or None when the points determine none."""
    fitted = fit_ellipse(points)
    return None if fitted is None else (fitted[0], fitted[1])


def _fit_ellipse_center(pupil_boundary: np.ndarray) -> Point2D | None:
    """Fit ellipse to pupil boundary points and return center.

    Uses least-squares ellipse fitting for robust center estimation, falling back to the boundary
    centroid when the fit is degenerate.

    Args:
        pupil_boundary: 2xN numpy array representing pupil boundary points

    Returns:
        Point2D with center coordinates, or None if there are too few points

    """
    if pupil_boundary.shape[1] < MIN_ELLIPSE_POINTS:
        return None
    center = _safe_ellipse_center(pupil_boundary.T)
    if center is not None:
        return Point2D(x=center[0], y=center[1])
    warnings.warn(
        f"Ellipse fit failed for {pupil_boundary.shape[1]} boundary points; reporting the boundary "
        f"centroid instead, which depends on how the boundary is sampled.",
        RuntimeWarning,
        stacklevel=2,
    )
    return Point2D(x=float(np.mean(pupil_boundary[0, :])), y=float(np.mean(pupil_boundary[1, :])))


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
    if pupil_boundary.shape[1] < MIN_ELLIPSE_POINTS:
        return None
    points = pupil_boundary.T
    hull_points = points[ConvexHull(points).vertices]
    if len(hull_points) < MIN_ELLIPSE_POINTS:
        return None
    center = _safe_ellipse_center(hull_points)
    if center is not None:
        return Point2D(x=center[0], y=center[1])
    warnings.warn(
        f"Ellipse fit failed for a convex hull of {len(hull_points)} vertices; reporting the mean of "
        f"those vertices instead, which depends on how densely the hull is sampled.",
        RuntimeWarning,
        stacklevel=2,
    )
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
