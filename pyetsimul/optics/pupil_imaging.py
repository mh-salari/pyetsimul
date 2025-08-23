"""
Pupil imaging functions extracted from the Eye class.

This module contains pupil imaging operations that were previously
part of the Eye class, extracted for better modularity and testability.
"""

import numpy as np
from typing import Optional, List

from skimage.measure import EllipseModel
from skimage.draw import polygon
from scipy import ndimage

from ..types.geometry import Point2D
from ..types.imaging import PupilData
from ..core.camera import Camera


def get_pupil_boundary_image(eye, camera: Camera, use_refraction: bool = True) -> PupilData:
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


def get_pupil_ellipse_image(eye, camera: Camera, use_refraction: bool = True) -> PupilData:
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

    # Fit ellipse to find center
    pupil_center = _fit_ellipse_center(pupil_data.boundary_points)

    return PupilData(
        boundary_points=pupil_data.boundary_points,
        center=pupil_center,
        ellipse_params=None,  # Could be extended to return full ellipse parameters
    )


def get_pupil_center_mass_image(eye, camera: Camera, use_refraction: bool = True) -> PupilData:
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

    # Calculate center of mass
    pupil_center = _calculate_center_of_mass(pupil_data.boundary_points, camera.camera_matrix.resolution)

    return PupilData(boundary_points=pupil_data.boundary_points, center=pupil_center)


def calculate_pupil_center_from_boundary(
    boundary_points: List[Point2D], camera_resolution: Point2D, center_method: str = "ellipse"
) -> Optional[Point2D]:
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
    elif center_method == "center_of_mass":
        return _calculate_center_of_mass(boundary_array, camera_resolution)
    else:
        raise ValueError(f"Unknown center_method '{center_method}'. Use 'ellipse' or 'center_of_mass'")


def calculate_pupil_center_methods(
    eye, camera: Camera, use_refraction: bool = True, center_method: str = "ellipse"
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
    elif center_method == "center_of_mass":
        return get_pupil_center_mass_image(eye, camera, use_refraction)
    else:
        raise ValueError(f"Unknown center_method '{center_method}'. Use 'ellipse' or 'center_of_mass'")


def _fit_ellipse_center(pupil_boundary) -> Optional[Point2D]:
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
    ellipse = EllipseModel()

    if ellipse.estimate(points):
        # Extract center coordinates
        center_x, center_y = ellipse.params[:2]
        return Point2D(x=float(center_x), y=float(center_y))
    else:
        # Fallback to simple centroid if ellipse fitting fails
        center_x = np.mean(pupil_boundary[0, :])
        center_y = np.mean(pupil_boundary[1, :])
        return Point2D(x=float(center_x), y=float(center_y))

    return None


def _calculate_center_of_mass(pupil_boundary, camera_resolution: Point2D) -> Optional[Point2D]:
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
