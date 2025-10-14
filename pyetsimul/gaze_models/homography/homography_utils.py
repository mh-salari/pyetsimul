"""Utility functions for homography computation and transformation."""

import cv2
import numpy as np


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray, ransac_threshold: float = 5.0) -> np.ndarray:
    """Compute homography using OpenCV with RANSAC for robustness.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        ransac_threshold: Maximum reprojection error in pixels for RANSAC inliers.
                         Points with larger error are considered outliers.
                         Default: 5.0 pixels (adjust based on camera resolution
                         and noise characteristics)

    Returns:
        3x3 homography matrix H such that dst = H @ src

    Raises:
        ValueError: If homography computation fails (degenerate configuration)

    """
    # Use RANSAC for robustness to outliers
    H, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, ransac_threshold)  # noqa: N806

    if H is None:
        raise ValueError("Failed to compute homography - points may be degenerate")

    return H


def apply_homography(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Apply homography to points using OpenCV.

    Args:
        H: 3x3 homography matrix
        points: Nx2 array of points or single 2D point

    Returns:
        Transformed points (Nx2 or 2D point)

    """
    # Ensure points are in correct shape for cv2.perspectiveTransform
    points = np.atleast_2d(points).astype(np.float32)

    # cv2.perspectiveTransform requires shape (N, 1, 2)
    points_reshaped = points.reshape(-1, 1, 2)

    # Apply homography
    transformed = cv2.perspectiveTransform(points_reshaped, H)

    # Reshape back to (N, 2) or (2,) for single point
    result = transformed.reshape(-1, 2)

    return result[0] if len(result) == 1 else result


def order_points_by_angle(points: np.ndarray) -> np.ndarray:
    """Sorts a list of 2D points counter-clockwise based on their angle around the centroid.

    Args:
        points: An Nx2 numpy array of 2D points.

    Returns:
        An Nx2 numpy array of sorted points.

    """
    # Calculate the centroid
    centroid = np.mean(points, axis=0)

    # Calculate the angle of each point with respect to the centroid
    angles = np.arctan2(points[:, 1] - centroid[1], points[:, 0] - centroid[0])

    # Sort the points based on the angles
    sorted_indices = np.argsort(angles)

    return points[sorted_indices]
