"""Utility functions for homography computation and transformation."""

import cv2
import numpy as np


def compute_homography(src_points: np.ndarray, dst_points: np.ndarray, ransac_threshold: float = 5.0) -> np.ndarray:
    """Compute homography using OpenCV with RANSAC for robustness.

    Args:
        src_points: Nx2 array of source points
        dst_points: Nx2 array of destination points
        ransac_threshold: Maximum reprojection error for RANSAC inliers, in the units of dst_points.
                         Points with larger error are considered outliers.

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
