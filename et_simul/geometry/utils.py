"""Geometric utility functions for eye tracking simulation.

Provides line-line closest point, 2D line intersection, and other basic geometric operations.
"""

import numpy as np
from typing import Tuple
from ..types import Point2D, Point3D, Vector3D


def lines_closest_point(
    p1: Point3D,
    d1: Vector3D,
    p2: Point3D,
    d2: Vector3D,
) -> Tuple[Point3D, Point3D]:
    """Find the closest points between two lines in 3D space.

    Uses the minimum distance approach to find points on each line that are closest
    to each other. For parallel lines, returns NaN points.

    Args:
        p1: Point on first line
        d1: Direction vector of first line
        p2: Point on second line
        d2: Direction vector of second line

    Returns:
        Tuple of (x1, x2) where x1 is closest point on first line, x2 is closest point on second line.
        Returns (NaN point, NaN point) if lines are parallel.
    """
    # Set up linear system A * alpha = b for closest point parameters
    A = np.array([[d1.dot(d1), -d2.dot(d1)], [d1.dot(d2), -d2.dot(d2)]])
    b = np.array([-(p1 - p2).dot(d1), -(p1 - p2).dot(d2)])

    try:
        alpha = np.linalg.solve(A, b)
        # Compute closest points using structured type arithmetic
        return p1 + d1 * alpha[0], p2 + d2 * alpha[1]
    except np.linalg.LinAlgError:
        # Lines are parallel - return NaN points
        return Point3D(np.nan, np.nan, np.nan), Point3D(np.nan, np.nan, np.nan)


def line_intersect_2d(p11: Point2D, p12: Point2D, p21: Point2D, p22: Point2D) -> Point2D:
    """Find the intersection point of two 2D lines.

    Uses parametric line equations to solve for the intersection point.
    Returns NaN point if lines are parallel.

    Args:
        p11: First point on first line
        p12: Second point on first line
        p21: First point on second line
        p22: Second point on second line

    Returns:
        Intersection point. Returns Point2D(nan, nan) if lines are parallel.
    """
    # Set up linear system for intersection parameters
    dir1 = p12 - p11  # Direction vector of first line
    dir2 = p22 - p21  # Direction vector of second line
    A = np.column_stack([np.array(dir1), -np.array(dir2)])
    b = np.array(p21 - p11)

    try:
        t = np.linalg.solve(A, b)
        return p11 + dir1 * t[0]  # Return intersection point
    except np.linalg.LinAlgError:
        return Point2D(np.nan, np.nan)  # Parallel lines
