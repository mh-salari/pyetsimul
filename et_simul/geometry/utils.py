import numpy as np
from typing import Tuple, Union
from ..types import Point3D, Point4D, Vector3D, CoordinateType


def lines_closest_point(
    p1: CoordinateType,
    d1: Union[Vector3D, Point4D],
    p2: CoordinateType,
    d2: Union[Vector3D, Point4D],
) -> Tuple[CoordinateType, CoordinateType]:
    """Computes points where two lines are closest.



    [x1, x2] = lines_closest_point(p1, d1, p2, d2) computes the points of
    closest proximity between the two lines given by the point 'p1' and the
    direction 'd1' on one hand and the point 'p2' and the direction 'd2' on
    the other hand. Returns the point 'x1' on the first line that is closest
    to the second line and the point 'x2' on the second line that is closest
    to the first line.

    If the lines are parallel, the result is undefined.

    3D or homogeneous coordinates may be passed in; the same type of
    coordinates is passed out.

    Args:
        p1: Point on line 1 (3D or homogeneous)
        d1: Direction of line 1 (3D or homogeneous)
        p2: Point on line 2 (3D or homogeneous)
        d2: Direction of line 2 (3D or homogeneous)

    Returns:
        Tuple of (x1, x2) where x1 is closest point on line 1, x2 is closest point on line 2
    """
    A = np.array([[np.dot(d1, d1), -np.dot(d2, d1)], [np.dot(d1, d2), -np.dot(d2, d2)]])

    diff = p1 - p2
    b = np.array([-np.dot(diff, d1), -np.dot(diff, d2)])

    try:
        alpha = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        nan_vec = np.full_like(p1, np.nan, dtype=np.float64)
        return nan_vec, nan_vec

    x1 = p1 + alpha[0] * d1
    x2 = p2 + alpha[1] * d2

    return x1, x2


def line_intersect_2d(p11: Point3D, p12: Point3D, p21: Point3D, p22: Point3D) -> Point3D:
    """Computes intersection of two-dimensional lines.



    p = line_intersect_2d(p11, p12, p21, p22) computes the intersection (in
    2D) of the line through 'p11' and 'p12' with the line through 'p21' and
    'p22'. The result is returned in 'p'. If the lines do not intersect, the
    result is undefined.

    Args:
        p11: First point on line 1 (2D)
        p12: Second point on line 1 (2D)
        p21: First point on line 2 (2D)
        p22: Second point on line 2 (2D)

    Returns:
        Intersection point (2D)
    """
    b = p21 - p11

    A = np.column_stack([p12 - p11, p22 - p21])

    try:
        x = np.linalg.solve(A, b)
        p = p11 + x[0] * (p12 - p11)
    except np.linalg.LinAlgError:
        # Handle parallel lines (singular matrix) - return NaN like MATLAB
        p = np.array([np.nan, np.nan])

    return p
