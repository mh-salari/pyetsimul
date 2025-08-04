"""Surface point generation utilities"""

import numpy as np
from typing import Callable
from ..types import Position3D, Direction3D, Ray


def generate(intersection_func: Callable, center: Position3D, radius: float, *args, n_points: int = 40) -> np.ndarray:
    """Generate surface points using a given intersection function.

    Args:
        intersection_func: Ray intersection function (intersect_ray_conic, intersect_ray_sphere, etc.)
        center: Surface center position
        radius: Surface radius
        *args: Additional arguments for the intersection function (e.g., k for conic)
        n_points: Number of sampling points per dimension

    Returns:
        points: (N, 3) array of surface points
    """
    # Create sampling directions from surface center
    sample_range = np.linspace(-0.8, 0.8, n_points)
    points = []

    for x_norm in sample_range:
        for y_norm in sample_range:
            for z_sign in [-1, 1]:  # Both sides of the surface
                # Create direction from center
                direction = Direction3D(x_norm, y_norm, z_sign * 0.5).normalize()
                ray = Ray(center, direction)

                # Use the provided intersection function with all arguments
                result1, result2 = intersection_func(ray, center, radius, *args)

                for result in [result1, result2]:
                    if result is not None and result.intersects:
                        point = result.point
                        points.append([point.x, point.y, point.z])

    # Remove duplicates and convert to array
    if points:
        points = np.unique(np.array(points), axis=0)
    else:
        points = np.array([])

    return points
